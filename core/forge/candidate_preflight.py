import re
from typing import Any

from core.forge.contracts import FeasiblePlan
from core.forge.fixture_oracle import (
    fixture_oracle_capability,
    fixture_oracle_mismatches,
)
from core.forge.requirement_evidence import requirement_assertion_evidence
from core.forge.semantic_contracts import (
    behaviorally_evidences,
    has_canonicalized_deduplication_assertion,
    has_expected_exception_assertion,
    has_json_lines_processing,
    interface_parameter_is_exercised,
    semantic_term_present,
    structurally_evidences,
)
from core.forge.test_evidence import (
    non_semantic_test_reasons,
    source_module_names,
)
from core.forge.validation.obligations import ObligationValidationLayer


def run_fixture_oracle_preflight(
    candidate_files: dict[str, str],
    plan: FeasiblePlan,
    contracts: dict[str, Any],
) -> dict[str, Any]:
    requirement = plan.build_spec.normalized_requirement
    capability_id = fixture_oracle_capability(requirement)
    if capability_id is None:
        return {"phase": "fixture_oracle", "passed": True, "failures": []}

    requirement_id = next(
        (
            atom.requirement_id
            for atom in plan.build_spec.requirement_atoms
            if fixture_oracle_capability(atom.text) == capability_id
        ),
        "",
    )
    failures: list[dict[str, Any]] = []
    for path in sorted(contracts):
        if not path.startswith("tests/") or path not in candidate_files:
            continue
        for mismatch in fixture_oracle_mismatches(candidate_files[path], requirement):
            failures.append(
                {
                    "path": path,
                    "kind": "fixture_oracle_mismatch",
                    "requirement_id": requirement_id,
                    **mismatch.to_evidence(),
                }
            )
    if not failures:
        return {"phase": "fixture_oracle", "passed": True, "failures": []}
    failed_paths = list(dict.fromkeys(str(item["path"]) for item in failures))
    return {
        "phase": "fixture_oracle",
        "ran": False,
        "passed": False,
        "failed_paths": failed_paths,
        "source_failed_paths": [],
        "test_failed_paths": failed_paths,
        "failures": failures,
        "correction_requirements": [
            (
                f"{failure['path']}:{failure['function']}: replace the manually transcribed "
                f"{failure['expected_name']} oracle with the source-independent {capability_id} "
                f"result {failure['derived_expected']} derived from {failure['input_name']}; do not "
                "call generated target code to compute the expectation."
            )
            for failure in failures
        ],
    }


def run_semantic_preflight(
    candidate_files: dict[str, str],
    plan: FeasiblePlan,
    contracts: dict[str, Any],
    executable_preflight: dict[str, Any],
) -> dict[str, Any]:
    test_failures, failed_paths, atoms_by_id = _test_contract_failures(
        candidate_files,
        plan,
        contracts,
    )
    test_failures.extend(
        _non_semantic_test_failures(
            candidate_files,
            plan,
            contracts,
            failed_paths,
        )
    )
    source_failures = _source_contract_failures(
        candidate_files,
        plan,
        failed_paths,
    )
    if not test_failures and not source_failures:
        return executable_preflight
    return {
        **executable_preflight,
        "phase": "semantic_contract",
        "passed": False,
        "executable_passed": True,
        "failed_paths": sorted(failed_paths),
        "source_failed_paths": sorted(path for path in failed_paths if path.startswith("src/")),
        "test_failed_paths": sorted(path for path in failed_paths if path.startswith("tests/")),
        "failures": [*test_failures, *source_failures],
        "correction_requirements": correction_requirements(
            test_failures,
            source_failures,
            atoms_by_id,
        ),
    }


def _test_contract_failures(
    candidate_files: dict[str, str],
    plan: FeasiblePlan,
    contracts: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    failed_paths: list[str] = []
    atoms_by_id = {
        atom.requirement_id: atom
        for atom in plan.build_spec.requirement_atoms
    }
    mapped_test_paths = {
        f"tests/{planned_test.test_name}.py"
        for planned_test in plan.required_tests
        if planned_test.required
    }
    public_interface_names = {
        interface.name
        for interface in plan.interfaces
        if interface.name.isidentifier()
    }
    plan_requires_byte_exact_observation = requires_byte_exact_observation(
        plan.build_spec.normalized_requirement
    )
    for path, contract in contracts.items():
        if path not in mapped_test_paths:
            continue
        content = candidate_files.get(path, "")
        for requirement in contract.get("requirements", []):
            requirement_id = str(requirement.get("id", ""))
            atom = atoms_by_id.get(requirement_id)
            evidence_terms = [
                str(term) for term in requirement.get("evidence_terms", [])
            ]
            local_requirement_text = " ".join(
                [getattr(atom, "text", ""), *evidence_terms]
            )
            byte_exact_required = requires_byte_exact_observation(
                local_requirement_text
            ) or (
                plan_requires_byte_exact_observation
                and has_line_boundary_signal(local_requirement_text)
            )
            if byte_exact_required and not has_byte_exact_test_observation(content):
                if path not in failed_paths:
                    failed_paths.append(path)
                failures.append(
                    {
                        "path": path,
                        "kind": (
                            "lossy_observation_api"
                            if ".read_text(" in content
                            else "missing_byte_exact_observation"
                        ),
                        "requirement_id": requirement_id,
                        "observation_api": (
                            "Path.read_text"
                            if ".read_text(" in content
                            else "none"
                        ),
                        "required_observation": "byte_exact",
                    }
                )
            coverage = plan.requirement_coverage.get(requirement_id, {})
            source_content = "\n\n".join(
                candidate_files[source_path]
                for source_path in coverage.get("files", [])
                if source_path.startswith("src/") and source_path in candidate_files
            )
            missing_terms = [
                term
                for term in requirement.get("evidence_terms", [])
                if not ObligationValidationLayer._semantic_term_present(
                    str(term),
                    content.lower(),
                    is_test=True,
                )
                and not behaviorally_evidences(
                    str(term),
                    content,
                    public_interface_names,
                )
                and not interface_parameter_is_exercised(
                    str(term),
                    content,
                    source_content,
                    public_interface_names,
                )
            ]
            expects_exception = requires_exception_rejection(atom)
            has_expected_exception = has_expected_exception_assertion(
                content,
                public_interface_names,
            )
            expects_canonicalized_deduplication = requires_canonicalized_deduplication(atom)
            has_canonicalized_deduplication = has_canonicalized_deduplication_assertion(content)
            if (
                missing_terms
                or (expects_exception and not has_expected_exception)
                or (
                    expects_canonicalized_deduplication
                    and not has_canonicalized_deduplication
                )
            ):
                if path not in failed_paths:
                    failed_paths.append(path)
                failures.append(
                    {
                        "path": path,
                        "kind": "semantic_contract_failure",
                        "requirement_id": requirement_id,
                        "missing_evidence_terms": missing_terms,
                        "expected_exception_missing": expects_exception and not has_expected_exception,
                        "canonicalized_deduplication_missing": (
                            expects_canonicalized_deduplication
                            and not has_canonicalized_deduplication
                        ),
                    }
                )
    requirement_terms: dict[str, list[str]] = {}
    requirement_test_paths: dict[str, list[str]] = {}
    for path, contract in contracts.items():
        if path not in mapped_test_paths:
            continue
        for requirement in contract.get("requirements", []):
            requirement_id = str(requirement.get("id", ""))
            if not requirement_id:
                continue
            requirement_terms.setdefault(requirement_id, []).extend(
                str(term) for term in requirement.get("evidence_terms", [])
            )
            requirement_test_paths.setdefault(requirement_id, []).append(path)

    assertion_report = requirement_assertion_evidence(
        requirement_terms,
        requirement_test_paths,
        candidate_files,
        target_names=public_interface_names,
        target_modules=source_module_names(candidate_files),
        term_matcher=lambda term, function_source: (
            semantic_term_present(term, function_source, is_test=True)
            or behaviorally_evidences(
                term,
                function_source,
                public_interface_names,
            )
        ),
    )
    for requirement_id, evidence in assertion_report.items():
        if evidence["passed"]:
            continue
        paths = list(evidence["existing_test_paths"] or evidence["mapped_test_paths"])
        for path in paths:
            if path not in failed_paths:
                failed_paths.append(path)
        failures.append(
            {
                "path": paths[0] if paths else "",
                "kind": "requirement_assertion_evidence_failure",
                "requirement_id": requirement_id,
                "missing_evidence_terms": list(evidence["missing_terms"]),
                "failure_reason": evidence["failure_reason"],
                "assertion_evidence": evidence,
            }
        )
    return failures, failed_paths, atoms_by_id


def _non_semantic_test_failures(
    candidate_files: dict[str, str],
    plan: FeasiblePlan,
    contracts: dict[str, Any],
    failed_paths: list[str],
) -> list[dict[str, Any]]:
    mapped_paths = [path for path in contracts if path.startswith("tests/")]
    failures: list[dict[str, Any]] = []
    reasons_by_path = non_semantic_test_reasons(
        mapped_paths,
        candidate_files,
        target_names={
            interface.name
            for interface in plan.interfaces
            if interface.name.isidentifier()
        },
        target_modules=source_module_names(candidate_files),
    )
    for path, reasons in reasons_by_path.items():
        if path not in failed_paths:
            failed_paths.append(path)
        requirement_ids = [
            str(requirement.get("id", ""))
            for requirement in contracts.get(path, {}).get("requirements", [])
            if str(requirement.get("id", ""))
        ]
        failures.append(
            {
                "path": path,
                "kind": "non_semantic_test",
                "requirement_ids": requirement_ids,
                "reasons": reasons,
            }
        )
    return failures


def _source_contract_failures(
    candidate_files: dict[str, str],
    plan: FeasiblePlan,
    failed_paths: list[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    public_interface_names = {
        interface.name
        for interface in plan.interfaces
        if interface.name.isidentifier()
    }
    for atom in plan.build_spec.requirement_atoms:
        if atom.category == "ambiguity" or not atom.evidence_terms:
            continue
        if not ObligationValidationLayer._requires_source_semantic_evidence(atom):
            continue
        coverage = plan.requirement_coverage.get(atom.requirement_id, {})
        source_paths = [
            path
            for path in coverage.get("files", [])
            if path.startswith("src/") and path in candidate_files
        ]
        if not source_paths:
            continue
        source_corpus = "\n".join(candidate_files[path].lower() for path in source_paths)
        source_content = "\n\n".join(candidate_files[path] for path in source_paths)
        test_corpus = "\n\n".join(
            candidate_files[path]
            for path in (
                f"tests/{test_name}.py"
                for test_name in coverage.get("tests", [])
            )
            if path in candidate_files
        )
        missing_terms = [
            term
            for term in atom.evidence_terms
            if not ObligationValidationLayer._semantic_term_present(term, source_corpus)
            and not structurally_evidences(term, source_content, plan.interfaces)
            and not (
                term in {"jsonl", "input_jsonl"}
                and has_json_lines_processing(source_content)
            )
            and not behaviorally_evidences(term, test_corpus, public_interface_names)
        ]
        if missing_terms:
            for path in source_paths:
                if path not in failed_paths:
                    failed_paths.append(path)
            failures.append(
                {
                    "paths": source_paths,
                    "kind": "source_semantic_contract_failure",
                    "requirement_id": atom.requirement_id,
                    "missing_evidence_terms": missing_terms,
                }
            )
    return failures


def correction_requirements(
    test_failures: list[dict[str, Any]],
    source_failures: list[dict[str, Any]],
    atoms_by_id: dict[str, Any],
) -> list[str]:
    requirements: list[str] = []
    for failure in test_failures:
        path = str(failure.get("path", ""))
        if failure.get("kind") in {
            "lossy_observation_api",
            "missing_byte_exact_observation",
        }:
            requirement_id = str(failure.get("requirement_id", ""))
            requirements.append(
                f"{path}: observe exact output bytes for requirement {requirement_id} with "
                "Path.read_bytes(), binary mode, or Path.open(encoding='utf-8', newline=''); "
                "Path.read_text() performs universal-newline translation and cannot prove preservation "
                "of CRLF, LF, or CR sequences."
            )
            continue
        if failure.get("kind") == "requirement_assertion_evidence_failure":
            requirement_id = str(failure.get("requirement_id", ""))
            atom = atoms_by_id.get(requirement_id)
            atom_text = atom.text if atom is not None else requirement_id
            missing_terms = [
                str(term) for term in failure.get("missing_evidence_terms", [])
            ]
            requirements.append(
                f"{path}: add a target-dependent assertion in the same test function that exercises "
                f"semantic evidence {missing_terms} for requirement {requirement_id}: {atom_text}"
            )
            continue
        if failure.get("kind") == "non_semantic_test":
            requirement_ids = [str(item) for item in failure.get("requirement_ids", [])]
            reasons = [str(item) for item in failure.get("reasons", [])]
            if "disconnected_assertion" in reasons:
                requirements.append(
                    f"{path}: assertions must observe values returned by a declared public interface, "
                    "arguments mutated by that interface, or file/stdout/database effects read after its "
                    f"invocation for requirements {requirement_ids}."
                )
                continue
            if "missing_target_invocation" in reasons:
                requirements.append(
                    f"{path}: invoke a declared generated public interface and assert an observable "
                    f"behavioral result for requirements {requirement_ids}."
                )
                continue
            requirements.append(
                f"{path}: replace callable/type/file-presence or placeholder checks with a test that invokes "
                "a declared public interface using concrete input and asserts an observable behavioral result "
                f"for requirements {requirement_ids}."
            )
            continue
        requirement_id = str(failure.get("requirement_id", ""))
        atom = atoms_by_id.get(requirement_id)
        atom_text = atom.text if atom is not None else requirement_id
        missing_terms = [str(term) for term in failure.get("missing_evidence_terms", [])]
        if "cli_entrypoint" in missing_terms:
            requirements.append(
                f"{path}: invoke the declared CLI entrypoint main(argv) with explicit temporary input/output paths "
                f"and assert its observable behavior for requirement {requirement_id}: {atom_text}"
            )
        remaining = [term for term in missing_terms if term != "cli_entrypoint"]
        if remaining:
            requirements.append(
                f"{path}: exercise and assert semantic evidence {remaining} for requirement "
                f"{requirement_id}: {atom_text}"
            )
        if failure.get("expected_exception_missing"):
            requirements.append(
                f"{path}: wrap the invalid-input call to main(argv) in "
                "pytest.raises((ValueError, TypeError, SystemExit)); the source implementation must raise "
                f"one of those exceptions for requirement {requirement_id}: {atom_text}"
            )
        if failure.get("canonicalized_deduplication_missing"):
            requirements.append(
                f"{path}: call deduplicate_emails with values that differ by surrounding whitespace "
                "and letter case, then assert that the returned first-seen values are trimmed and "
                f"lowercased canonical addresses for requirement {requirement_id}: {atom_text}"
            )
    for failure in source_failures:
        paths = ", ".join(str(path) for path in failure.get("paths", []))
        requirement_id = str(failure.get("requirement_id", ""))
        atom = atoms_by_id.get(requirement_id)
        atom_text = atom.text if atom is not None else requirement_id
        missing_terms = [str(term) for term in failure.get("missing_evidence_terms", [])]
        requirements.append(
            f"{paths}: implement semantic evidence {missing_terms} as executable behavior for requirement "
            f"{requirement_id}: {atom_text}"
        )
    return list(dict.fromkeys(requirements))


def requires_exception_rejection(atom: Any) -> bool:
    if atom is None or atom.category != "validation":
        return False
    text = atom.text.lower()
    rejects = bool(re.search(r"\b(?:reject|rejects|rejected)\b", text))
    boundary_type = any(
        token in text
        for token in ("root", "argument type", "input type", "unsupported type")
    )
    operational_handling = any(
        token in text
        for token in ("quarantine", "skip", "report", "log")
    )
    return rejects and boundary_type and not operational_handling


def requires_canonicalized_deduplication(atom: Any) -> bool:
    if atom is None:
        return False
    normalized = " ".join(atom.text.lower().replace("-", " ").split())
    return (
        "deduplic" in normalized
        and "canonical" in normalized
        and "first seen order" in normalized
    )


def has_line_boundary_signal(text: str) -> bool:
    normalized = " ".join(str(text).lower().replace("-", " ").split())
    return any(
        signal in normalized
        for signal in (
            "line ending",
            "mixed line",
            "newline",
            "crlf",
            "carriage return",
        )
    )


def requires_byte_exact_observation(text: str) -> bool:
    normalized = " ".join(str(text).lower().replace("-", " ").split())
    exactness = any(
        signal in normalized
        for signal in (
            "preserv",
            "unchanged",
            "exact byte",
            "byte exact",
        )
    )
    return exactness and has_line_boundary_signal(normalized)


def has_byte_exact_test_observation(content: str) -> bool:
    if ".read_bytes(" in content:
        return True
    if re.search(r"(?:\.|\b)open\([^\n)]*['\"]rb['\"]", content):
        return True
    return bool(
        re.search(r"(?:\.|\b)open\([^\n)]*newline\s*=\s*['\"]['\"]", content)
        and re.search(r"\.read\(", content)
    )
