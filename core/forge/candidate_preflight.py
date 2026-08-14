import re
from typing import Any

from core.forge.contracts import FeasiblePlan
from core.forge.semantic_contracts import (
    behaviorally_evidences,
    has_canonicalized_deduplication_assertion,
    has_expected_exception_assertion,
)
from core.forge.validation.obligations import ObligationValidationLayer


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
    for path, contract in contracts.items():
        if path not in mapped_test_paths:
            continue
        content = candidate_files.get(path, "")
        for requirement in contract.get("requirements", []):
            requirement_id = str(requirement.get("id", ""))
            atom = atoms_by_id.get(requirement_id)
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
    return failures, failed_paths, atoms_by_id


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
