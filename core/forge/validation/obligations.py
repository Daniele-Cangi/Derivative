import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from core.forge.contracts import BuildSpec, CodeArtifact, FeasiblePlan, ValidationLayerResult
from core.forge.execution import ProcessExecutor, SandboxProcessRequest
from core.forge.semantic_contracts import (
    behaviorally_evidences,
    has_json_lines_processing,
    interface_parameter_is_exercised,
)
from core.forge.validation.common import ValidationLayerBase
from core.forge.validation.adapter_capabilities import AdapterCapabilityContractChecker
from core.forge.validation.capabilities import CapabilityContractChecker
from core.forge.validation.quality import QualityContractChecker


class ObligationValidationLayer(ValidationLayerBase):
    def __init__(
        self,
        executor: ProcessExecutor,
        timeout_seconds: int,
        quality_checker: QualityContractChecker,
    ):
        self.executor = executor
        self.timeout_seconds = timeout_seconds
        self.quality_checker = quality_checker
        self.capability_checker = CapabilityContractChecker()
        self.adapter_capability_checker = AdapterCapabilityContractChecker()

    def validate(
        self,
        code_artifact: CodeArtifact,
        plan: FeasiblePlan,
        build_spec: BuildSpec,
        materialized: Dict[str, Path],
        workspace: Path,
    ) -> ValidationLayerResult:
        started = time.perf_counter()
        failures: List[str] = []
        signatures: List[str] = []
        evidence: Dict[str, object] = {}

        material_ambiguities = [
            flag
            for flag in build_spec.ambiguity_flags
            if "materially unspecified" in flag.lower()
        ]
        if material_ambiguities:
            failures.append(
                "Requirement contains material ambiguities that cannot be validated without inventing policy: "
                f"{material_ambiguities}."
            )
            self._append_unique(signatures, "underspecified_requirement")
        evidence["material_ambiguities"] = material_ambiguities

        actual_paths = set(materialized.keys())
        required_paths = {plan_file.path for plan_file in plan.file_tree_plan}
        missing_required_files = sorted(required_paths - actual_paths)
        if missing_required_files:
            failures.append(f"Missing required plan files: {missing_required_files}.")
            self._append_unique(signatures, "missing_required_file")
        evidence["missing_required_files"] = missing_required_files

        expected_test_paths = {f"tests/{test.test_name}.py" for test in plan.required_tests if test.required}
        artifact_test_paths = set(code_artifact.test_paths)
        missing_required_tests = sorted(expected_test_paths - artifact_test_paths)
        if missing_required_tests:
            failures.append(f"Required tests are missing from artifact.test_paths: {missing_required_tests}.")
            self._append_unique(signatures, "missing_acceptance_coverage")
        evidence["missing_required_tests"] = missing_required_tests

        manifest_required_obligations = set(
            code_artifact.artifact_manifest.get("required_obligations", [])
            if isinstance(code_artifact.artifact_manifest, dict)
            else []
        )
        missing_manifest_obligations = sorted(set(plan.required_obligations) - manifest_required_obligations)
        if missing_manifest_obligations:
            failures.append(
                "Manifest does not declare required obligations: "
                f"{missing_manifest_obligations}."
            )
            self._append_unique(signatures, "missing_obligation")
        evidence["missing_manifest_obligations"] = missing_manifest_obligations

        provenance_obligations = self._collect_prefixed_tokens(code_artifact, prefix="obligation:")
        missing_provenance_obligations = sorted(set(plan.required_obligations) - provenance_obligations)
        if missing_provenance_obligations:
            failures.append(
                "Provenance does not cover required obligations: "
                f"{missing_provenance_obligations}."
            )
            self._append_unique(signatures, "missing_obligation")
        evidence["missing_provenance_obligations"] = missing_provenance_obligations

        required_acceptance = set(plan.acceptance_criterion_ids)
        provenance_acceptance = self._collect_prefixed_tokens(code_artifact, prefix="acceptance:")
        missing_acceptance = sorted(required_acceptance - provenance_acceptance)
        if missing_acceptance:
            failures.append(f"Missing acceptance provenance coverage: {missing_acceptance}.")
            self._append_unique(signatures, "missing_acceptance_coverage")
        evidence["missing_acceptance_coverage"] = missing_acceptance

        requirement_failures, requirement_signatures, requirement_evidence = self._validate_requirement_coverage(
            build_spec=build_spec,
            plan=plan,
            code_artifact=code_artifact,
        )
        failures.extend(requirement_failures)
        for signature in requirement_signatures:
            self._append_unique(signatures, signature)
        evidence["requirement_coverage_checks"] = requirement_evidence

        semantic_failures, semantic_signatures, semantic_evidence = self._validate_requirement_semantics(
            build_spec=build_spec,
            plan=plan,
            code_artifact=code_artifact,
        )
        failures.extend(semantic_failures)
        for signature in semantic_signatures:
            self._append_unique(signatures, signature)
        evidence["requirement_semantic_checks"] = semantic_evidence

        quality_contract_failures, quality_contract_evidence = self.quality_checker.check(
            materialized=materialized,
            code_artifact=code_artifact,
            build_spec=build_spec,
        )
        if quality_contract_failures:
            failures.extend(quality_contract_failures)
            self._append_unique(signatures, "quality_contract_violation")
        evidence["quality_contract_checks"] = quality_contract_evidence

        capability_failures, capability_signatures, capability_evidence = self.capability_checker.check(
            code_artifact=code_artifact,
            plan=plan,
            materialized=materialized,
        )
        failures.extend(capability_failures)
        for signature in capability_signatures:
            self._append_unique(signatures, signature)
        evidence["capability_contract_checks"] = capability_evidence

        adapter_failures, adapter_signatures, adapter_evidence = (
            self.adapter_capability_checker.check(
                code_artifact=code_artifact,
                plan=plan,
            )
        )
        failures.extend(adapter_failures)
        for signature in adapter_signatures:
            self._append_unique(signatures, signature)
        evidence["adapter_capability_checks"] = adapter_evidence

        declared_test_paths = set(code_artifact.test_paths)
        test_result = self._run_required_tests(
            workspace,
            expected_test_paths | declared_test_paths,
            actual_paths,
        )
        evidence["test_execution"] = test_result
        evidence["declared_test_paths"] = sorted(declared_test_paths)
        if not test_result["ran"]:
            failures.append("Required tests were not executed.")
            self._append_unique(signatures, "test_execution_failure")
        elif test_result["returncode"] != 0:
            failures.append("Required test execution failed.")
            self._append_unique(
                signatures,
                "sandbox_unavailable" if test_result.get("launch_error") else "test_execution_failure",
            )

        evidence["failure_signatures"] = signatures
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        passed = len(failures) == 0
        metrics = {
            "duration_ms": elapsed_ms,
            "required_file_count": len(required_paths),
            "required_test_count": len(expected_test_paths),
            "declared_test_count": len(declared_test_paths),
            "required_obligation_count": len(plan.required_obligations),
            "required_acceptance_count": len(plan.acceptance_criterion_ids),
            "build_id": build_spec.build_id,
        }
        return ValidationLayerResult(
            layer_name="layer2_obligations_tests_acceptance",
            passed=passed,
            failures=failures,
            evidence=evidence,
            metrics=metrics,
        )

    def _run_required_tests(
        self,
        workspace: Path,
        expected_test_paths: set[str],
        actual_paths: set[str],
    ) -> Dict[str, object]:
        runnable = sorted(path for path in expected_test_paths if path in actual_paths)
        if not runnable:
            return {
                "ran": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "tests": [],
            }
        command = [
            "python",
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp=.pytest_tmp",
            *runnable,
        ]
        completed = self.executor.run(
            SandboxProcessRequest(
                command=command,
                workspace=workspace,
                timeout_seconds=self.timeout_seconds,
                environment={
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                },
            )
        )
        return {
            "ran": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "tests": runnable,
            "backend": completed.backend,
            "timed_out": completed.timed_out,
            "launch_error": completed.launch_error,
            "isolation": completed.isolation,
        }

    def _collect_prefixed_tokens(self, artifact: CodeArtifact, prefix: str) -> set[str]:
        tokens: set[str] = set()
        for generated in artifact.files:
            for section in generated.generated_from_plan_sections:
                if section.startswith(prefix):
                    tokens.add(section[len(prefix):])
        return tokens

    def _validate_requirement_coverage(
        self,
        build_spec: BuildSpec,
        plan: FeasiblePlan,
        code_artifact: CodeArtifact,
    ) -> Tuple[List[str], List[str], Dict[str, object]]:
        failures: List[str] = []
        signatures: List[str] = []
        evidence: Dict[str, object] = {"requirements": {}}

        atoms = [atom for atom in build_spec.requirement_atoms if atom.category != "ambiguity"]
        artifact_requirement_ids = self._collect_prefixed_tokens(code_artifact, prefix="requirement:")
        universal_proofs = self._collect_prefixed_tokens(code_artifact, prefix="universal_proof:")
        acceptance_requirement_ids = set()
        for criterion in build_spec.acceptance_contract.criteria:
            acceptance_requirement_ids.update(criterion.requirement_ids)

        semantic_omissions: List[str] = []
        missing_coverage: List[str] = []
        universal_unproven: List[str] = []
        for atom in atoms:
            coverage_entry = plan.requirement_coverage.get(
                atom.requirement_id,
                {"files": [], "tests": [], "acceptance_criteria": []},
            )
            files = list(coverage_entry.get("files", []))
            tests = list(coverage_entry.get("tests", []))
            acceptance = list(coverage_entry.get("acceptance_criteria", []))

            has_plan_mapping = bool(files or tests or acceptance)
            has_artifact_mapping = atom.requirement_id in artifact_requirement_ids
            has_acceptance_mapping = atom.requirement_id in acceptance_requirement_ids and bool(acceptance)
            has_test_mapping = bool(tests)

            evidence["requirements"][atom.requirement_id] = {
                "text": atom.text,
                "category": atom.category,
                "strength": atom.strength,
                "files": files,
                "tests": tests,
                "acceptance_criteria": acceptance,
                "has_plan_mapping": has_plan_mapping,
                "has_artifact_mapping": has_artifact_mapping,
                "has_acceptance_mapping": has_acceptance_mapping,
            }

            if not has_plan_mapping or not has_artifact_mapping:
                semantic_omissions.append(atom.requirement_id)
            if not has_test_mapping or not has_acceptance_mapping:
                missing_coverage.append(atom.requirement_id)
            if (
                atom.category == "universal_constraint" or atom.strength == "universal"
            ) and atom.requirement_id not in universal_proofs:
                universal_unproven.append(atom.requirement_id)

        if semantic_omissions:
            failures.append(f"Semantic omission detected for requirements: {semantic_omissions}.")
            self._append_unique(signatures, "semantic_omission")
        if missing_coverage:
            failures.append(f"Missing requirement coverage for requirements: {missing_coverage}.")
            self._append_unique(signatures, "missing_requirement_coverage")
        if universal_unproven:
            failures.append(
                "Universal constraints are unproven and fail closed: "
                f"{universal_unproven}."
            )
            self._append_unique(signatures, "universal_constraint_unproven")

        evidence["semantic_omissions"] = semantic_omissions
        evidence["missing_coverage"] = missing_coverage
        evidence["universal_unproven"] = universal_unproven
        return failures, signatures, evidence

    def _validate_requirement_semantics(
        self,
        build_spec: BuildSpec,
        plan: FeasiblePlan,
        code_artifact: CodeArtifact,
    ) -> Tuple[List[str], List[str], Dict[str, object]]:
        files_by_path = {generated.path: generated for generated in code_artifact.files}
        mismatches: List[Dict[str, object]] = []
        checks: Dict[str, object] = {}

        for atom in build_spec.requirement_atoms:
            if atom.category == "ambiguity" or not atom.evidence_terms:
                continue
            coverage = plan.requirement_coverage.get(atom.requirement_id, {})
            source_paths = [
                path
                for path in coverage.get("files", [])
                if path.startswith("src/") and path in files_by_path
            ]
            test_paths = [
                f"tests/{name}.py"
                for name in coverage.get("tests", [])
                if f"tests/{name}.py" in files_by_path
            ]
            source_corpus = "\n".join(files_by_path[path].content.lower() for path in source_paths)
            source_content = "\n\n".join(
                files_by_path[path].content for path in source_paths
            )
            test_corpus = "\n".join(files_by_path[path].content.lower() for path in test_paths)
            behavioral_test_corpus = "\n\n".join(
                files_by_path[path].content for path in test_paths
            )
            public_interface_names = {
                interface.name
                for interface in plan.interfaces
                if interface.name.isidentifier()
            }
            source_evidence_required = self._requires_source_semantic_evidence(atom)
            missing_source_terms = (
                [
                    term
                    for term in atom.evidence_terms
                    if not self._semantic_term_present(term, source_corpus)
                    and not (
                        term in {"jsonl", "input_jsonl"}
                        and has_json_lines_processing(source_content)
                    )
                    and not behaviorally_evidences(
                        term,
                        behavioral_test_corpus,
                        public_interface_names,
                    )
                ]
                if source_evidence_required
                else []
            )
            missing_test_terms = [
                term
                for term in atom.evidence_terms
                if not self._semantic_term_present(term, test_corpus, is_test=True)
                and not behaviorally_evidences(
                    term,
                    behavioral_test_corpus,
                    public_interface_names,
                )
                and not interface_parameter_is_exercised(
                    term,
                    behavioral_test_corpus,
                    "\n\n".join(
                        files_by_path[path].content for path in source_paths
                    ),
                    public_interface_names,
                )
            ]
            item = {
                "text": atom.text,
                "evidence_terms": list(atom.evidence_terms),
                "source_paths": source_paths,
                "test_paths": test_paths,
                "source_evidence_required": source_evidence_required,
                "missing_source_terms": missing_source_terms,
                "missing_test_terms": missing_test_terms,
            }
            checks[atom.requirement_id] = item
            if missing_source_terms or missing_test_terms:
                mismatches.append({"requirement_id": atom.requirement_id, **item})

        failures: List[str] = []
        signatures: List[str] = []
        source_mismatches = [item for item in mismatches if item["missing_source_terms"]]
        test_mismatches = [item for item in mismatches if item["missing_test_terms"]]
        if source_mismatches:
            failed_ids = [str(item["requirement_id"]) for item in source_mismatches]
            failures.append(
                "Requirement content is not evidenced by generated source code: "
                f"{failed_ids}."
            )
            signatures.extend(["semantic_omission", "semantic_content_mismatch"])
        if test_mismatches:
            failed_ids = [str(item["requirement_id"]) for item in test_mismatches]
            failures.append(
                "Requirement content is not evidenced by mapped behavioral tests: "
                f"{failed_ids}."
            )
            self._append_unique(signatures, "missing_semantic_requirement_coverage")
        return failures, signatures, {
            "requirements": checks,
            "semantic_content_mismatches": mismatches,
            "source_semantic_mismatches": source_mismatches,
            "test_semantic_mismatches": test_mismatches,
        }

    @staticmethod
    def _requires_source_semantic_evidence(atom) -> bool:
        if atom.category != "validation":
            return True
        normalized = " ".join(atom.text.lower().replace("-", " ").split())
        test_obligation_patterns = (
            r"\bincludes?\s+(?:behavioral\s+|integration\s+|end\s+to\s+end\s+|unit\s+)?tests?\b",
            r"\btests?\s+for\b",
            r"\btests?\s+(?:that\s+)?(?:cover|covers|verify|verifies|exercise|exercises)\b",
        )
        return not any(re.search(pattern, normalized) for pattern in test_obligation_patterns)

    @staticmethod
    def _semantic_term_present(term: str, corpus: str, is_test: bool = False) -> bool:
        normalized = corpus.replace("-", "_").replace(" ", "_")
        aliases = {
            "cli_entrypoint": ("argparse", "typer", "click.command", "sys.argv"),
            "cli_flow": ("argparse", "typer", "click.command", "sys.argv"),
            "input_jsonl": ("json.loads", "json.load(", "jsonlines", "parse_jsonl"),
            "jsonl": (".jsonl", "json_lines", "jsonl"),
            "input_csv": ("csv.dictreader", "csv.reader", "read_csv"),
            "summary_csv": ("csv.dictwriter", "csv.writer(", "write_summary"),
            "malformed_records": ("jsondecodeerror", "malformed", "invalid_json"),
            "duplicate_ids": ("duplicate_id", "duplicate_ids"),
            "missing_fields": ("missing_field", "missing_fields"),
            "invalid_timestamp": ("invalid_timestamp", "timestamp_error"),
            "invalid_dates": ("invalid_date", "invalid_dates"),
            "malformed_rows": ("malformed_row", "malformed_rows"),
            "quarantine": ("quarantine", "quarantined"),
            "minimum": ("min(", "minimum", "min_temperature", "['min']", '"min"'),
            "maximum": ("max(", "maximum", "max_temperature", "['max']", '"max"'),
            "average": ("average", "avg_temperature", "mean(", "['average']", "['avg']", '"average"'),
            "aggregation": (
                "aggregate",
                "aggregation",
                "groupby",
                "group_by",
                "compute_summary",
                "device_temps",
            ),
            "per_device": ("device_id", "per_device"),
            "per_customer": ("customer_id", "per_customer"),
            "summary_json": ("json.dumps", "json.dump(", "summary_json", ".json"),
            "idempotent_event": (
                "insert_or_ignore",
                "on_conflict",
                "idempotent",
                "event_id_text_primary_key",
            ),
            "totals": ("total", "totals"),
            "counts": ("count", "counts"),
            "recursive_json_merge": (
                "recursive_json_merge",
                "recursive_merge",
                "deep_merge",
            ),
            "json_list_replacement": (
                "json_list_replacement",
                "replace_lists",
                "replaces_json_lists",
                "lists_are_replaced",
                "list_replacement",
                "lists_instead_of_concatenating",
            ),
            "json_object_root_validation": (
                "json_object_root_validation",
                "validate_object_root",
                "non_object_root",
                "rejects_non_object",
                "root_must_be_an_object",
                "json_root_must_be_an_object",
                "root_element_must_be_an_object",
            ),
        }
        candidates = aliases.get(term, (term,))
        if is_test and term == "input_jsonl":
            candidates = (*candidates, ".jsonl")
        if is_test and term == "input_csv":
            candidates = (*candidates, ".csv")
        if is_test and term in {"cli_entrypoint", "cli_flow"}:
            candidates = (*candidates, ".main(", "clirunner", "subprocess.run")
        return any(candidate in corpus or candidate in normalized for candidate in candidates)
