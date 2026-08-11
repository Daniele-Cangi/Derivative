import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

from core.forge.contracts import BuildSpec, CodeArtifact, FeasiblePlan, ValidationLayerResult
from core.forge.validation.common import ValidationLayerBase
from core.forge.validation.capabilities import CapabilityContractChecker
from core.forge.validation.quality import QualityContractChecker


class ObligationValidationLayer(ValidationLayerBase):
    def __init__(
        self,
        python_executable: str,
        timeout_seconds: int,
        quality_checker: QualityContractChecker,
    ):
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.quality_checker = quality_checker
        self.capability_checker = CapabilityContractChecker()

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

        test_result = self._run_required_tests(workspace, expected_test_paths, actual_paths)
        evidence["test_execution"] = test_result
        if not test_result["ran"]:
            failures.append("Required tests were not executed.")
            self._append_unique(signatures, "test_execution_failure")
        elif test_result["returncode"] != 0:
            failures.append("Required test execution failed.")
            self._append_unique(signatures, "test_execution_failure")

        evidence["failure_signatures"] = signatures
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        passed = len(failures) == 0
        metrics = {
            "duration_ms": elapsed_ms,
            "required_file_count": len(required_paths),
            "required_test_count": len(expected_test_paths),
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
            self.python_executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--basetemp={workspace / '.pytest_tmp'}",
            *runnable,
        ]
        completed = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        return {
            "ran": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "tests": runnable,
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
            if atom.strength == "universal" and atom.requirement_id not in universal_proofs:
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
