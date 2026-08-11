import importlib
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from core.forge.contracts import (
    BuildSpec,
    CodeArtifact,
    FailureCategory,
    FeasiblePlan,
    ValidationArtifact,
    ValidationLayerResult,
)
from core.forge.validation.adversarial import AdversarialValidationLayer
from core.forge.validation.obligations import ObligationValidationLayer
from core.forge.validation.quality import QualityContractChecker
from core.forge.validation.runtime import RuntimeValidationLayer


class ValidatorStage:
    def __init__(self, python_executable: str | None = None, timeout_seconds: int = 120):
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = timeout_seconds
        quality_checker = QualityContractChecker()
        self.runtime_layer = RuntimeValidationLayer(self.python_executable, timeout_seconds)
        self.obligation_layer = ObligationValidationLayer(
            self.python_executable,
            timeout_seconds,
            quality_checker,
        )
        self.adversarial_layer = AdversarialValidationLayer()

    def validate(
        self,
        code_artifact: CodeArtifact,
        plan: FeasiblePlan,
        build_spec: BuildSpec,
    ) -> ValidationArtifact:
        failures: List[str] = []
        signatures: List[str] = []
        evidence: Dict[str, object] = {}
        metrics: Dict[str, object] = {}

        with tempfile.TemporaryDirectory(prefix="forge_validator_") as tmp_dir:
            workspace = Path(tmp_dir)
            materialized = self._materialize_workspace(code_artifact, workspace)
            evidence["workspace"] = str(workspace)
            evidence["materialized_files"] = sorted(str(path) for path in materialized.values())

            layer1 = self.runtime_layer.validate(
                code_artifact,
                plan,
                build_spec,
                materialized,
                workspace,
            )
            layer2 = self.obligation_layer.validate(
                code_artifact,
                plan,
                build_spec,
                materialized,
                workspace,
            )
            layer3 = self.adversarial_layer.validate(
                code_artifact,
                plan,
                build_spec,
                materialized,
                workspace,
            )

            for layer in (layer1, layer2, layer3):
                failures.extend(layer.failures)
                for signature in layer.evidence.get("failure_signatures", []):
                    self._append_unique(signatures, str(signature))

            evidence["layer1"] = layer1.evidence
            evidence["layer2"] = layer2.evidence
            evidence["layer3"] = layer3.evidence
            metrics["layer1"] = layer1.metrics
            metrics["layer2"] = layer2.metrics
            metrics["layer3"] = layer3.metrics

        passed = layer1.passed and layer2.passed and layer3.passed
        metrics["failure_count"] = len(failures)
        metrics["failure_signature_count"] = len(signatures)
        metrics["passed_layers"] = {
            "layer1": layer1.passed,
            "layer2": layer2.passed,
            "layer3": layer3.passed,
        }
        structured_evidence = self._build_structured_evidence(layer1, layer2, layer3)
        evidence["validated_entrypoints"] = structured_evidence["validated_entrypoints"]
        evidence["executed_tests"] = structured_evidence["executed_tests"]
        evidence["manifest_provenance_checks"] = structured_evidence["manifest_provenance_checks"]
        evidence["obligation_acceptance_checks"] = structured_evidence["obligation_acceptance_checks"]
        evidence["layer_status"] = {
            "layer1": layer1.passed,
            "layer2": layer2.passed,
            "layer3": layer3.passed,
        }
        evidence["failure_signatures"] = list(signatures)
        return ValidationArtifact(
            passed=passed,
            failures=failures,
            failure_signatures=signatures,
            evidence=evidence,
            metrics=metrics,
            layer1_result=layer1,
            layer2_result=layer2,
            layer3_result=layer3,
            failure_category=None if passed else self._classify_failure_category(signatures),
            next_route=None,
        )

    def _materialize_workspace(
        self,
        code_artifact: CodeArtifact,
        workspace: Path,
    ) -> Dict[str, Path]:
        materialized: Dict[str, Path] = {}
        for generated_file in code_artifact.files:
            target = workspace / generated_file.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(generated_file.content, encoding="utf-8")
            materialized[generated_file.path] = target
        return materialized

    def _sample_input_csv_content(self, build_spec: BuildSpec) -> str:
        return self.runtime_layer._sample_input_csv_content(build_spec)

    def _classify_failure_category(self, signatures: List[str]) -> FailureCategory | None:
        signature_set = set(signatures)
        if not signature_set:
            return None
        if {"missing_required_file", "manifest_mismatch", "provenance_mismatch"} & signature_set:
            return FailureCategory.ARCHITECTURAL
        if {
            "missing_obligation",
            "missing_acceptance_coverage",
            "semantic_omission",
            "missing_requirement_coverage",
            "missing_semantic_requirement_coverage",
            "universal_constraint_unproven",
            "quality_contract_violation",
            "capability_contract_violation",
            "missing_capability",
            "non_semantic_test",
            "fake_acceptance_coverage",
        } & signature_set:
            return FailureCategory.VALIDATION
        if {"syntax_error", "import_failure", "missing_entrypoint", "test_execution_failure", "superficial_stub"} & signature_set:
            return FailureCategory.IMPLEMENTATION
        return FailureCategory.UNKNOWN

    def _append_unique(self, collection: List[str], value: str) -> None:
        if value not in collection:
            collection.append(value)

    def _build_structured_evidence(
        self,
        layer1: ValidationLayerResult,
        layer2: ValidationLayerResult,
        layer3: ValidationLayerResult,
    ) -> Dict[str, object]:
        entrypoint_results = layer1.evidence.get("entrypoint_results", {})
        validated_entrypoints: Dict[str, object] = {}
        if isinstance(entrypoint_results, dict):
            for path, result in entrypoint_results.items():
                if not isinstance(result, dict):
                    continue
                validated_entrypoints[str(path)] = {
                    "exists": bool(result.get("exists", False)),
                    "function_present": bool(result.get("function_present", False)),
                    "executed": bool(result.get("executed", False)),
                    "returncode": result.get("returncode"),
                }

        test_execution = layer2.evidence.get("test_execution", {})
        if not isinstance(test_execution, dict):
            test_execution = {}

        manifest_provenance_checks = {
            "manifest_missing_files": layer3.evidence.get("manifest_missing_files", []),
            "provenance_mismatches": layer3.evidence.get("provenance_mismatches", []),
            "traceability_extras": layer3.evidence.get("traceability_extras", []),
            "missing_entrypoint_interfaces": layer3.evidence.get("missing_entrypoint_interfaces", []),
            "superficial_interfaces": layer3.evidence.get("superficial_interfaces", []),
            "non_semantic_tests": layer3.evidence.get("non_semantic_tests", []),
        }

        obligation_acceptance_checks = {
            "missing_required_files": layer2.evidence.get("missing_required_files", []),
            "missing_required_tests": layer2.evidence.get("missing_required_tests", []),
            "missing_manifest_obligations": layer2.evidence.get("missing_manifest_obligations", []),
            "missing_provenance_obligations": layer2.evidence.get("missing_provenance_obligations", []),
            "missing_acceptance_coverage": layer2.evidence.get("missing_acceptance_coverage", []),
            "requirement_coverage_checks": layer2.evidence.get("requirement_coverage_checks", {}),
            "quality_contract_checks": layer2.evidence.get("quality_contract_checks", []),
            "capability_contract_checks": layer2.evidence.get("capability_contract_checks", {}),
            "semantic_requirement_test_coverage": layer3.evidence.get("semantic_requirement_test_coverage", {}),
        }

        return {
            "validated_entrypoints": validated_entrypoints,
            "executed_tests": {
                "ran": bool(test_execution.get("ran", False)),
                "returncode": test_execution.get("returncode"),
                "tests": test_execution.get("tests", []),
                "stdout": test_execution.get("stdout", ""),
                "stderr": test_execution.get("stderr", ""),
            },
            "manifest_provenance_checks": manifest_provenance_checks,
            "obligation_acceptance_checks": obligation_acceptance_checks,
        }
