import importlib  # noqa: F401 - retained as a monkeypatch seam for dependency checks
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from core.forge.artifact_files import (
    ArtifactPathError,
    artifact_file_changes,
    artifact_file_hashes,
    artifact_file_targets,
)
from core.forge.contracts import (
    BuildSpec,
    CodeArtifact,
    FailureCategory,
    FeasiblePlan,
    ValidationArtifact,
    ValidationLayerResult,
)
from core.forge.execution import LocalProcessExecutor, ProcessExecutor
from core.forge.evidence_integrity import (
    artifact_validation_seal,
    validation_artifact_seal,
)
from core.forge.repair_support import behavioral_contract_seal
from core.forge.validation.adversarial import AdversarialValidationLayer
from core.forge.validation.obligations import ObligationValidationLayer
from core.forge.validation.quality import QualityContractChecker
from core.forge.validation.runtime import RuntimeValidationLayer


class ValidatorStage:
    def __init__(
        self,
        python_executable: str | None = None,
        timeout_seconds: int = 120,
        executor: ProcessExecutor | None = None,
        require_isolation: bool = False,
    ):
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = timeout_seconds
        self.executor = executor or LocalProcessExecutor(
            python_executable=self.python_executable,
        )
        self.require_isolation = require_isolation
        quality_checker = QualityContractChecker()
        self.runtime_layer = RuntimeValidationLayer(self.executor, timeout_seconds)
        self.obligation_layer = ObligationValidationLayer(
            self.executor,
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
        if self.require_isolation and not self.executor.policy.isolated:
            contract_seal = (
                behavioral_contract_seal(plan) if plan is not None else {}
            )
            return self._isolation_refusal(contract_seal)

        contract_seal = behavioral_contract_seal(plan)
        validated_artifact_seal = artifact_validation_seal(code_artifact)

        failures: List[str] = []
        signatures: List[str] = []
        evidence: Dict[str, object] = {}
        metrics: Dict[str, object] = {}
        evidence["execution_policy"] = self.executor.policy.evidence()
        evidence["behavioral_contract_seal"] = contract_seal
        evidence["validated_artifact_seal"] = validated_artifact_seal

        with tempfile.TemporaryDirectory(
            prefix="forge_validator_",
            ignore_cleanup_errors=True,
        ) as tmp_dir:
            workspace = Path(tmp_dir)
            try:
                materialized = self._materialize_workspace(code_artifact, workspace)
            except ArtifactPathError as exc:
                return self._validation_refusal(
                    contract_seal, "artifact_path_violation", str(exc),
                    validated_artifact_seal,
                )
            evidence["workspace"] = str(workspace)
            evidence["materialized_files"] = sorted(str(path) for path in materialized.values())
            expected_hashes = artifact_file_hashes(materialized)
            evidence["workspace_integrity"] = {
                "digest_mode": "materialized_bytes_sha256",
                "initial_file_hashes": expected_hashes,
                "checkpoints": [],
            }
            layers = []
            integrity_failed = False
            for layer, layer_name in (
                (self.runtime_layer, "layer1_syntax_import_run"),
                (self.obligation_layer, "layer2_obligations_tests_acceptance"),
                (self.adversarial_layer, "layer3_adversarial"),
            ):
                if integrity_failed:
                    result = ValidationLayerResult(
                        layer_name=layer_name, passed=False,
                        evidence={"executed": False, "skip_reason": "artifact_integrity_violation"},
                        metrics={"duration_ms": 0},
                    )
                else:
                    result = layer.validate(code_artifact, plan, build_spec, materialized, workspace)
                    changes = artifact_file_changes(materialized, expected_hashes, workspace)
                    if artifact_validation_seal(code_artifact) != validated_artifact_seal:
                        changes.append({"path": "<artifact>", "reason": "artifact_metadata_changed"})
                    evidence["workspace_integrity"]["checkpoints"].append({
                        "after_layer": layer_name, "passed": not changes, "changes": changes,
                    })
                    if changes:
                        integrity_failed = True
                        result.passed = False
                        result.failures.append("Artifact files or identity changed during validation.")
                        result.evidence.setdefault("failure_signatures", []).append(
                            "artifact_integrity_violation"
                        )
                layers.append(result)
            layer1, layer2, layer3 = layers

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
        validation = ValidationArtifact(
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
        validation.integrity_seal = validation_artifact_seal(validation)
        return validation

    def _isolation_refusal(
        self,
        contract_seal: Dict[str, object],
    ) -> ValidationArtifact:
        return self._validation_refusal(
            contract_seal, "sandbox_policy_violation",
            "Validation requires an isolated execution backend; local execution was refused.",
        )

    def _validation_refusal(
        self, contract_seal: Dict[str, object], signature: str, failure: str,
        validated_artifact_seal: Dict[str, object] | None = None,
    ) -> ValidationArtifact:
        policy_evidence = self.executor.policy.evidence()
        layers = [
            ValidationLayerResult(
                layer_name=layer_name,
                passed=False,
                failures=[failure],
                evidence={
                    "failure_signatures": [signature],
                    "execution_policy": policy_evidence,
                    "executed": False,
                },
                metrics={"duration_ms": 0},
            )
            for layer_name in (
                "layer1_syntax_import_run",
                "layer2_obligations_tests_acceptance",
                "layer3_adversarial",
            )
        ]
        validation = ValidationArtifact(
            passed=False,
            failures=[failure],
            failure_signatures=[signature],
            evidence={
                "execution_policy": policy_evidence,
                "behavioral_contract_seal": contract_seal,
                "validated_entrypoints": {},
                "executed_tests": {"ran": False},
                "manifest_provenance_checks": {},
                "obligation_acceptance_checks": {},
                "layer_status": {"layer1": False, "layer2": False, "layer3": False},
                "failure_signatures": [signature],
            },
            metrics={
                "failure_count": 1,
                "failure_signature_count": 1,
                "passed_layers": {"layer1": False, "layer2": False, "layer3": False},
            },
            layer1_result=layers[0],
            layer2_result=layers[1],
            layer3_result=layers[2],
            failure_category=FailureCategory.VALIDATION,
            next_route=None,
        )
        if validated_artifact_seal is not None:
            validation.evidence["validated_artifact_seal"] = validated_artifact_seal
        validation.integrity_seal = validation_artifact_seal(validation)
        return validation

    def _materialize_workspace(
        self,
        code_artifact: CodeArtifact,
        workspace: Path,
    ) -> Dict[str, Path]:
        materialized = artifact_file_targets(code_artifact.files, workspace)
        for generated_file in code_artifact.files:
            target = materialized[generated_file.path]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(generated_file.content, encoding="utf-8")
            materialized[generated_file.path] = target
        return materialized

    def _sample_input_csv_content(self, build_spec: BuildSpec) -> str:
        return self.runtime_layer._sample_input_csv_content(build_spec)

    def _classify_failure_category(self, signatures: List[str]) -> FailureCategory | None:
        signature_set = set(signatures)
        if {"artifact_integrity_violation", "artifact_path_violation"} & signature_set:
            return FailureCategory.VALIDATION
        if not signature_set:
            return None
        if "semantic_content_mismatch" in signature_set:
            return FailureCategory.IMPLEMENTATION
        if {"missing_required_file", "manifest_mismatch", "provenance_mismatch"} & signature_set:
            return FailureCategory.ARCHITECTURAL
        if {
            "sandbox_policy_violation",
            "sandbox_unavailable",
            "underspecified_requirement",
            "missing_obligation",
            "missing_acceptance_coverage",
            "semantic_omission",
            "missing_requirement_coverage",
            "missing_semantic_requirement_coverage",
            "universal_constraint_unproven",
            "quality_contract_violation",
            "capability_contract_violation",
            "missing_capability",
            "adapter_capability_mismatch",
            "adapter_capability_manifest_mismatch",
            "candidate_preflight_failure",
            "non_semantic_test",
            "fake_acceptance_coverage",
            "uncompiled_hard_conditional",
            "missing_conditional_coverage",
            "conditional_probe_unavailable",
            "test_expectation_contradiction",
            "lossy_observation_fidelity",
            "behavioral_contract_seal_mismatch",
        } & signature_set:
            return FailureCategory.VALIDATION
        if {
            "syntax_error",
            "import_failure",
            "entrypoint_execution_failure",
            "missing_entrypoint",
            "test_execution_failure",
            "superficial_stub",
            "interface_contract_mismatch",
            "conditional_obligation_mismatch",
        } & signature_set:
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
                    "backend": result.get("backend", ""),
                    "timed_out": bool(result.get("timed_out", False)),
                    "launch_error": result.get("launch_error"),
                    "isolation": result.get("isolation", {}),
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
            "material_ambiguities": layer2.evidence.get("material_ambiguities", []),
            "requirement_coverage_checks": layer2.evidence.get("requirement_coverage_checks", {}),
            "quality_contract_checks": layer2.evidence.get("quality_contract_checks", []),
            "capability_contract_checks": layer2.evidence.get("capability_contract_checks", {}),
            "adapter_capability_checks": layer2.evidence.get("adapter_capability_checks", {}),
            "conditional_obligation_checks": layer2.evidence.get(
                "conditional_obligation_checks",
                {},
            ),
            "exact_output_contract_checks": layer2.evidence.get(
                "exact_output_contract_checks",
                [],
            ),
            "behavioral_contract_seal": layer2.evidence.get(
                "behavioral_contract_seal",
                {},
            ),
            "repair_behavioral_contract_bindings": layer2.evidence.get(
                "repair_behavioral_contract_bindings",
                [],
            ),
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
                "backend": test_execution.get("backend", ""),
                "timed_out": bool(test_execution.get("timed_out", False)),
                "launch_error": test_execution.get("launch_error"),
                "isolation": test_execution.get("isolation", {}),
            },
            "manifest_provenance_checks": manifest_provenance_checks,
            "obligation_acceptance_checks": obligation_acceptance_checks,
        }
