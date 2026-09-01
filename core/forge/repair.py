import hashlib
import json
from typing import Any, Dict, Iterable, List

from core.forge.contracts import (
    CodeArtifact,
    FeasiblePlan,
    ForgeRoute,
    RepairDirective,
    ValidationArtifact,
)
from core.forge.evidence_integrity import to_jsonable
from core.forge.repair_evidence import compile_requirement_assertion_targets


class RepairPolicy:
    """Compiles validator evidence into domain-neutral coder directives."""

    _OPERATIONS = {
        "syntax_error": "rerender_invalid_python",
        "import_failure": "restore_import_graph",
        "entrypoint_execution_failure": "rerender_interface_implementation",
        "missing_entrypoint": "restore_declared_entrypoint",
        "test_execution_failure": "rerender_failing_tests_and_targets",
        "superficial_stub": "rerender_interface_implementation",
        "non_semantic_test": "rerender_semantic_tests",
        "fake_acceptance_coverage": "rerender_semantic_tests",
        "missing_acceptance_coverage": "restore_acceptance_tests",
        "missing_obligation": "restore_obligation_provenance",
        "semantic_content_mismatch": "implement_missing_requirement_semantics",
        "exact_output_mismatch": "implement_missing_requirement_semantics",
        "lossy_observation_fidelity": "repair_observation_fidelity",
        "missing_semantic_requirement_coverage": "rerender_semantic_tests",
        "missing_requirement_assertion_evidence": "repair_requirement_assertions",
        "quality_contract_violation": "rerender_quality_contract_targets",
        "capability_contract_violation": "restore_capability_contract",
        "missing_capability": "restore_capability_modules",
        "adapter_capability_mismatch": "compile_uncovered_capabilities",
        "candidate_preflight_failure": "recompile_candidate_transaction",
    }

    def compile(
        self,
        validation: ValidationArtifact,
        plan: FeasiblePlan,
        artifact: CodeArtifact,
        attempt: int,
        route: ForgeRoute = ForgeRoute.TO_CODER,
    ) -> RepairDirective:
        signatures = self._dedupe(validation.failure_signatures)
        operations = self._dedupe(
            self._OPERATIONS[signature]
            for signature in signatures
            if signature in self._OPERATIONS
        )
        evidence_targets = compile_requirement_assertion_targets(validation)
        target_paths, evidence_refs = self._target_paths(
            validation,
            plan,
            artifact,
            signatures,
            evidence_targets,
        )
        requirement_ids = self._requirement_ids(validation, evidence_targets)
        metadata = (
            artifact.artifact_manifest.get("metadata", {})
            if isinstance(artifact.artifact_manifest, dict)
            else {}
        )
        if (
            operations
            and isinstance(metadata, dict)
            and metadata.get("generator") == "forge_candidate_compiler"
            and not self._is_assertion_only_repair(operations)
        ):
            operations = self._dedupe([*operations, "recompile_candidate_transaction"])
            manifest_paths = {
                path.replace("\\", "/")
                for path in artifact.manifest_paths
            }
            target_paths = self._dedupe(
                generated.path
                for generated in artifact.files
                if generated.path.replace("\\", "/") not in manifest_paths
            )
            evidence_refs = self._dedupe(
                [*evidence_refs, "artifact_manifest.metadata.candidate_compilation"]
            )
        target_symbols = self._target_symbols(
            plan,
            target_paths,
            requirement_ids,
            evidence_targets,
        )
        digest_source = "|".join(
            [
                plan.plan_id,
                str(attempt),
                *signatures,
                *operations,
                *target_paths,
                *requirement_ids,
                *target_symbols,
                json.dumps(to_jsonable(evidence_targets), sort_keys=True),
            ]
        )
        repair_id = f"repair-{hashlib.sha256(digest_source.encode('utf-8')).hexdigest()[:12]}"
        repairable = bool(operations)
        stop_reason = (
            ""
            if repairable
            else "No deterministic coder repair is registered for the validation failure signatures."
        )
        return RepairDirective(
            repair_id=repair_id,
            attempt=attempt,
            route=route,
            failure_signatures=signatures,
            target_paths=target_paths,
            operations=operations,
            evidence_refs=evidence_refs,
            requirement_ids=requirement_ids,
            target_symbols=target_symbols,
            evidence_targets=evidence_targets,
            repairable=repairable,
            stop_reason=stop_reason,
        )

    def _target_paths(
        self,
        validation: ValidationArtifact,
        plan: FeasiblePlan,
        artifact: CodeArtifact,
        signatures: List[str],
        evidence_targets: Dict[str, Any],
    ) -> tuple[List[str], List[str]]:
        evidence = validation.evidence if isinstance(validation.evidence, dict) else {}
        paths: List[str] = []
        refs: List[str] = []

        layer1 = self._mapping(evidence.get("layer1"))
        for item in self._list(layer1.get("parse_errors")):
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                paths.append(item["path"])
                refs.append(f"layer1.parse_errors:{item['path']}")

        import_results = self._mapping(layer1.get("import_results"))
        for module_name, result in self._mapping(import_results.get("modules")).items():
            if isinstance(result, dict) and not result.get("ok", False):
                paths.append(self._source_path_for_module(str(module_name), artifact))
                refs.append(f"layer1.import_results:{module_name}")

        entrypoint_results = self._mapping(layer1.get("entrypoint_results"))
        for path, result in entrypoint_results.items():
            if isinstance(result, dict) and not result.get("executed", False):
                paths.append(str(path))
                refs.append(f"layer1.entrypoint_results:{path}")

        layer2 = self._mapping(evidence.get("layer2"))
        for key in ("missing_required_files", "missing_required_tests"):
            before = len(paths)
            self._extend_strings(paths, layer2.get(key))
            if len(paths) > before:
                refs.append(f"layer2.{key}")

        test_execution = self._mapping(layer2.get("test_execution"))
        if test_execution.get("returncode") not in (None, 0):
            failed_test_paths = self._pytest_failure_paths(
                test_execution,
                artifact.test_paths,
            )
            if failed_test_paths:
                paths.extend(failed_test_paths)
            else:
                self._extend_strings(paths, test_execution.get("tests"))
            refs.append("layer2.test_execution")

        capability_checks = self._mapping(layer2.get("capability_contract_checks"))
        for capability_id, item in self._mapping(capability_checks.get("capabilities")).items():
            if not isinstance(item, dict):
                continue
            failed = (
                not item.get("file_exists", False)
                or not item.get("dependencies_valid", False)
                or bool(item.get("missing_dependency_imports"))
                or not item.get("provenance_matches", False)
            )
            if failed and isinstance(item.get("module_path"), str):
                paths.append(item["module_path"])
                refs.append(f"layer2.capability_contract_checks:{capability_id}")

        layer3 = self._mapping(evidence.get("layer3"))
        for key in ("manifest_missing_files", "provenance_mismatches", "non_semantic_tests"):
            before = len(paths)
            self._extend_strings(paths, layer3.get(key))
            if len(paths) > before:
                refs.append(f"layer3.{key}")

        if "missing_requirement_assertion_evidence" in signatures:
            for item in evidence_targets.values():
                self._extend_strings(paths, item.get("test_paths"))
                self._extend_strings(refs, item.get("evidence_refs"))

        if "quality_contract_violation" in signatures:
            paths.extend(
                capability.module_path
                for capability in plan.implementation_blueprint.capabilities
                if capability.enabled and capability.quality_fields
            )
            refs.append("layer2.quality_contract_checks")

        if any(
            signature in signatures
            for signature in (
                "missing_acceptance_coverage",
                "missing_obligation",
            )
        ):
            paths.extend(artifact.test_paths)

        if {
            "semantic_content_mismatch",
            "missing_semantic_requirement_coverage",
        } & set(signatures):
            semantic_checks = self._mapping(layer2.get("requirement_semantic_checks"))
            for mismatch in self._list(semantic_checks.get("semantic_content_mismatches")):
                if not isinstance(mismatch, dict):
                    continue
                if "semantic_content_mismatch" in signatures:
                    self._extend_strings(paths, mismatch.get("source_paths"))
                self._extend_strings(paths, mismatch.get("test_paths"))
            refs.append("layer2.requirement_semantic_checks")

        if "exact_output_mismatch" in signatures:
            for item in self._list(layer2.get("exact_output_contract_checks")):
                if not isinstance(item, dict) or item.get("passed", False):
                    continue
                self._extend_strings(paths, item.get("paths"))
            refs.append("layer2.exact_output_contract_checks")

        if "lossy_observation_fidelity" in signatures:
            conditional_checks = self._mapping(
                layer2.get("conditional_obligation_checks")
            )
            for item in self._list(conditional_checks.get("observation_fidelity")):
                if not isinstance(item, dict) or not item.get("lossy_observations"):
                    continue
                path = item.get("path")
                if isinstance(path, str):
                    paths.append(path)
            refs.append("layer2.conditional_obligation_checks.observation_fidelity")

        if "missing_entrypoint" in signatures:
            paths.extend(artifact.runnable_entrypoints)
            if plan.implementation_blueprint.entrypoint_path:
                paths.append(plan.implementation_blueprint.entrypoint_path)

        if "superficial_stub" in signatures:
            capability_paths = [
                capability.module_path
                for capability in plan.implementation_blueprint.capabilities
                if capability.enabled and capability.interfaces
            ]
            paths.extend(
                capability_paths
                or [
                    plan_file.path
                    for plan_file in plan.file_tree_plan
                    if plan_file.path.startswith("src/")
                ]
            )

        if "adapter_capability_mismatch" in signatures:
            manifest_paths = {
                path.replace("\\", "/")
                for path in artifact.manifest_paths
            }
            paths.extend(
                generated.path
                for generated in artifact.files
                if generated.path.replace("\\", "/") not in manifest_paths
            )
            refs.append("layer2.adapter_capability_checks")

        if "candidate_preflight_failure" in signatures:
            refs.append("layer2.adapter_capability_checks.preflight_passed")

        return self._dedupe(paths), self._dedupe(refs)

    def _requirement_ids(
        self,
        validation: ValidationArtifact,
        evidence_targets: Dict[str, Any],
    ) -> List[str]:
        evidence = validation.evidence if isinstance(validation.evidence, dict) else {}
        layer2 = self._mapping(evidence.get("layer2"))
        layer3 = self._mapping(evidence.get("layer3"))
        requirement_ids: List[str] = []
        requirement_ids.extend(evidence_targets)

        semantic_checks = self._mapping(layer2.get("requirement_semantic_checks"))
        for mismatch in self._list(semantic_checks.get("semantic_content_mismatches")):
            if isinstance(mismatch, dict) and isinstance(mismatch.get("requirement_id"), str):
                requirement_ids.append(mismatch["requirement_id"])

        coverage_checks = self._mapping(layer2.get("requirement_coverage_checks"))
        for key in ("semantic_omissions", "missing_coverage", "universal_unproven"):
            self._extend_strings(requirement_ids, coverage_checks.get(key))

        adversarial_coverage = self._mapping(
            layer3.get("semantic_requirement_test_coverage")
        )
        self._extend_strings(
            requirement_ids,
            adversarial_coverage.get("missing_semantic_coverage"),
        )
        return self._dedupe(requirement_ids)

    @staticmethod
    def _is_assertion_only_repair(operations: List[str]) -> bool:
        test_operations = {
            "repair_requirement_assertions",
            "repair_observation_fidelity",
            "rerender_semantic_tests",
            "restore_acceptance_tests",
        }
        return (
            bool(
                {
                    "repair_requirement_assertions",
                    "repair_observation_fidelity",
                }
                & set(operations)
            )
            and set(operations).issubset(test_operations)
        )

    def _target_symbols(
        self,
        plan: FeasiblePlan,
        target_paths: List[str],
        requirement_ids: List[str],
        evidence_targets: Dict[str, Any],
    ) -> List[str]:
        normalized_targets = {path.replace("\\", "/") for path in target_paths}
        symbols: List[str] = []
        entrypoint_path = plan.implementation_blueprint.entrypoint_path.replace("\\", "/")
        for interface in plan.interfaces:
            expected_path = self._interface_source_path(
                interface.module_path,
                entrypoint_path,
            )
            if expected_path and expected_path in normalized_targets:
                symbols.append(interface.name)
        required = set(requirement_ids)
        for capability in plan.implementation_blueprint.capabilities:
            if capability.module_path.replace("\\", "/") not in normalized_targets:
                continue
            if required and not (required & set(capability.requirement_ids)):
                continue
            symbols.extend(capability.interfaces)
        for target in evidence_targets.values():
            for function in self._list(target.get("causal_functions")):
                if not isinstance(function, dict):
                    continue
                path = str(function.get("path", "")).replace("\\", "/")
                name = function.get("function")
                if path in normalized_targets and isinstance(name, str):
                    symbols.append(name)
        return self._dedupe(symbols)

    @staticmethod
    def _interface_source_path(module_path: str, entrypoint_path: str) -> str:
        raw = module_path.replace("\\", "/").strip("/")
        if not raw:
            return entrypoint_path
        if raw.endswith(".py"):
            return raw if raw.startswith("src/") else f"src/{raw}"
        normalized = raw.removeprefix("src/").replace(".", "/")
        return f"src/{normalized}.py"

    @staticmethod
    def _source_path_for_module(module_name: str, artifact: CodeArtifact) -> str:
        module_path = module_name.replace(".", "/")
        file_path = f"src/{module_path}.py"
        package_path = f"src/{module_path}/__init__.py"
        artifact_paths = {generated.path.replace("\\", "/") for generated in artifact.files}
        if package_path in artifact_paths:
            return package_path
        return file_path

    @staticmethod
    def _mapping(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _list(value: Any) -> List[Any]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _extend_strings(target: List[str], value: Any) -> None:
        if isinstance(value, list):
            target.extend(item for item in value if isinstance(item, str))

    @staticmethod
    def _pytest_failure_paths(
        test_execution: Dict[str, Any],
        known_test_paths: List[str],
    ) -> List[str]:
        output = "\n".join(
            str(test_execution.get(key, ""))
            for key in ("stdout", "stderr")
        ).replace("\\", "/")
        failures: List[str] = []
        for line in output.splitlines():
            normalized = line.strip()
            if not any(marker in normalized for marker in ("FAILED ", "ERROR ", "ERROR collecting")):
                continue
            for path in known_test_paths:
                normalized_path = path.replace("\\", "/")
                if normalized_path in normalized and path not in failures:
                    failures.append(path)
        return failures

    @staticmethod
    def _dedupe(values: Iterable[str]) -> List[str]:
        result: List[str] = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result
