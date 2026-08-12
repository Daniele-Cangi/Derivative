import hashlib
from typing import Any, Dict, Iterable, List

from core.forge.contracts import (
    CodeArtifact,
    FeasiblePlan,
    ForgeRoute,
    RepairDirective,
    ValidationArtifact,
)


class RepairPolicy:
    """Compiles validator evidence into domain-neutral coder directives."""

    _OPERATIONS = {
        "syntax_error": "rerender_invalid_python",
        "import_failure": "restore_import_graph",
        "missing_entrypoint": "restore_declared_entrypoint",
        "test_execution_failure": "rerender_failing_tests_and_targets",
        "superficial_stub": "rerender_interface_implementation",
        "non_semantic_test": "rerender_semantic_tests",
        "fake_acceptance_coverage": "rerender_semantic_tests",
        "missing_acceptance_coverage": "restore_acceptance_tests",
        "missing_obligation": "restore_obligation_provenance",
        "semantic_content_mismatch": "implement_missing_requirement_semantics",
        "missing_semantic_requirement_coverage": "rerender_semantic_tests",
        "quality_contract_violation": "rerender_quality_contract_targets",
        "capability_contract_violation": "restore_capability_contract",
        "missing_capability": "restore_capability_modules",
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
        target_paths, evidence_refs = self._target_paths(
            validation,
            plan,
            artifact,
            signatures,
        )
        digest_source = "|".join(
            [plan.plan_id, str(attempt), *signatures, *operations, *target_paths]
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
            repairable=repairable,
            stop_reason=stop_reason,
        )

    def _target_paths(
        self,
        validation: ValidationArtifact,
        plan: FeasiblePlan,
        artifact: CodeArtifact,
        signatures: List[str],
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
                paths.append(f"src/{module_name}.py")
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

        return self._dedupe(paths), self._dedupe(refs)

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
