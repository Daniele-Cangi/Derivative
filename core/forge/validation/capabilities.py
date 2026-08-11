from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from core.forge.contracts import CodeArtifact, FeasiblePlan


class CapabilityContractChecker:
    def check(
        self,
        code_artifact: CodeArtifact,
        plan: FeasiblePlan,
        materialized: Dict[str, Path],
    ) -> Tuple[List[str], List[str], Dict[str, object]]:
        blueprint = plan.implementation_blueprint
        capabilities = [capability for capability in blueprint.capabilities if capability.enabled]
        if not capabilities:
            return [], [], {"required": False, "passed": True, "capabilities": {}}

        failures: List[str] = []
        signatures: List[str] = []
        evidence: Dict[str, object] = {"required": True, "capabilities": {}}
        capability_ids = [capability.capability_id for capability in capabilities]
        capability_id_set = set(capability_ids)

        if len(capability_ids) != len(capability_id_set):
            failures.append("Capability contract contains duplicate capability ids.")
            self._append_unique(signatures, "capability_contract_violation")

        manifest_blueprint = (
            code_artifact.artifact_manifest.get("implementation_blueprint")
            if isinstance(code_artifact.artifact_manifest, dict)
            else None
        )
        manifest_matches = manifest_blueprint == asdict(blueprint)
        evidence["manifest_blueprint_matches"] = manifest_matches
        if not manifest_matches:
            failures.append("Artifact manifest implementation blueprint does not match FeasiblePlan.")
            self._append_unique(signatures, "capability_contract_violation")

        for capability in capabilities:
            path = capability.module_path
            target = materialized.get(path)
            file_exists = target is not None and target.exists()
            generated = next((item for item in code_artifact.files if item.path == path), None)
            provenance = set(generated.generated_from_plan_sections if generated is not None else [])
            capability_token = f"capability:{capability.capability_id}"
            expected_quality_tokens = {
                f"quality_field:{field_name}" for field_name in capability.quality_fields
            }
            expected_dependency_tokens = {
                f"capability_dependency:{dependency_id}"
                for dependency_id in capability.dependencies
            }
            dependencies_valid = all(
                dependency_id in capability_id_set for dependency_id in capability.dependencies
            )
            provenance_matches = (
                capability_token in provenance
                and expected_quality_tokens.issubset(provenance)
                and expected_dependency_tokens.issubset(provenance)
            )

            source = target.read_text(encoding="utf-8") if file_exists and target is not None else ""
            missing_dependency_imports: List[str] = []
            for dependency_id in capability.dependencies:
                dependency = next(
                    (item for item in capabilities if item.capability_id == dependency_id),
                    None,
                )
                if dependency is None:
                    continue
                module_name = Path(dependency.module_path).stem
                if f"from {module_name} import" not in source and f"import {module_name}" not in source:
                    missing_dependency_imports.append(dependency_id)

            capability_evidence = {
                "type": capability.capability_type,
                "module_path": path,
                "file_exists": file_exists,
                "dependencies_valid": dependencies_valid,
                "missing_dependency_imports": missing_dependency_imports,
                "provenance_matches": provenance_matches,
                "quality_fields": list(capability.quality_fields),
            }
            evidence["capabilities"][capability.capability_id] = capability_evidence

            if not file_exists:
                failures.append(
                    f"Required capability module is missing: {capability.capability_id} -> {path}."
                )
                self._append_unique(signatures, "missing_capability")
            if not dependencies_valid or missing_dependency_imports:
                failures.append(
                    f"Capability dependencies are not implemented for {capability.capability_id}: "
                    f"{missing_dependency_imports}."
                )
                self._append_unique(signatures, "capability_contract_violation")
            if not provenance_matches:
                failures.append(
                    f"Capability provenance is incomplete for {capability.capability_id}."
                )
                self._append_unique(signatures, "capability_contract_violation")

        evidence["failures"] = list(failures)
        evidence["passed"] = not failures
        return failures, signatures, evidence

    @staticmethod
    def _append_unique(collection: List[str], value: str) -> None:
        if value not in collection:
            collection.append(value)
