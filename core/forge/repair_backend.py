from typing import Any, Protocol

from core.forge.contracts import (
    CodeArtifact,
    FeasiblePlan,
    RepairDirective,
    RepairPatchCandidate,
    ValidationArtifact,
)
from core.kernel import ReasoningKernel
from core.substrate import CognitiveSubstrate


class ArtifactRepairBackend(Protocol):
    def propose(
        self,
        plan: FeasiblePlan,
        artifact: CodeArtifact,
        validation: ValidationArtifact,
        directive: RepairDirective,
    ) -> RepairPatchCandidate: ...


class SubstrateRepairBackend:
    """Produces untrusted file revisions through the existing Derivative substrate."""

    def __init__(
        self,
        execution_mode: str = "hybrid",
        substrate: CognitiveSubstrate | None = None,
        kernel: ReasoningKernel | None = None,
    ) -> None:
        self.substrate = substrate or CognitiveSubstrate(execution_mode=execution_mode)
        self.kernel = kernel or ReasoningKernel(execution_mode=execution_mode)

    def propose(
        self,
        plan: FeasiblePlan,
        artifact: CodeArtifact,
        validation: ValidationArtifact,
        directive: RepairDirective,
    ) -> RepairPatchCandidate:
        if not directive.repairable:
            return RepairPatchCandidate(
                backend_name="substrate",
                available=False,
                stop_reason=directive.stop_reason or "Repair directive is not repairable.",
            )
        if not getattr(self.kernel, "use_live_model", False):
            return RepairPatchCandidate(
                backend_name="substrate",
                available=False,
                stop_reason="Live kernel revision is unavailable in the selected execution mode.",
            )

        files_by_path = {
            self._normalize_path(generated.path): generated.content
            for generated in artifact.files
        }
        allowed_paths = self._allowed_paths(directive, files_by_path, artifact)
        if not allowed_paths:
            return RepairPatchCandidate(
                backend_name="substrate",
                available=True,
                stop_reason="No validator-grounded source or test paths are eligible for revision.",
            )

        problem = self._repair_problem(plan, validation, directive)
        framings = self.substrate.decompose(problem)
        target_files = {path: files_by_path[path] for path in allowed_paths}
        payload = self.kernel.propose_code_revision(
            repair_context=self._repair_context(plan, validation, directive),
            target_files=target_files,
            lens_framings=framings,
        )
        proposed_files = self._coerce_files(payload.get("files"))
        kernel_status = str(payload.get("status", "candidate"))
        kernel_reason = str(payload.get("reason", "")).strip()
        backend_available = kernel_status != "unavailable"
        accepted: dict[str, str] = {}
        rejected: list[str] = []
        for raw_path, content in proposed_files.items():
            path = self._normalize_path(raw_path)
            if path not in allowed_paths:
                rejected.append(path)
                continue
            accepted[path] = content

        evidence = {
            "repair_id": directive.repair_id,
            "operations": list(directive.operations),
            "failure_signatures": list(directive.failure_signatures),
            "allowed_paths": allowed_paths,
            "accepted_paths": sorted(accepted),
            "rejected_paths": sorted(set(rejected)),
            "lens_names": [framing.lens_name for framing in framings],
            "kernel_status": kernel_status,
            "kernel_reason": kernel_reason,
        }
        return RepairPatchCandidate(
            backend_name="substrate",
            files=accepted,
            evidence=evidence,
            rejected_paths=sorted(set(rejected)),
            available=backend_available,
            stop_reason=(
                ""
                if accepted
                else kernel_reason or "Kernel returned no eligible file revisions."
            ),
        )

    @staticmethod
    def _allowed_paths(
        directive: RepairDirective,
        files_by_path: dict[str, str],
        artifact: CodeArtifact,
    ) -> list[str]:
        manifest_paths = {
            SubstrateRepairBackend._normalize_path(path)
            for path in artifact.manifest_paths
        }
        allowed: list[str] = []
        for raw_path in directive.target_paths:
            path = SubstrateRepairBackend._normalize_path(raw_path)
            if (
                path in files_by_path
                and path not in manifest_paths
                and path not in allowed
            ):
                allowed.append(path)
        return allowed

    @staticmethod
    def _repair_problem(
        plan: FeasiblePlan,
        validation: ValidationArtifact,
        directive: RepairDirective,
    ) -> str:
        return (
            f"Repair generated software for this requirement: {plan.build_spec.normalized_requirement}\n"
            f"Validation failures: {', '.join(validation.failure_signatures)}\n"
            f"Required operations: {', '.join(directive.operations)}\n"
            "Preserve the plan, acceptance contract, obligations, provenance, and quality contract."
        )

    @staticmethod
    def _repair_context(
        plan: FeasiblePlan,
        validation: ValidationArtifact,
        directive: RepairDirective,
    ) -> dict[str, Any]:
        return {
            "requirement": plan.build_spec.normalized_requirement,
            "requirement_atoms": [
                {
                    "id": atom.requirement_id,
                    "text": atom.text,
                    "category": atom.category,
                    "strength": atom.strength,
                }
                for atom in plan.build_spec.requirement_atoms
            ],
            "architecture_summary": plan.architecture_summary,
            "interfaces": [
                {
                    "name": interface.name,
                    "type": interface.interface_type,
                    "signature": interface.signature,
                }
                for interface in plan.interfaces
            ],
            "quality_contract": vars(plan.quality_contract),
            "required_obligations": list(plan.required_obligations),
            "failure_signatures": list(validation.failure_signatures),
            "failures": list(validation.failures),
            "validator_evidence": validation.evidence,
            "repair_operations": list(directive.operations),
            "evidence_refs": list(directive.evidence_refs),
        }

    @staticmethod
    def _coerce_files(value: Any) -> dict[str, str]:
        if isinstance(value, dict):
            return {
                str(path): content
                for path, content in value.items()
                if isinstance(content, str)
            }
        if not isinstance(value, list):
            return {}
        files: dict[str, str] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            content = item.get("content")
            if isinstance(path, str) and isinstance(content, str):
                files[path] = content
        return files

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = str(path).strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized
