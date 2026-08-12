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
        base_context = self._repair_context(plan, validation, directive)
        accepted: dict[str, str] = {}
        rejected: list[str] = []
        kernel_calls: list[dict[str, Any]] = []
        available_call_count = 0
        synthetic_reasons: list[str] = []

        def invoke(target_paths: list[str], *, atomic: bool) -> bool:
            nonlocal available_call_count
            target_paths = sorted(target_paths)
            target_set = set(target_paths)
            repair_context = dict(base_context)
            repair_context["current_target_path"] = target_paths[0] if len(target_paths) == 1 else "source_transaction"
            repair_context["current_target_paths"] = target_paths
            if atomic:
                repair_context["related_repaired_source_files"] = {
                    path: accepted.get(path, content)
                    for path, content in sorted(files_by_path.items())
                    if path.startswith("src/") and path not in target_set
                }
            else:
                repair_context["related_repaired_source_files"] = self._related_source_files(
                    target_paths[0],
                    files_by_path,
                    accepted,
                    artifact,
                )
            payload = self.kernel.propose_code_revision(
                repair_context=repair_context,
                target_files={path: files_by_path[path] for path in target_paths},
                lens_framings=framings,
            )
            status = str(payload.get("status", "candidate"))
            reason = str(payload.get("reason", "")).strip()
            if status != "unavailable":
                available_call_count += 1
            proposed_files = self._coerce_files(payload.get("files"))
            eligible: dict[str, str] = {}
            for raw_path, content in proposed_files.items():
                path = self._normalize_path(raw_path)
                if path not in target_set:
                    rejected.append(path)
                    continue
                eligible[path] = content
            omitted = sorted(target_set - set(eligible))
            target_accepted = bool(eligible) and (not atomic or not omitted)
            if target_accepted:
                accepted.update(eligible)
            elif atomic and omitted and status != "unavailable":
                synthetic_reasons.append(
                    f"Atomic source revision omitted required targets: {', '.join(omitted)}"
                )
            kernel_calls.append(
                {
                    "target_path": target_paths[0] if len(target_paths) == 1 else "source_transaction",
                    "target_paths": target_paths,
                    "status": status,
                    "reason": reason,
                    "accepted": str(target_accepted).lower(),
                    "omitted_paths": omitted,
                }
            )
            return target_accepted

        source_paths = sorted(path for path in allowed_paths if path.startswith("src/"))
        test_paths = sorted(path for path in allowed_paths if path.startswith("tests/"))
        other_paths = sorted(path for path in allowed_paths if path not in source_paths and path not in test_paths)
        semantic_transaction = "implement_missing_requirement_semantics" in directive.operations

        source_ready = True
        if semantic_transaction and source_paths:
            source_ready = invoke(source_paths, atomic=True)
        else:
            for target_path in source_paths:
                invoke([target_path], atomic=False)
        for target_path in other_paths:
            invoke([target_path], atomic=False)
        if source_ready:
            for target_path in test_paths:
                invoke([target_path], atomic=False)

        omitted_paths = sorted(path for path in allowed_paths if path not in accepted)
        reasons = self._dedupe_strings(
            [*(call["reason"] for call in kernel_calls if call["reason"]), *synthetic_reasons]
        )
        backend_available = available_call_count > 0
        if accepted and not omitted_paths:
            kernel_status = "candidate"
        elif accepted:
            kernel_status = "partial"
        elif not backend_available:
            kernel_status = "unavailable"
        else:
            kernel_status = "empty"
        kernel_reason = "; ".join(reasons)

        evidence = {
            "repair_id": directive.repair_id,
            "operations": list(directive.operations),
            "failure_signatures": list(directive.failure_signatures),
            "allowed_paths": allowed_paths,
            "accepted_paths": sorted(accepted),
            "rejected_paths": sorted(set(rejected)),
            "omitted_paths": omitted_paths,
            "lens_names": [framing.lens_name for framing in framings],
            "kernel_status": kernel_status,
            "kernel_reason": kernel_reason,
            "kernel_calls": kernel_calls,
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
                    "evidence_terms": list(atom.evidence_terms),
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
    def _related_source_files(
        target_path: str,
        files_by_path: dict[str, str],
        accepted: dict[str, str],
        artifact: CodeArtifact,
    ) -> dict[str, str]:
        source_paths = sorted(path for path in files_by_path if path.startswith("src/"))
        related: dict[str, str] = {}
        for raw_path in source_paths:
            path = raw_path.replace("\\", "/")
            if path == target_path:
                continue
            related[path] = accepted.get(path, files_by_path[path])
        return related

    @staticmethod
    def _dedupe_strings(values) -> list[str]:
        deduplicated: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in deduplicated:
                deduplicated.append(text)
        return deduplicated

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = str(path).strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized
