from typing import Any, Callable, Protocol

from core.forge.contracts import (
    CodeArtifact,
    FeasiblePlan,
    RepairDirective,
    RepairPatchCandidate,
    ValidationArtifact,
)
from core.forge.execution import ProcessExecutor
from core.kernel import ReasoningKernel
from core.substrate import CognitiveSubstrate
from core.forge.repair_support import (
    preflight_failed_paths,
    preflight_has_source_failure,
    run_test_preflight,
    source_api_contracts,
    test_generation_contracts,
)


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
        preflight_timeout_seconds: int = 60,
        max_source_preflight_corrections: int = 2,
        max_test_preflight_corrections: int = 3,
        test_preflight_runner: Callable[[dict[str, str], list[str]], dict[str, Any]] | None = None,
        executor: ProcessExecutor | None = None,
    ) -> None:
        self.substrate = substrate or CognitiveSubstrate(execution_mode=execution_mode)
        self.kernel = kernel or ReasoningKernel(execution_mode=execution_mode)
        self.preflight_timeout_seconds = preflight_timeout_seconds
        self.max_source_preflight_corrections = max(
            1,
            int(max_source_preflight_corrections),
        )
        self.max_test_preflight_corrections = max(
            1,
            int(max_test_preflight_corrections),
        )
        self.test_preflight_runner = test_preflight_runner or (
            lambda candidate_files, test_paths: run_test_preflight(
                candidate_files,
                test_paths,
                timeout_seconds=self.preflight_timeout_seconds,
                executor=executor,
            )
        )

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
        preflight_attempts: list[dict[str, Any]] = []
        available_call_count = 0
        synthetic_reasons: list[str] = []

        def invoke(
            target_paths: list[str],
            *,
            atomic: bool,
            context_updates: dict[str, Any] | None = None,
            target_contents: dict[str, str] | None = None,
        ) -> bool:
            nonlocal available_call_count
            target_paths = sorted(target_paths)
            target_set = set(target_paths)
            repair_context = dict(base_context)
            repair_context.setdefault("repair_phase", "file_revision")
            if len(target_paths) == 1:
                transaction_name = target_paths[0]
            elif all(path.startswith("tests/") for path in target_paths):
                transaction_name = "test_suite_transaction"
            else:
                transaction_name = "source_transaction"
            repair_context["current_target_path"] = transaction_name
            repair_context["current_target_paths"] = target_paths
            if context_updates:
                repair_context.update(context_updates)
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
            if any(path.startswith("tests/") for path in target_paths):
                related_sources = repair_context["related_repaired_source_files"]
                repair_context["source_api_contracts"] = source_api_contracts(
                    related_sources
                )
                repair_context["test_generation_contracts"] = test_generation_contracts(
                    target_paths,
                    plan,
                    artifact,
                )
            current_targets = {
                path: (target_contents or {}).get(
                    path,
                    accepted.get(path, files_by_path[path]),
                )
                for path in target_paths
            }
            payload = self.kernel.propose_code_revision(
                repair_context=repair_context,
                target_files=current_targets,
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
                    f"Atomic revision omitted required targets: {', '.join(omitted)}"
                )
            kernel_calls.append(
                {
                    "target_path": transaction_name,
                    "target_paths": target_paths,
                    "status": status,
                    "reason": reason,
                    "accepted": str(target_accepted).lower(),
                    "omitted_paths": omitted,
                    "repair_phase": repair_context.get("repair_phase", "file_revision"),
                }
            )
            return target_accepted

        source_paths = sorted(path for path in allowed_paths if path.startswith("src/"))
        test_paths = sorted(path for path in allowed_paths if path.startswith("tests/"))
        other_paths = sorted(path for path in allowed_paths if path not in source_paths and path not in test_paths)
        impact_expanded_paths: list[str] = []
        if source_paths:
            required_test_paths = {
                f"tests/{planned_test.test_name}.py"
                for planned_test in plan.required_tests
                if planned_test.required
            }
            required_test_paths.update(
                self._normalize_path(plan_file.path)
                for plan_file in plan.file_tree_plan
                if self._normalize_path(plan_file.path).startswith("tests/")
            )
            impact_expanded_paths = sorted(
                path
                for path in required_test_paths
                if path in files_by_path and path not in test_paths
            )
            test_paths = sorted({*test_paths, *impact_expanded_paths})
            allowed_paths = [*allowed_paths, *impact_expanded_paths]
        source_ready = True
        if source_paths:
            source_ready = invoke(source_paths, atomic=True)
        for target_path in other_paths:
            invoke([target_path], atomic=False)
        if source_ready and test_paths:
            tests_ready = invoke(
                test_paths,
                atomic=True,
                context_updates={"repair_phase": "test_suite_generation"},
            )
            if tests_ready:
                first_candidate = {path: accepted.pop(path) for path in test_paths}
                preflight = self.test_preflight_runner(
                    {**files_by_path, **accepted, **first_candidate},
                    test_paths,
                )
                preflight_attempts.append(preflight)
                if preflight.get("passed", False):
                    accepted.update(first_candidate)
                else:
                    source_correction_attempt = 0
                    while (
                        source_paths
                        and not preflight.get("passed", False)
                        and source_correction_attempt < self.max_source_preflight_corrections
                        and (
                            (
                                source_correction_attempt == 0
                                and self._should_attempt_initial_source_correction(preflight)
                            )
                            or preflight_has_source_failure(preflight)
                        )
                    ):
                        source_correction_attempt += 1
                        correction_source_paths = self._preflight_target_paths(
                            preflight,
                            source_paths,
                        ) or source_paths
                        source_corrected = invoke(
                            correction_source_paths,
                            atomic=True,
                            context_updates={
                                "repair_phase": "source_preflight_correction",
                                "source_preflight_correction_attempt": source_correction_attempt,
                                "preflight_test_execution": preflight,
                                "preflight_failed_paths": preflight_failed_paths(preflight),
                                "candidate_test_suite": first_candidate,
                            },
                            target_contents={
                                path: accepted.get(path, files_by_path[path])
                                for path in correction_source_paths
                            },
                        )
                        if not source_corrected:
                            break
                        preflight = self.test_preflight_runner(
                            {**files_by_path, **accepted, **first_candidate},
                            test_paths,
                        )
                        preflight_attempts.append(preflight)
                    if preflight.get("passed", False):
                        accepted.update(first_candidate)
                    else:
                        current_test_candidate = first_candidate
                        test_suite_passed = False
                        correction_returned_all_paths = True
                        for correction_attempt in range(
                            1,
                            self.max_test_preflight_corrections + 1,
                        ):
                            correction_test_paths = self._preflight_target_paths(
                                preflight,
                                test_paths,
                            ) or test_paths
                            corrected = invoke(
                                correction_test_paths,
                                atomic=True,
                                context_updates={
                                    "repair_phase": "test_suite_correction",
                                    "preflight_correction_attempt": correction_attempt,
                                    "preflight_test_execution": preflight,
                                    "preflight_failed_paths": preflight_failed_paths(preflight),
                                },
                                target_contents={
                                    path: current_test_candidate[path]
                                    for path in correction_test_paths
                                },
                            )
                            if not corrected:
                                correction_returned_all_paths = False
                                break
                            for path in correction_test_paths:
                                current_test_candidate[path] = accepted.pop(path)
                            preflight = self.test_preflight_runner(
                                {
                                    **files_by_path,
                                    **accepted,
                                    **current_test_candidate,
                                },
                                test_paths,
                            )
                            preflight_attempts.append(preflight)
                            while (
                                source_paths
                                and not preflight.get("passed", False)
                                and source_correction_attempt < self.max_source_preflight_corrections
                                and preflight_has_source_failure(preflight)
                            ):
                                source_correction_attempt += 1
                                correction_source_paths = self._preflight_target_paths(
                                    preflight,
                                    source_paths,
                                ) or source_paths
                                source_corrected = invoke(
                                    correction_source_paths,
                                    atomic=True,
                                    context_updates={
                                        "repair_phase": "source_preflight_correction",
                                        "source_preflight_correction_attempt": source_correction_attempt,
                                        "preflight_test_execution": preflight,
                                        "preflight_failed_paths": preflight_failed_paths(preflight),
                                        "candidate_test_suite": current_test_candidate,
                                    },
                                    target_contents={
                                        path: accepted.get(path, files_by_path[path])
                                        for path in correction_source_paths
                                    },
                                )
                                if not source_corrected:
                                    break
                                preflight = self.test_preflight_runner(
                                    {
                                        **files_by_path,
                                        **accepted,
                                        **current_test_candidate,
                                    },
                                    test_paths,
                                )
                                preflight_attempts.append(preflight)
                            if preflight.get("passed", False):
                                accepted.update(current_test_candidate)
                                test_suite_passed = True
                                break
                        if not test_suite_passed:
                            for path in source_paths:
                                accepted.pop(path, None)
                            if correction_returned_all_paths:
                                synthetic_reasons.append(
                                    "Generated source and test transaction failed executable preflight after bounded corrections."
                                )
                            else:
                                synthetic_reasons.append(
                                    "Generated test suite correction did not return every required test path."
                                )

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
            "impact_expanded_paths": impact_expanded_paths,
            "accepted_paths": sorted(accepted),
            "rejected_paths": sorted(set(rejected)),
            "omitted_paths": omitted_paths,
            "lens_names": [framing.lens_name for framing in framings],
            "kernel_status": kernel_status,
            "kernel_reason": kernel_reason,
            "kernel_calls": kernel_calls,
            "test_preflight_attempts": preflight_attempts,
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
            "file_tree_plan": [
                {
                    "path": plan_file.path,
                    "purpose": plan_file.purpose,
                    "requirement_ids": list(plan_file.source_requirement_refs),
                }
                for plan_file in plan.file_tree_plan
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
    def _preflight_target_paths(
        preflight: dict[str, Any],
        eligible_paths: list[str],
    ) -> list[str]:
        eligible = set(eligible_paths)
        return [
            path
            for path in preflight_failed_paths(preflight)
            if path in eligible
        ]

    @staticmethod
    def _should_attempt_initial_source_correction(preflight: dict[str, Any]) -> bool:
        phase = str(preflight.get("phase", "")).lower()
        if phase in {"syntax", "import"}:
            return bool(preflight.get("source_failed_paths"))
        return True

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = str(path).strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized
