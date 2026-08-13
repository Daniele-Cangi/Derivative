from typing import Any, Callable

from core.forge.contracts import (
    CodeArtifact,
    FeasiblePlan,
    RepairDirective,
    RepairPatchCandidate,
    ValidationArtifact,
)
from core.forge.candidate_preflight import run_semantic_preflight
from core.forge.repair_backend import SubstrateRepairBackend
from core.forge.repair_support import (
    run_test_preflight,
    test_generation_contracts as build_test_generation_contracts,
)
from core.kernel import ReasoningKernel
from core.substrate import CognitiveSubstrate


class SubstrateCandidateCompiler:
    """Compiles one complete, untrusted plan-bound candidate transaction."""

    backend_name = "candidate_compiler"

    def __init__(
        self,
        execution_mode: str = "hybrid",
        substrate: CognitiveSubstrate | None = None,
        kernel: ReasoningKernel | None = None,
        preflight_timeout_seconds: int = 60,
        max_preflight_corrections: int = 2,
        test_preflight_runner: Callable[[dict[str, str], list[str]], dict[str, Any]] | None = None,
    ) -> None:
        self.substrate = substrate or CognitiveSubstrate(execution_mode=execution_mode)
        self.kernel = kernel or ReasoningKernel(execution_mode=execution_mode)
        self.max_preflight_corrections = max(0, int(max_preflight_corrections))
        self.test_preflight_runner = test_preflight_runner or (
            lambda files, tests: run_test_preflight(
                files,
                tests,
                timeout_seconds=preflight_timeout_seconds,
            )
        )

    def propose(
        self,
        plan: FeasiblePlan,
        artifact: CodeArtifact,
        validation: ValidationArtifact,
        directive: RepairDirective,
    ) -> RepairPatchCandidate:
        if not any(
            operation in directive.operations
            for operation in (
                "compile_uncovered_capabilities",
                "recompile_candidate_transaction",
            )
        ):
            return RepairPatchCandidate(
                backend_name=self.backend_name,
                available=False,
                stop_reason="Candidate compilation requires an uncovered-capability directive.",
            )
        if not getattr(self.kernel, "use_live_model", False):
            return RepairPatchCandidate(
                backend_name=self.backend_name,
                available=False,
                stop_reason="Live candidate compilation is unavailable in the selected execution mode.",
            )

        files_by_path = {
            self._normalize_path(generated.path): generated.content
            for generated in artifact.files
        }
        planned_paths = self._planned_paths(plan)
        manifest_paths = {self._normalize_path(path) for path in artifact.manifest_paths}
        target_paths = sorted(planned_paths - manifest_paths)
        missing_paths = sorted(set(target_paths) - set(files_by_path))
        directive_targets = {
            self._normalize_path(path)
            for path in directive.target_paths
        }
        target_not_authorized = sorted(set(target_paths) - directive_targets)
        if missing_paths or target_not_authorized or not target_paths:
            return RepairPatchCandidate(
                backend_name=self.backend_name,
                available=True,
                evidence={
                    "planned_paths": target_paths,
                    "missing_paths": missing_paths,
                    "unauthorized_paths": target_not_authorized,
                },
                stop_reason="The complete planned candidate transaction is not materialized and authorized.",
            )

        problem = (
            "Compile a complete software candidate for this feasible plan without changing its architecture: "
            f"{plan.build_spec.normalized_requirement}"
        )
        framings = self.substrate.decompose(problem)
        context = SubstrateRepairBackend._repair_context(plan, validation, directive)
        context.update(
            {
                "repair_phase": "initial_candidate_compilation",
                "current_target_path": "complete_candidate_transaction",
                "current_target_paths": target_paths,
                "candidate_transaction_required": True,
                "candidate_transaction_rules": [
                    "Return every allowed path in one response.",
                    "Implement only the declared plan and requirement atoms.",
                    "Tests must execute the generated public interfaces and assert concrete behavior.",
                    "Every mapped test must exercise every evidence term in its test_generation_contract.",
                    "Invalid-input rejection tests must assert ValueError, TypeError, or SystemExit unless the requirement explicitly defines a return-code contract.",
                    "Do not author manifests, provenance, capability declarations, or validation claims.",
                    "Use requirement evidence terms as implementation or test identifiers where practical.",
                ],
            }
        )
        test_paths = sorted(path for path in target_paths if path.startswith("tests/"))
        context["test_generation_contracts"] = build_test_generation_contracts(
            test_paths,
            plan,
            artifact,
        )
        current_targets = {path: files_by_path[path] for path in target_paths}
        attempts: list[dict[str, Any]] = []
        rejected_paths: list[str] = []
        candidate_files: dict[str, str] = {}
        backend_available = False

        for attempt in range(self.max_preflight_corrections + 1):
            attempt_context = dict(context)
            attempt_context["candidate_compilation_attempt"] = attempt + 1
            if attempts:
                previous_preflight = attempts[-1]["preflight"]
                attempt_context["preflight_test_execution"] = previous_preflight
                attempt_context["candidate_correction_requirements"] = list(
                    previous_preflight.get("correction_requirements", [])
                )
            payload = self.kernel.propose_code_revision(
                repair_context=attempt_context,
                target_files=current_targets,
                lens_framings=framings,
            )
            status = str(payload.get("status", "candidate"))
            if status != "unavailable":
                backend_available = True
            else:
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": status,
                        "omitted_paths": target_paths,
                        "rejected_paths": sorted(set(rejected_paths)),
                        "preflight": {
                            "ran": False,
                            "passed": False,
                            "phase": "backend_unavailable",
                            "failed_paths": target_paths,
                        },
                    }
                )
                break
            proposed = SubstrateRepairBackend._coerce_files(payload.get("files"))
            normalized: dict[str, str] = {}
            for raw_path, content in proposed.items():
                path = self._normalize_path(raw_path)
                if path not in target_paths:
                    rejected_paths.append(path)
                    continue
                normalized[path] = content
            omitted = sorted(set(target_paths) - set(normalized))
            attempt_record: dict[str, Any] = {
                "attempt": attempt + 1,
                "status": status,
                "omitted_paths": omitted,
                "rejected_paths": sorted(set(rejected_paths)),
            }
            if omitted:
                attempt_record["preflight"] = {
                    "ran": False,
                    "passed": False,
                    "phase": "candidate_completeness",
                    "failed_paths": omitted,
                }
                attempts.append(attempt_record)
                candidate_files = {}
                continue

            preflight = self.test_preflight_runner(normalized, test_paths)
            if preflight.get("passed", False):
                preflight = run_semantic_preflight(
                    normalized,
                    plan,
                    context["test_generation_contracts"],
                    preflight,
                )
            attempt_record["preflight"] = preflight
            attempts.append(attempt_record)
            candidate_files = normalized
            if preflight.get("passed", False):
                break
            current_targets = normalized
        else:
            candidate_files = {}

        preflight_passed = bool(
            attempts
            and attempts[-1]["preflight"].get("passed", False)
            and set(candidate_files) == set(target_paths)
            and not rejected_paths
        )
        if not preflight_passed:
            candidate_files = {}

        evidence = {
            "repair_id": directive.repair_id,
            "operations": list(directive.operations),
            "allowed_paths": target_paths,
            "planned_paths": target_paths,
            "accepted_paths": sorted(candidate_files),
            "rejected_paths": sorted(set(rejected_paths)),
            "complete_transaction": bool(candidate_files),
            "preflight_passed": preflight_passed,
            "candidate_attempts": attempts,
            "lens_names": [getattr(framing, "lens_name", "") for framing in framings],
        }
        return RepairPatchCandidate(
            backend_name=self.backend_name,
            files=candidate_files,
            evidence=evidence,
            rejected_paths=sorted(set(rejected_paths)),
            available=backend_available,
            stop_reason=(
                ""
                if candidate_files
                else "Complete candidate transaction did not pass executable preflight."
            ),
        )

    @staticmethod
    def _planned_paths(plan: FeasiblePlan) -> set[str]:
        paths = {
            SubstrateCandidateCompiler._normalize_path(plan_file.path)
            for plan_file in plan.file_tree_plan
        }
        paths.update(
            f"tests/{planned_test.test_name}.py"
            for planned_test in plan.required_tests
            if planned_test.required
        )
        return paths

    @staticmethod
    def _normalize_path(path: str) -> str:
        return SubstrateRepairBackend._normalize_path(path)
