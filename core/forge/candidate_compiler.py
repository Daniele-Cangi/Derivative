import ast
from typing import Any, Callable

from core.forge.contracts import (
    CodeArtifact,
    FeasiblePlan,
    RepairDirective,
    RepairPatchCandidate,
    ValidationArtifact,
)
from core.forge.execution import ProcessExecutor
from core.forge.candidate_preflight import run_semantic_preflight
from core.forge.repair_backend import SubstrateRepairBackend
from core.forge.repair_support import (
    preflight_failed_paths,
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
        executor: ProcessExecutor | None = None,
    ) -> None:
        self.substrate = substrate or CognitiveSubstrate(execution_mode=execution_mode)
        self.kernel = kernel or ReasoningKernel(execution_mode=execution_mode)
        self.max_preflight_corrections = max(0, int(max_preflight_corrections))
        self.test_preflight_runner = test_preflight_runner or (
            lambda files, tests: run_test_preflight(
                files,
                tests,
                timeout_seconds=preflight_timeout_seconds,
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
                    "Requirement atoms and declared public interfaces override incompatible semantics in stale target files.",
                    "Derive numeric test expectations by tracing every fixture record through the normalized requirement.",
                    "When a requirement preserves exact bytes or CRLF/LF/CR line endings, observe output "
                    "with read_bytes(), binary mode, or open(..., newline=''); Path.read_text() normalizes "
                    "newlines and is not valid evidence.",
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
        active_paths = list(target_paths)

        for attempt in range(self.max_preflight_corrections + 1):
            attempt_context = dict(context)
            attempt_context["candidate_compilation_attempt"] = attempt + 1
            attempt_context["current_target_paths"] = list(active_paths)
            attempt_context["candidate_transaction_correction"] = bool(attempts)
            preserved_paths = sorted(set(target_paths) - set(active_paths))
            attempt_context["preserve_passing_paths"] = preserved_paths
            if candidate_files and preserved_paths:
                attempt_context["preserved_candidate_files"] = {
                    path: candidate_files[path]
                    for path in preserved_paths
                    if path in candidate_files
                }
                attempt_context["preservation_contract"] = (
                    "These files passed the previous preflight and are immutable context. "
                    "The revision must remain compatible with their imports, fixtures, and assertions."
                )
            if attempts:
                previous_preflight = attempts[-1]["preflight"]
                attempt_context["preflight_test_execution"] = previous_preflight
                attempt_context["candidate_correction_requirements"] = (
                    self._correction_requirements(previous_preflight)
                )
                if previous_preflight.get("phase") == "tests":
                    attempt_context["candidate_correction_requirements"].append(
                        "Trace each failing test fixture against the requirement, then correct the implementation "
                        "or the expected assertion according to that trace; never preserve a contradicted count."
                    )
            payload = self.kernel.propose_code_revision(
                repair_context=attempt_context,
                target_files={path: current_targets[path] for path in active_paths},
                lens_framings=framings,
            )
            status = str(payload.get("status", "candidate"))
            reason = " ".join(str(payload.get("reason", "")).split())[:240]
            if status != "unavailable":
                backend_available = True
            else:
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": status,
                        "reason": reason,
                        "target_paths": list(active_paths),
                        "omitted_paths": list(active_paths),
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
            ignored_correction_paths: list[str] = []
            for raw_path, content in proposed.items():
                path = self._normalize_path(raw_path)
                if path not in target_paths:
                    rejected_paths.append(path)
                    continue
                if path not in active_paths:
                    ignored_correction_paths.append(path)
                    continue
                normalized[path] = content
            omitted = sorted(set(active_paths) - set(normalized))
            attempt_record: dict[str, Any] = {
                "attempt": attempt + 1,
                "status": status,
                "reason": reason,
                "target_paths": list(active_paths),
                "preserved_paths": sorted(set(target_paths) - set(active_paths)),
                "omitted_paths": omitted,
                "ignored_correction_paths": sorted(set(ignored_correction_paths)),
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

            merged_candidate = dict(candidate_files)
            merged_candidate.update(normalized)
            if set(merged_candidate) != set(target_paths):
                missing_transaction_paths = sorted(set(target_paths) - set(merged_candidate))
                attempt_record["preflight"] = {
                    "ran": False,
                    "passed": False,
                    "phase": "candidate_completeness",
                    "failed_paths": missing_transaction_paths,
                }
                attempts.append(attempt_record)
                candidate_files = merged_candidate
                active_paths = missing_transaction_paths
                current_targets = {
                    path: files_by_path[path]
                    for path in active_paths
                }
                continue

            preflight = self.test_preflight_runner(merged_candidate, test_paths)
            if preflight.get("passed", False):
                preflight = run_semantic_preflight(
                    merged_candidate,
                    plan,
                    context["test_generation_contracts"],
                    preflight,
                )
            attempt_record["preflight"] = preflight
            attempts.append(attempt_record)
            candidate_files = merged_candidate
            if preflight.get("passed", False):
                break
            failed = preflight_failed_paths(preflight)
            impact_expanded_paths: list[str] = []
            if preflight.get("phase") == "tests":
                impact_expanded_paths = self._imported_source_paths(
                    candidate_files,
                    [path for path in failed if path.startswith("tests/")],
                    target_paths,
                )
                preflight["impact_expanded_paths"] = impact_expanded_paths
                failed = list(dict.fromkeys([*failed, *impact_expanded_paths]))
            active_paths = [path for path in failed if path in target_paths]
            if not active_paths:
                active_paths = list(target_paths)
            current_targets = {
                path: candidate_files[path]
                for path in active_paths
            }
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
            "backend_reason": next(
                (
                    str(attempt.get("reason", ""))
                    for attempt in reversed(attempts)
                    if str(attempt.get("reason", "")).strip()
                ),
                "",
            ),
            "lens_names": [getattr(framing, "lens_name", "") for framing in framings],
        }
        backend_reason = str(evidence["backend_reason"])
        return RepairPatchCandidate(
            backend_name=self.backend_name,
            files=candidate_files,
            evidence=evidence,
            rejected_paths=sorted(set(rejected_paths)),
            available=backend_available,
            stop_reason=(
                ""
                if candidate_files
                else backend_reason
                or "Complete candidate transaction did not pass executable preflight."
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

    @classmethod
    def _imported_source_paths(
        cls,
        candidate_files: dict[str, str],
        test_paths: list[str],
        allowed_paths: list[str],
    ) -> list[str]:
        imported_modules: set[str] = set()
        for test_path in test_paths:
            try:
                tree = ast.parse(candidate_files.get(test_path, ""))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module)

        impacted: list[str] = []
        for path in allowed_paths:
            if not path.startswith("src/") or not path.endswith(".py"):
                continue
            module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
            if module.endswith(".__init__"):
                module = module.removesuffix(".__init__")
            module_leaf = module.rsplit(".", 1)[-1]
            if any(
                imported == module
                or imported.startswith(f"{module}.")
                or module.startswith(f"{imported}.")
                or imported == module_leaf
                for imported in imported_modules
            ):
                impacted.append(path)
        return sorted(set(impacted))

    @staticmethod
    def _correction_requirements(preflight: dict[str, Any]) -> list[str]:
        requirements = [
            str(item).strip()
            for item in preflight.get("correction_requirements", [])
            if str(item).strip()
        ]
        for detail in preflight.get("failure_details", []):
            if not isinstance(detail, dict):
                continue
            node_id = str(detail.get("node_id") or detail.get("path") or "pytest failure")
            message = str(detail.get("message") or "pytest assertion failed")
            requirements.append(
                f"{node_id}: resolve the observed failure '{message}' against the normalized requirement; "
                "preserve every previously passing test and do not change an expected value unless its fixture "
                "trace proves that the expectation contradicts the requirement."
            )
        execution_output = "\n".join(
            str(preflight.get(field, ""))
            for field in ("stdout", "stderr")
        ).lower()
        if "read_text" in execution_output and any(
            signal in execution_output
            for signal in ("newline", "\\r\\n", "crlf", "line ending")
        ):
            failed_paths = preflight_failed_paths(preflight, prefix="tests/")
            path_label = ", ".join(failed_paths) or "failing newline-preservation tests"
            requirements.append(
                f"{path_label}: replace Path.read_text() output observation with Path.read_bytes(), "
                "binary mode, or Path.open(encoding='utf-8', newline=''); read_text() normalizes CRLF, "
                "LF, and CR and therefore cannot verify exact line-ending preservation."
            )
        return list(dict.fromkeys(requirements))
