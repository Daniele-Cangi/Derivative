import ast
import re
from typing import Any, Callable, ClassVar

from core.forge.contracts import (
    CodeArtifact,
    FeasiblePlan,
    RepairDirective,
    RepairPatchCandidate,
    ValidationArtifact,
)
from core.forge.execution import ProcessExecutor
from core.forge.candidate_preflight import (
    run_fixture_oracle_preflight,
    run_semantic_preflight,
)
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
    _PREFLIGHT_PHASE_RANK: ClassVar[dict[str, int]] = {
        "backend_unavailable": -2,
        "candidate_completeness": -1,
        "fixture_oracle": 0,
        "materialization": 1,
        "syntax": 2,
        "import": 3,
        "tests": 4,
        "semantic_contract": 5,
    }

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
                    "For deterministic transformations, derive expected output from the exact fixture with a "
                    "source-independent reference operation instead of manually transcribing transformed literals.",
                    "When a requirement preserves exact bytes or CRLF/LF/CR line endings, observe output "
                    "with read_bytes(), binary mode, or open(..., newline=''); Path.read_text() normalizes "
                    "newlines and is not valid evidence.",
                    "Python string literals already contain decoded Unicode; never round-trip them through "
                    "encode('utf-8').decode('unicode_escape') when constructing fixtures or expectations.",
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
        baseline_preflight = self._run_preflight(
            current_targets,
            plan,
            context["test_generation_contracts"],
            test_paths,
        )
        baseline_quality = self._preflight_quality(baseline_preflight)
        attempts: list[dict[str, Any]] = []
        safe_candidate_files = dict(current_targets)
        rejected_paths: list[str] = []
        candidate_files: dict[str, str] = {}
        selected_candidate_files: dict[str, str] = {}
        selected_preflight: dict[str, Any] = {}
        selected_quality: dict[str, Any] | None = None
        complete_candidate_count = 0
        regression_rejected_attempts: list[int] = []
        backend_available = False
        active_paths = list(target_paths)

        for attempt in range(self.max_preflight_corrections + 1):
            attempt_context = dict(context)
            attempt_context["candidate_compilation_attempt"] = attempt + 1
            attempt_context["current_target_paths"] = list(active_paths)
            attempt_context["candidate_transaction_correction"] = bool(attempts)
            preserved_paths = sorted(set(target_paths) - set(active_paths))
            previous_phase = (
                str(attempts[-1]["preflight"].get("phase", ""))
                if attempts
                else ""
            )
            attempt_context["preserve_passing_paths"] = (
                [] if previous_phase == "fixture_oracle" else preserved_paths
            )
            attempt_context["preserve_unvalidated_paths"] = (
                preserved_paths if previous_phase == "fixture_oracle" else []
            )
            if candidate_files and preserved_paths:
                attempt_context["preserved_candidate_files"] = {
                    path: candidate_files[path]
                    for path in preserved_paths
                    if path in candidate_files
                }
                if previous_phase == "fixture_oracle":
                    attempt_context["preservation_contract"] = (
                        "These files were not implicated by static fixture-oracle sanity but remain "
                        "behaviorally unvalidated. Keep them unchanged for this targeted oracle correction; "
                        "a later executable preflight may make them revisable."
                    )
                else:
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

            preflight = self._run_preflight(
                merged_candidate,
                plan,
                context["test_generation_contracts"],
                test_paths,
            )
            attempt_record["preflight"] = preflight
            candidate_files = merged_candidate
            complete_candidate_count += 1
            if not preflight.get("passed", False):
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
                if preflight.get("phase") == "fixture_oracle":
                    preflight["behaviorally_unvalidated_paths"] = list(target_paths)
                    active_paths = [path for path in failed if path in target_paths]
                elif preflight.get("phase") not in {"tests", "semantic_contract"}:
                    preflight["behaviorally_unvalidated_paths"] = list(target_paths)
                    active_paths = list(target_paths)
                else:
                    active_paths = [path for path in failed if path in target_paths]
                if not active_paths:
                    active_paths = list(target_paths)
                current_targets = {
                    path: candidate_files[path]
                    for path in active_paths
                }

            quality = self._preflight_quality(preflight)
            regresses_from_baseline = self._preflight_regresses(
                quality,
                baseline_quality,
            )
            attempt_record["preflight_quality"] = quality
            attempt_record["regresses_from_baseline"] = regresses_from_baseline
            candidate_selected = False
            if regresses_from_baseline:
                regression_rejected_attempts.append(attempt + 1)
            elif selected_quality is None or self._preflight_score(
                quality
            ) >= self._preflight_score(selected_quality):
                selected_candidate_files = dict(merged_candidate)
                safe_candidate_files = dict(merged_candidate)
                selected_preflight = dict(preflight)
                selected_quality = quality
                candidate_selected = True
            if not candidate_selected:
                candidate_files = dict(safe_candidate_files)
                current_targets = {
                    path: candidate_files[path]
                    for path in active_paths
                }
            attempt_record["selected_for_handoff"] = candidate_selected
            attempt_record["working_state_restored"] = not candidate_selected
            attempts.append(attempt_record)
            if preflight.get("passed", False):
                break

        candidate_files = selected_candidate_files
        preflight_passed = bool(
            selected_preflight.get("passed", False)
            and set(candidate_files) == set(target_paths)
            and not rejected_paths
        )
        complete_transaction = bool(
            set(candidate_files) == set(target_paths)
            and not rejected_paths
        )
        if not complete_transaction:
            candidate_files = {}

        evidence = {
            "repair_id": directive.repair_id,
            "operations": list(directive.operations),
            "allowed_paths": target_paths,
            "planned_paths": target_paths,
            "accepted_paths": sorted(candidate_files),
            "rejected_paths": sorted(set(rejected_paths)),
            "complete_transaction": complete_transaction,
            "preflight_passed": preflight_passed,
            "baseline_preflight": baseline_preflight,
            "baseline_preflight_quality": baseline_quality,
            "selected_preflight": selected_preflight,
            "selected_preflight_quality": selected_quality or {},
            "complete_candidate_count": complete_candidate_count,
            "regression_rejected": bool(
                complete_candidate_count
                and not candidate_files
                and regression_rejected_attempts
            ),
            "regression_rejected_attempts": regression_rejected_attempts,
            "handoff_status": (
                "preflight_passed"
                if preflight_passed
                else "validator_repair_required"
                if candidate_files
                else "rejected"
            ),
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
        if preflight_passed:
            stop_reason = ""
        elif candidate_files:
            stop_reason = "Complete candidate transaction requires validator-guided repair."
        elif evidence["regression_rejected"]:
            stop_reason = (
                "All generated candidate transactions regressed from the current artifact preflight."
            )
        else:
            stop_reason = backend_reason or "Complete candidate transaction did not pass executable preflight."
        return RepairPatchCandidate(
            backend_name=self.backend_name,
            files=candidate_files,
            evidence=evidence,
            rejected_paths=sorted(set(rejected_paths)),
            available=backend_available,
            stop_reason=stop_reason,
        )

    def _run_preflight(
        self,
        candidate_files: dict[str, str],
        plan: FeasiblePlan,
        contracts: dict[str, Any],
        test_paths: list[str],
    ) -> dict[str, Any]:
        preflight = run_fixture_oracle_preflight(
            candidate_files,
            plan,
            contracts,
        )
        if preflight.get("passed", False):
            preflight = self.test_preflight_runner(candidate_files, test_paths)
        if preflight.get("passed", False):
            preflight = run_semantic_preflight(
                candidate_files,
                plan,
                contracts,
                preflight,
            )
        return preflight

    @classmethod
    def _preflight_quality(cls, preflight: dict[str, Any]) -> dict[str, Any]:
        passed = bool(preflight.get("passed", False))
        phase = str(preflight.get("phase", "unknown"))
        failed_paths = preflight_failed_paths(preflight)
        failures = [
            item
            for field in ("failures", "failure_details")
            for item in preflight.get(field, [])
        ]
        return {
            "passed": passed,
            "phase": phase,
            "phase_rank": max(cls._PREFLIGHT_PHASE_RANK.values()) + 1
            if passed
            else cls._PREFLIGHT_PHASE_RANK.get(phase, -3),
            "failed_path_count": len(set(failed_paths)),
            "failure_count": len(failures),
        }

    @staticmethod
    def _preflight_score(quality: dict[str, Any]) -> tuple[int, int, int]:
        """Rank phases first and use failure counts only to break phase ties."""

        return (
            int(quality.get("phase_rank", -3)),
            -int(quality.get("failed_path_count", 0)),
            -int(quality.get("failure_count", 0)),
        )

    @classmethod
    def _preflight_regresses(
        cls,
        candidate_quality: dict[str, Any],
        baseline_quality: dict[str, Any],
    ) -> bool:
        return cls._preflight_score(candidate_quality) < cls._preflight_score(
            baseline_quality
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
        for failure in preflight.get("failures", []):
            if not isinstance(failure, dict):
                continue
            kind = str(failure.get("kind", ""))
            if kind not in {"syntax_error", "import_error", "import_failure"}:
                continue
            path = str(failure.get("path", "candidate file"))
            line = failure.get("line")
            location = f" at line {line}" if line is not None else ""
            message = str(failure.get("message", kind))
            requirements.append(
                f"{path}: resolve {kind}{location}: {message}. The transaction has not completed "
                "executable preflight, so all supplied paths remain behaviorally unvalidated."
            )
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
        if "unicode_escape" in execution_output and re.search(
            r"encode\(\s*['\"]utf-8['\"]\s*\)", execution_output
        ):
            requirements.append(
                "Generated Python test fixtures already hold decoded Unicode characters: remove chained "
                "encode('utf-8').decode('unicode_escape') conversions from inputs and expectations, then "
                "compare the direct Unicode value or its single UTF-8 encoding."
            )
        return list(dict.fromkeys(requirements))
