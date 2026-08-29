import json
import re
from pathlib import Path
from typing import Any, Mapping

from core.forge.contracts import BuildSpec, ConditionalObligation, FeasiblePlan
from core.forge.conditional_test_evidence import (
    analyze_observation_fidelity,
    analyze_test_expectations,
)
from core.forge.execution import ProcessExecutor, SandboxProcessRequest




class ConditionalEvidenceValidator:
    def __init__(self, executor: ProcessExecutor, timeout_seconds: int) -> None:
        self.executor = executor
        self.timeout_seconds = timeout_seconds

    def validate(
        self,
        build_spec: BuildSpec,
        plan: FeasiblePlan,
        materialized: Mapping[str, Path],
        workspace: Path,
    ) -> tuple[list[str], list[str], dict[str, Any]]:
        failures: list[str] = []
        signatures: list[str] = []
        evidence: dict[str, Any] = {}

        hard_issues = [
            {
                "parent_requirement_id": issue.parent_requirement_id,
                "source_fragment": issue.source_fragment,
                "reason": issue.reason,
            }
            for issue in build_spec.conditional_normalization_issues
            if issue.hard
        ]
        evidence["normalization_issues"] = hard_issues
        if hard_issues:
            failures.append(
                "Hard compound conditions could not be normalized safely: "
                f"{hard_issues}."
            )
            self._append_unique(signatures, "uncompiled_hard_conditional")

        branch_coverage = self._branch_coverage_evidence(build_spec, plan)
        evidence["branch_coverage"] = branch_coverage
        missing_branch_coverage = [
            obligation_id
            for obligation_id, item in branch_coverage.items()
            if not item["covered"]
        ]
        if missing_branch_coverage:
            failures.append(
                "Conditional obligations are missing branch-aware plan coverage: "
                f"{missing_branch_coverage}."
            )
            self._append_unique(signatures, "missing_conditional_coverage")

        contents = {
            path: target.read_text(encoding="utf-8")
            for path, target in materialized.items()
            if path.startswith("tests/") and path.endswith(".py") and target.exists()
        }
        expectation_evidence = analyze_test_expectations(contents, build_spec, plan)
        evidence["test_expectation_consistency"] = expectation_evidence
        contradictions = [
            item for item in expectation_evidence if item["contradictions"]
        ]
        if contradictions:
            failures.append(
                "Generated test expectations contradict conditional obligations: "
                f"{contradictions}."
            )
            self._append_unique(signatures, "test_expectation_contradiction")

        lossy_evidence = analyze_observation_fidelity(contents, build_spec, plan)
        evidence["observation_fidelity"] = lossy_evidence
        lossy_checks = [item for item in lossy_evidence if item["lossy_observations"]]
        if lossy_checks:
            failures.append(
                "Exact observations use lossy transformations and cannot count as evidence: "
                f"{lossy_checks}."
            )
            self._append_unique(signatures, "lossy_observation_fidelity")

        probe_evidence = self._run_branch_probes(build_spec, plan, workspace)
        evidence["validator_branch_probes"] = probe_evidence
        failed_probes = [item for item in probe_evidence if item["status"] == "failed"]
        unavailable_hard_probes = [
            item
            for item in probe_evidence
            if item["status"] == "unavailable" and item["required"]
        ]
        if failed_probes:
            failures.append(f"Validator-owned branch probes failed: {failed_probes}.")
            self._append_unique(signatures, "conditional_obligation_mismatch")
            self._append_unique(signatures, "semantic_content_mismatch")
        if unavailable_hard_probes:
            failures.append(
                "Hard deterministic branch probes could not be constructed: "
                f"{unavailable_hard_probes}."
            )
            self._append_unique(signatures, "conditional_probe_unavailable")

        directive_evidence = self._coverage_directive_evidence(build_spec)
        evidence["coverage_directives"] = directive_evidence
        unresolved_directives = [
            item for item in directive_evidence if item["unresolved_witness_classes"]
        ]
        if unresolved_directives:
            failures.append(
                "Coverage directives reference witness classes without behavioral obligations: "
                f"{unresolved_directives}."
            )
            self._append_unique(signatures, "missing_conditional_coverage")

        return failures, signatures, evidence

    @staticmethod
    def _branch_coverage_evidence(
        build_spec: BuildSpec,
        plan: FeasiblePlan,
    ) -> dict[str, dict[str, Any]]:
        evidence: dict[str, dict[str, Any]] = {}
        for obligation in build_spec.conditional_obligations:
            coverage = plan.conditional_obligation_coverage.get(
                obligation.obligation_id,
                {},
            )
            tests = list(coverage.get("tests", []))
            structural = obligation.verification_method in {
                "interface_contract",
                "static_analysis",
            }
            evidence[obligation.obligation_id] = {
                "parent_requirement_id": obligation.parent_requirement_id,
                "trigger": obligation.trigger,
                "witness_class": obligation.witness_class,
                "tests": tests,
                "covered": structural or bool(tests),
            }
        return evidence

    def _run_branch_probes(
        self,
        build_spec: BuildSpec,
        plan: FeasiblePlan,
        workspace: Path,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[ConditionalObligation]] = {}
        for obligation in build_spec.conditional_obligations:
            if obligation.verification_method != "deterministic_probe":
                continue
            grouped.setdefault(obligation.obligation_id.rsplit(".O", 1)[0], []).append(obligation)

        results: list[dict[str, Any]] = []
        for branch_obligations in grouped.values():
            lead = branch_obligations[0]
            probe = self._build_cli_probe(build_spec, plan, lead, workspace)
            if probe is None:
                required = self._has_derivable_cli_probe(
                    build_spec,
                    plan,
                    lead.witness_class,
                )
                results.append(
                    {
                        "obligation_ids": [item.obligation_id for item in branch_obligations],
                        "witness_class": lead.witness_class,
                        "status": "unavailable",
                        "required": required,
                        "reason": "no_deterministic_witness_for_declared_interface",
                    }
                )
                continue
            completed = self.executor.run(
                SandboxProcessRequest(
                    command=["python", "-c", probe["script"]],
                    workspace=workspace,
                    timeout_seconds=self.timeout_seconds,
                    environment={
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": "src",
                    },
                )
            )
            observed = self._decode_probe_result(completed.stdout)
            checks = [
                self._compare_observation(obligation, observed)
                for obligation in branch_obligations
            ]
            passed = completed.returncode == 0 and observed is not None and all(
                item["passed"] for item in checks
            )
            results.append(
                {
                    "obligation_ids": [item.obligation_id for item in branch_obligations],
                    "witness_class": lead.witness_class,
                    "status": "passed" if passed else "failed",
                    "required": True,
                    "probe_inputs": probe["evidence"],
                    "observed": observed,
                    "checks": checks,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                    "backend": completed.backend,
                    "launch_error": completed.launch_error,
                }
            )
        return results

    @staticmethod
    def _has_derivable_cli_probe(
        build_spec: BuildSpec,
        plan: FeasiblePlan,
        witness_class: str,
    ) -> bool:
        supported = {
            "empty_input",
            "numeric_argument_exceeds_input_length",
            "invalid_positive_integer",
            "invalid_argument_count",
            "file_read_failure",
            "utf8_decode_failure",
        }
        if witness_class not in supported:
            return False
        interface = next(
            (
                item
                for item in plan.interfaces
                if item.interface_type == "cli_entrypoint" and item.module_path
            ),
            None,
        )
        if interface is None:
            return False
        return bool(
            interface.explicit_argv_count
            or ConditionalEvidenceValidator._declared_argv_count(
                build_spec.normalized_requirement
            )
        )

    def _build_cli_probe(
        self,
        build_spec: BuildSpec,
        plan: FeasiblePlan,
        obligation: ConditionalObligation,
        workspace: Path,
    ) -> dict[str, Any] | None:
        interface = next(
            (
                item
                for item in plan.interfaces
                if item.interface_type == "cli_entrypoint" and item.module_path
            ),
            None,
        )
        if interface is None or not obligation.witness_class:
            return None
        argument_count = interface.explicit_argv_count or self._declared_argv_count(
            build_spec.normalized_requirement
        )
        if argument_count is None:
            return None

        args = ["sample" for _ in range(argument_count)]
        filename_index = self._argv_index(build_spec.normalized_requirement, r"filename|file\s+path")
        numeric_index = self._argv_index(
            build_spec.normalized_requirement,
            r"chunk\s+size|size|count|limit|shift",
        )
        fixture_path = workspace / ".forge_branch_probe_input"
        if filename_index is not None and filename_index < len(args):
            args[filename_index] = str(fixture_path)
        if numeric_index is not None and numeric_index < len(args):
            args[numeric_index] = "2"

        witness = obligation.witness_class
        fixture_argument = ".forge_branch_probe_input"
        if witness == "empty_input" and filename_index is not None:
            fixture_path.write_bytes(b"")
        elif witness == "numeric_argument_exceeds_input_length" and filename_index is not None and numeric_index is not None:
            fixture_path.write_text("abc", encoding="utf-8")
            args[numeric_index] = "4"
        elif witness == "invalid_positive_integer" and numeric_index is not None:
            if filename_index is not None:
                fixture_path.write_text("sample", encoding="utf-8")
            args[numeric_index] = "0"
        elif witness == "invalid_argument_count":
            args = args[:-1]
        elif witness == "file_read_failure" and filename_index is not None:
            args[filename_index] = ".forge_missing_branch_probe_input"
        elif witness == "utf8_decode_failure" and filename_index is not None:
            fixture_path.write_bytes(b"\xff\xfe")
        else:
            return None

        if filename_index is not None and witness != "file_read_failure":
            args[filename_index] = fixture_argument

        script = (
            "import contextlib, importlib, io, json, sys\n"
            f"target = getattr(importlib.import_module({interface.module_path!r}), {interface.name!r})\n"
            f"args = {args!r}\n"
            "stdout = io.StringIO()\n"
            "stderr = io.StringIO()\n"
            "result = None\n"
            "exception = None\n"
            "try:\n"
            "    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):\n"
            "        result = target(args)\n"
            "except SystemExit as exc:\n"
            "    result = exc.code\n"
            "except BaseException as exc:\n"
            "    exception = type(exc).__name__\n"
            "payload = {'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(), "
            "'exit_code': result, 'return_value': result, 'exception': exception}\n"
            "print(json.dumps(payload, ensure_ascii=False), file=sys.__stdout__)\n"
        )
        return {
            "script": script,
            "evidence": {
                "module": interface.module_path,
                "callable": interface.name,
                "argv": args,
                "witness_class": witness,
            },
        }

    @staticmethod
    def _declared_argv_count(requirement: str) -> int | None:
        indexes = [int(value) for value in re.findall(r"argv\s*\[\s*(\d+)\s*\]", requirement, re.IGNORECASE)]
        if not indexes:
            return None
        return max(indexes)

    @staticmethod
    def _argv_index(requirement: str, role_pattern: str) -> int | None:
        forward = re.search(
            rf"(?:{role_pattern}).{{0,80}}argv\s*\[\s*(\d+)\s*\]",
            requirement,
            re.IGNORECASE,
        )
        reverse = re.search(
            rf"argv\s*\[\s*(\d+)\s*\].{{0,80}}(?:{role_pattern})",
            requirement,
            re.IGNORECASE,
        )
        match = forward or reverse
        if match is None:
            return None
        return max(0, int(match.group(1)) - 1)

    @staticmethod
    def _decode_probe_result(stdout: str) -> dict[str, Any] | None:
        for line in reversed(stdout.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "stdout" in value and "stderr" in value:
                return value
        return None

    @staticmethod
    def _compare_observation(
        obligation: ConditionalObligation,
        observed: dict[str, Any] | None,
    ) -> dict[str, Any]:
        actual = None if observed is None else observed.get(obligation.observable_channel)
        relation = obligation.comparison_relation
        expected = obligation.expected_value
        if relation == "equals":
            passed = actual == expected
        elif relation == "not_equals":
            passed = actual != expected
        elif relation == "contains":
            passed = actual is not None and expected in actual
        elif relation == "not_contains":
            passed = actual is not None and expected not in actual
        elif relation == "raises":
            passed = actual == expected
        else:
            passed = False
        return {
            "obligation_id": obligation.obligation_id,
            "channel": obligation.observable_channel,
            "relation": relation,
            "expected": expected,
            "actual": actual,
            "fidelity": obligation.observation_fidelity,
            "passed": passed,
        }

    @staticmethod
    def _coverage_directive_evidence(build_spec: BuildSpec) -> list[dict[str, Any]]:
        known_witnesses = {
            obligation.witness_class
            for obligation in build_spec.conditional_obligations
            if obligation.witness_class
        }
        return [
            {
                "directive_id": directive.directive_id,
                "parent_requirement_id": directive.parent_requirement_id,
                "referenced_obligation_ids": list(directive.referenced_obligation_ids),
                "witness_classes": list(directive.witness_classes),
                "unresolved_witness_classes": sorted(
                    set(directive.witness_classes) - known_witnesses
                ),
            }
            for directive in build_spec.coverage_directives
        ]

    @staticmethod
    def _append_unique(values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)
