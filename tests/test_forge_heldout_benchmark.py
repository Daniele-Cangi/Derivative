import json
from pathlib import Path

import pytest

from core.forge.benchmark import (
    TERMINAL_INFEASIBLE_PROVEN,
    TERMINAL_VALIDATION_FAILED,
    TERMINAL_VERIFIED,
)
from core.forge.contracts import ForgeResult, ForgeRoute, ForgeRunMetrics
from core.forge.execution import ExecutionPolicy, SandboxProcessResult
from core.forge.heldout_benchmark import (
    HeldoutBenchmarkCase,
    HeldoutThresholds,
    OracleResult,
    OracleSpec,
    bundled_heldout_dataset_path,
    evaluate_heldout_thresholds,
    execute_pytest_oracle,
    inspect_oracle_sanity,
    load_heldout_cases,
    persist_heldout_summary,
    render_heldout_summary,
    run_heldout_cases,
)


from core.forge.oracle_contract import oracle_contract_mismatches

class _OracleRecordingExecutor:
    def __init__(self):
        self.policy = ExecutionPolicy(backend="docker")
        self.request = None
        self.staged_files = []

    def run(self, request):
        self.request = request
        self.staged_files = sorted(
            path.relative_to(request.workspace).as_posix()
            for path in request.workspace.rglob("*")
            if path.is_file()
        )
        return SandboxProcessResult(
            returncode=0,
            stdout="1 passed\n",
            stderr="",
            backend="docker",
            execution_time_seconds=0.02,
            isolation=self.policy.evidence(),
        )


def _forge_result(
    status: str,
    artifact_path: str = "",
    run_metrics: ForgeRunMetrics | None = None,
) -> ForgeResult:
    routes = {
        TERMINAL_VERIFIED: ForgeRoute.TERMINAL_VERIFIED,
        TERMINAL_VALIDATION_FAILED: ForgeRoute.TERMINAL_VALIDATION_FAILED,
        TERMINAL_INFEASIBLE_PROVEN: ForgeRoute.TERMINAL_INFEASIBLE,
    }
    return ForgeResult(
        route=routes[status],
        terminal_status=status,
        summary=status,
        artifact_path=artifact_path,
        run_metrics=run_metrics
        or ForgeRunMetrics(
            validation_attempts=1 if status != TERMINAL_INFEASIBLE_PROVEN else 0,
            verified_at_1=status == TERMINAL_VERIFIED,
        ),
    )


def test_bundled_heldout_dataset_is_frozen_and_oracle_complete():
    cases = load_heldout_cases(bundled_heldout_dataset_path())

    assert len(cases) == 15
    assert len({case.case_id for case in cases}) == 15
    verified = [case for case in cases if case.expected_terminal_status == TERMINAL_VERIFIED]
    assert len(verified) == 8
    assert all(case.oracle is not None for case in verified)
    assert all(Path(case.oracle.path).is_file() for case in verified if case.oracle)


def test_loader_rejects_verified_case_without_oracle(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "case_id": "X001",
                    "requirement": "Build a feasible library.",
                    "expected_terminal_status": TERMINAL_VERIFIED,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires an external oracle"):
        load_heldout_cases(str(dataset))


def test_loader_rejects_oracle_outside_benchmark_directory(tmp_path):
    dataset_root = tmp_path / "benchmark"
    dataset_root.mkdir()
    outside = tmp_path / "outside_oracle.py"
    outside.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    dataset = dataset_root / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "case_id": "X001",
                    "requirement": "Build a feasible library.",
                    "expected_terminal_status": TERMINAL_VERIFIED,
                    "oracle": {"path": "../outside_oracle.py"},
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes the benchmark directory"):
        load_heldout_cases(str(dataset))


def test_pytest_oracle_executes_against_packaged_source(tmp_path):
    package_root = tmp_path / "package"
    source_root = package_root / "src"
    source_root.mkdir(parents=True)
    (source_root / "calculator.py").write_text(
        "def add(left, right):\n    return left + right\n",
        encoding="utf-8",
    )
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(
        "from calculator import add\n\n"
        "def test_external_contract():\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    result = execute_pytest_oracle(
        OracleSpec(path=str(oracle_path), timeout_seconds=20),
        str(package_root),
    )

    assert result.executed is True
    assert result.passed is True
    assert result.exit_code == 0
    assert "1 passed" in result.stdout


def test_pytest_oracle_failure_is_external_evidence(tmp_path):
    package_root = tmp_path / "package"
    source_root = package_root / "src"
    source_root.mkdir(parents=True)
    (source_root / "calculator.py").write_text(
        "def add(left, right):\n    return left - right\n",
        encoding="utf-8",
    )
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(
        "from calculator import add\n\n"
        "def test_external_contract():\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    result = execute_pytest_oracle(
        OracleSpec(path=str(oracle_path), timeout_seconds=20),
        str(package_root),
    )

    assert result.executed is True
    assert result.passed is False
    assert result.exit_code == 1
    assert "1 failed" in result.stdout


def test_pytest_oracle_stages_only_package_and_oracle_for_executor(tmp_path):
    package_root = tmp_path / "original-package"
    source_root = package_root / "src"
    source_root.mkdir(parents=True)
    (source_root / "calculator.py").write_text(
        "def add(left, right):\n    return left + right\n",
        encoding="utf-8",
    )
    oracle_path = tmp_path / "private-oracle.py"
    oracle_path.write_text("def test_external():\n    assert True\n", encoding="utf-8")
    executor = _OracleRecordingExecutor()

    result = execute_pytest_oracle(
        OracleSpec(path=str(oracle_path), timeout_seconds=20),
        str(package_root),
        executor=executor,
    )

    assert result.passed is True
    assert result.backend == "docker"
    assert result.isolation["isolated"] is True
    assert executor.request is not None
    assert executor.request.working_directory == "package"
    assert "../oracle.py" in executor.request.command
    assert executor.request.environment["PYTHONPATH"] == "src"
    assert executor.staged_files == ["oracle.py", "package/src/calculator.py"]
    assert str(package_root) not in " ".join(executor.request.command)
    assert str(oracle_path) not in " ".join(executor.request.command)


def test_heldout_metrics_require_external_oracle_success():
    oracle = OracleSpec(path="oracle.py")
    cases = [
        HeldoutBenchmarkCase("A", "verified-pass", TERMINAL_VERIFIED, oracle=oracle),
        HeldoutBenchmarkCase("B", "verified-missed", TERMINAL_VERIFIED, oracle=oracle),
        HeldoutBenchmarkCase("C", "must-fail", TERMINAL_VALIDATION_FAILED),
        HeldoutBenchmarkCase("D", "impossible", TERMINAL_INFEASIBLE_PROVEN),
    ]
    observed = {
        "verified-pass": _forge_result(TERMINAL_VERIFIED, "pkg-a"),
        "verified-missed": _forge_result(TERMINAL_VALIDATION_FAILED),
        "must-fail": _forge_result(TERMINAL_VERIFIED, "pkg-c"),
        "impossible": _forge_result(TERMINAL_INFEASIBLE_PROVEN),
    }

    summary = run_heldout_cases(
        cases,
        run_case=lambda requirement: observed[requirement],
        run_oracle=lambda spec, package: OracleResult(
            executed=True,
            passed=package == "pkg-a",
            exit_code=0 if package == "pkg-a" else 1,
        ),
    )

    assert summary.total_cases == 4
    assert summary.passed_cases == 2
    assert summary.status_accuracy == 0.5
    assert summary.external_verified_at_1 == 0.5
    assert summary.oracle_pass_rate == 1.0
    assert summary.external_false_verified_rate == 0.5
    assert summary.infeasible_detection_rate == 1.0


def test_oracle_failure_turns_matching_verified_status_into_failed_case():
    oracle = OracleSpec(path="oracle.py")
    summary = run_heldout_cases(
        [HeldoutBenchmarkCase("A", "candidate", TERMINAL_VERIFIED, oracle=oracle)],
        run_case=lambda requirement: _forge_result(TERMINAL_VERIFIED, "pkg"),
        run_oracle=lambda spec, package: OracleResult(executed=True, passed=False, exit_code=1),
    )

    result = summary.case_results[0]
    assert result.status_matched is True
    assert result.passed is False
    assert summary.status_accuracy == 1.0
    assert summary.external_verified_at_1 == 0.0
    assert summary.external_false_verified_rate == 1.0


def test_oracle_sanity_detects_word_boundary_contradiction(tmp_path):
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(
        "def test_stdin_to_file():\n"
        "    input_content = 'x yz\\n'\n"
        "    expected = 'x z y\\n'\n"
        "    assert expected\n",
        encoding="utf-8",
    )
    case = HeldoutBenchmarkCase(
        "V4-001",
        (
            "Reverse every word defined as a sequence of non-whitespace characters "
            "separated by ASCII whitespace, with word order preserved."
        ),
        TERMINAL_VERIFIED,
        oracle=OracleSpec(path=str(oracle_path)),
    )

    result = inspect_oracle_sanity(case)

    assert result is not None
    assert result.valid is False
    assert result.executed is False
    assert result.error == "oracle_invalid: fixture expectations contradict the requirement"
    assert result.sanity_failures == [
        {
            "capability_id": "reverse_ascii_whitespace_tokens",
            "function": "test_stdin_to_file",
            "input_name": "input_content",
            "expected_name": "expected",
            "input_line": 2,
            "expected_line": 3,
            "declared_expected": "'x z y\\n'",
            "derived_expected": "'x zy\\n'",
        }
    ]


def test_oracle_sanity_rejects_injected_cli_name_before_forge(tmp_path):
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(
        "from pycolmask import main\n\n"
        "def test_nominal_masking(tmp_path):\n"
        "    argv = ['pycolmask', str(tmp_path / 'in.csv'), '--mask=1']\n"
        "    rc = main(argv)\n"
        "    assert rc == 0\n",
        encoding="utf-8",
    )
    requirement = (
        "Implement a verified CLI utility named 'pycolmask' that reads a CSV file. "
        "The main(argv: list[str] | None = None) -> int contract must be importable."
    )
    case = HeldoutBenchmarkCase(
        "V5-001",
        requirement,
        TERMINAL_VERIFIED,
        oracle=OracleSpec(path=str(oracle_path)),
    )
    forge_calls: list[str] = []

    summary = run_heldout_cases(
        [case],
        run_case=lambda current_requirement: (
            forge_calls.append(current_requirement)
            or _forge_result(TERMINAL_VERIFIED, "pkg")
        ),
    )

    assert forge_calls == []
    assert summary.invalid_oracle_cases == 1
    assert summary.invalid_benchmark_rejection_rate == 1.0
    assert summary.case_results[0].observed_terminal_status == "oracle_invalid"
    assert summary.case_results[0].model_request_count == 0
    assert summary.case_results[0].oracle_result is not None
    assert summary.case_results[0].oracle_result.error == (
        "oracle_invalid: invocation contract contradicts the requirement"
    )
    assert summary.case_results[0].oracle_result.sanity_failures == [
        {
            "contract_id": "in_process_main_argv",
            "function": "test_nominal_masking",
            "call_line": 5,
            "declared_cli_name": "pycolmask",
            "argument_name": "argv",
            "first_argument": "pycolmask",
            "message": (
                "oracle injects the declared CLI name as argv[0], but the "
                "requirement does not define main(argv) as full sys.argv"
            ),
        }
    ]


def test_oracle_sanity_allows_cli_name_when_requirement_defines_argv_zero(tmp_path):
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(
        "from pycolmask import main\n\n"
        "def test_nominal_masking():\n"
        "    rc = main(['pycolmask', 'in.csv', '--mask=1'])\n"
        "    assert rc == 0\n",
        encoding="utf-8",
    )
    case = HeldoutBenchmarkCase(
        "explicit-argv-zero",
        (
            "Implement a CLI utility named 'pycolmask'. The main(argv) contract "
            "receives full sys.argv, including the program name in argv[0]."
        ),
        TERMINAL_VERIFIED,
        oracle=OracleSpec(path=str(oracle_path)),
    )

    assert inspect_oracle_sanity(case) is None


def test_oracle_sanity_rejects_unusable_context_manager_binding(tmp_path):
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(
        "class capture:\n"
        "    def __enter__(self):\n"
        "        self.stdout = object()\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        result = 1\n"
        "    assert redir.stdout\n",
        encoding="utf-8",
    )
    case = HeldoutBenchmarkCase(
        "broken-context-binding",
        "Build an importable function and include tests.",
        TERMINAL_VERIFIED,
        oracle=OracleSpec(path=str(oracle_path)),
    )

    result = inspect_oracle_sanity(case)

    assert result is not None
    assert result.valid is False
    assert result.executed is False
    assert result.error == (
        "oracle_invalid: test harness context binding is unusable"
    )
    assert result.sanity_failures == [
        {
            "contract_id": "context_manager_binding",
            "function": "test_output",
            "class_name": "capture",
            "context_line": 8,
            "enter_line": 2,
            "bound_name": "redir",
            "message": (
                "capture.__enter__ does not return a non-None value on every path bound as redir, "
                "but the oracle dereferences it"
            ),
        }
    ]


def test_oracle_sanity_matches_sync_and_async_context_protocols(tmp_path):
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(
        "class dual_capture:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    async def __aenter__(self):\n"
        "        self.stdout = object()\n"
        "    def __exit__(self, *args):\n"
        "        pass\n"
        "    async def __aexit__(self, *args):\n"
        "        pass\n\n"
        "def test_sync():\n"
        "    with dual_capture() as redir:\n"
        "        assert redir.stdout\n\n"
        "async def test_async():\n"
        "    async with dual_capture() as redir:\n"
        "        assert redir.stdout\n",
        encoding="utf-8",
    )
    case = HeldoutBenchmarkCase(
        "dual-context-protocol",
        "Build an importable function and include tests.",
        TERMINAL_VERIFIED,
        oracle=OracleSpec(path=str(oracle_path)),
    )

    result = inspect_oracle_sanity(case)

    assert result is not None
    assert result.valid is False
    assert len(result.sanity_failures) == 1
    assert result.sanity_failures[0]["function"] == "test_async"
    assert "dual_capture.__aenter__" in result.sanity_failures[0]["message"]


def test_oracle_sanity_ignores_rebound_and_nested_shadowed_names(tmp_path):
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(
        "class capture:\n"
        "    def __enter__(self):\n"
        "        self.stdout = object()\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_rebound():\n"
        "    with capture() as redir:\n"
        "        pass\n"
        "    redir = object()\n"
        "    assert redir.stdout\n\n"
        "def test_nested_shadow():\n"
        "    with capture() as redir:\n"
        "        pass\n"
        "    def inspect(redir):\n"
        "        return redir.stdout\n"
        "    assert inspect\n",
        encoding="utf-8",
    )
    case = HeldoutBenchmarkCase(
        "rebound-context-name",
        "Build an importable function and include tests.",
        TERMINAL_VERIFIED,
        oracle=OracleSpec(path=str(oracle_path)),
    )

    assert inspect_oracle_sanity(case) is None


def test_oracle_sanity_requires_non_none_return_on_every_path(tmp_path):
    partial_oracle = tmp_path / "partial_oracle.py"
    partial_oracle.write_text(
        "class capture:\n"
        "    def __enter__(self):\n"
        "        if enabled:\n"
        "            return self\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        assert redir.stdout\n",
        encoding="utf-8",
    )
    complete_oracle = tmp_path / "complete_oracle.py"
    complete_oracle.write_text(
        "class capture:\n"
        "    def __enter__(self):\n"
        "        if enabled:\n"
        "            return self\n"
        "        return self\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        assert redir.stdout\n",
        encoding="utf-8",
    )
    partial_case = HeldoutBenchmarkCase(
        "partial-context-return",
        "Build an importable function and include tests.",
        TERMINAL_VERIFIED,
        oracle=OracleSpec(path=str(partial_oracle)),
    )
    complete_case = HeldoutBenchmarkCase(
        "complete-context-return",
        "Build an importable function and include tests.",
        TERMINAL_VERIFIED,
        oracle=OracleSpec(path=str(complete_oracle)),
    )

    partial_result = inspect_oracle_sanity(partial_case)

    assert partial_result is not None
    assert partial_result.valid is False
    assert partial_result.sanity_failures[0]["contract_id"] == (
        "context_manager_binding"
    )
    assert inspect_oracle_sanity(complete_case) is None

def test_oracle_sanity_distinguishes_sync_and_async_generator_enter():
    sync_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        yield self\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        assert redir.stdout\n"
    )
    async_oracle = (
        "class capture:\n"
        "    async def __aenter__(self):\n"
        "        yield self\n"
        "        raise RuntimeError\n"
        "    async def __aexit__(self, *args):\n"
        "        pass\n\n"
        "async def test_output():\n"
        "    async with capture() as redir:\n"
        "        assert redir.stdout\n"
    )
    requirement = "Build an importable function and include tests."

    assert oracle_contract_mismatches(sync_oracle, requirement) == []
    async_mismatches = oracle_contract_mismatches(async_oracle, requirement)
    assert len(async_mismatches) == 1
    assert "capture.__aenter__" in async_mismatches[0].message


def test_oracle_sanity_handles_raise_and_finally_exit_paths():
    raise_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        if enabled:\n"
        "            raise RuntimeError\n"
        "        return self\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        assert redir.stdout\n"
    )
    incomplete_finally_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        try:\n"
        "            if enabled:\n"
        "                return self\n"
        "        finally:\n"
        "            cleanup()\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        assert redir.stdout\n"
    )
    complete_finally_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        try:\n"
        "            return None\n"
        "        finally:\n"
        "            return self\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        assert redir.stdout\n"
    )
    requirement = "Build an importable function and include tests."

    assert oracle_contract_mismatches(raise_oracle, requirement) == []
    incomplete = oracle_contract_mismatches(
        incomplete_finally_oracle, requirement
    )
    assert len(incomplete) == 1
    assert incomplete[0].contract_id == "context_manager_binding"
    assert oracle_contract_mismatches(complete_finally_oracle, requirement) == []

def test_oracle_sanity_models_literal_infinite_loop_exits():
    returning_loop_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        while True:\n"
        "            return self\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        assert redir.stdout\n"
    )
    breaking_loop_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        while True:\n"
        "            if enabled:\n"
        "                break\n"
        "            return self\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        assert redir.stdout\n"
    )
    requirement = "Build an importable function and include tests."

    assert oracle_contract_mismatches(returning_loop_oracle, requirement) == []
    breaking = oracle_contract_mismatches(breaking_loop_oracle, requirement)
    assert len(breaking) == 1
    assert breaking[0].contract_id == "context_manager_binding"


def test_oracle_sanity_tracks_comprehension_capture_and_shadowing():
    capture_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        pass\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        values = [redir.stdout for item in items]\n"
    )
    iterable_capture_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        pass\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        values = [item for redir in redir.stdout]\n"
    )
    shadowed_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        pass\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        values = [redir.stdout for redir in items]\n"
    )
    requirement = "Build an importable function and include tests."

    captured = oracle_contract_mismatches(capture_oracle, requirement)
    iterable_captured = oracle_contract_mismatches(
        iterable_capture_oracle, requirement
    )
    assert len(captured) == 1
    assert captured[0].contract_id == "context_manager_binding"
    assert len(iterable_captured) == 1
    assert iterable_captured[0].contract_id == "context_manager_binding"
    assert oracle_contract_mismatches(shadowed_oracle, requirement) == []

def test_oracle_sanity_resolves_local_context_manager_classes():
    local_invalid_oracle = (
        "def test_output():\n"
        "    class capture:\n"
        "        def __enter__(self):\n"
        "            pass\n"
        "        def __exit__(self, *args):\n"
        "            pass\n"
        "    with capture() as redir:\n"
        "        assert redir.stdout\n"
    )
    local_shadow_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        pass\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    class capture:\n"
        "        def __enter__(self):\n"
        "            return self\n"
        "        def __exit__(self, *args):\n"
        "            pass\n"
        "    with capture() as redir:\n"
        "        assert redir.stdout\n"
    )
    requirement = "Build an importable function and include tests."

    local_invalid = oracle_contract_mismatches(
        local_invalid_oracle, requirement
    )
    assert len(local_invalid) == 1
    assert local_invalid[0].class_name == "capture"
    assert oracle_contract_mismatches(local_shadow_oracle, requirement) == []


def test_oracle_sanity_tracks_attribute_and_subscript_targets():
    attribute_target_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        pass\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        values = [item for redir.stdout in items]\n"
    )
    subscript_target_oracle = (
        "class capture:\n"
        "    def __enter__(self):\n"
        "        pass\n"
        "    def __exit__(self, *args):\n"
        "        pass\n\n"
        "def test_output():\n"
        "    with capture() as redir:\n"
        "        values = [item for redir[index] in items]\n"
    )
    requirement = "Build an importable function and include tests."

    attribute_target = oracle_contract_mismatches(
        attribute_target_oracle, requirement
    )
    subscript_target = oracle_contract_mismatches(
        subscript_target_oracle, requirement
    )
    assert len(attribute_target) == 1
    assert attribute_target[0].contract_id == "context_manager_binding"
    assert len(subscript_target) == 1
    assert subscript_target[0].contract_id == "context_manager_binding"

def test_frozen_v7_001_oracle_is_rejected_by_context_binding_gate():
    dataset = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "blind_v7"
        / "external_001"
        / "cases.json"
    )
    case = next(
        item for item in load_heldout_cases(str(dataset)) if item.case_id == "V7-001"
    )

    result = inspect_oracle_sanity(case)

    assert result is not None
    assert result.valid is False
    assert result.executed is False
    assert result.error == (
        "oracle_invalid: test harness context binding is unusable"
    )
    assert result.sanity_failures
    assert {
        item["contract_id"] for item in result.sanity_failures
    } == {"context_manager_binding"}
    assert {
        item["class_name"] for item in result.sanity_failures
    } == {"redirect_std"}
    assert {
        item["enter_line"] for item in result.sanity_failures
    } == {54}


def test_frozen_v5_001_oracle_is_rejected_by_invocation_contract_gate():
    dataset = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "blind_v5"
        / "external_001"
        / "cases.json"
    )
    case = next(
        item for item in load_heldout_cases(str(dataset)) if item.case_id == "V5-001"
    )

    result = inspect_oracle_sanity(case)

    assert result is not None
    assert result.valid is False
    assert result.executed is False
    assert result.error == (
        "oracle_invalid: invocation contract contradicts the requirement"
    )
    assert len(result.sanity_failures) == 10
    assert {item["contract_id"] for item in result.sanity_failures} == {
        "in_process_main_argv"
    }
    assert {item["declared_cli_name"] for item in result.sanity_failures} == {
        "pycolmask"
    }


def test_frozen_v5_004_oracle_is_rejected_by_explicit_pattern_gate():
    dataset = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "blind_v5"
        / "external_001"
        / "cases.json"
    )
    case = next(
        item for item in load_heldout_cases(str(dataset)) if item.case_id == "V5-004"
    )

    result = inspect_oracle_sanity(case)

    assert result is not None
    assert result.valid is False
    assert result.executed is False
    assert result.error == (
        "oracle_invalid: acceptance examples contradict an explicit requirement pattern"
    )
    assert result.sanity_failures == [
        {
            "contract_id": "explicit_regex_fixture",
            "function": "<module>",
            "fixture_name": "INVALID_LINES",
            "fixture_line": 96,
            "declared_pattern": "^[A-Z_][A-Z0-9_]*=[^\\n]*$",
            "sample": "FOO=bar extra\n",
            "oracle_classification": "invalid",
            "derived_classification": "valid",
            "message": (
                "fixture INVALID_LINES classifies the sample as invalid, but the "
                "requirement's explicit pattern classifies it as valid"
            ),
        }
    ]


def test_frozen_v5_003_requirement_is_rejected_before_forge():
    dataset = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "blind_v5"
        / "external_001"
        / "cases.json"
    )
    case = next(
        item for item in load_heldout_cases(str(dataset)) if item.case_id == "V5-003"
    )
    forge_calls: list[str] = []

    summary = run_heldout_cases(
        [case],
        run_case=lambda requirement: (
            forge_calls.append(requirement)
            or _forge_result(TERMINAL_VERIFIED, "pkg")
        ),
    )

    result = summary.case_results[0]
    assert forge_calls == []
    assert result.observed_terminal_status == "oracle_invalid"
    assert result.model_request_count == 0
    assert result.oracle_result is not None
    assert result.oracle_result.executed is False
    assert result.oracle_result.valid is False
    assert result.oracle_result.error == (
        "oracle_invalid: expected terminal status contradicts a finite "
        "requirement witness"
    )
    assert result.oracle_result.sanity_failures[0]["contract_id"] == (
        "unicode_case_cardinality"
    )
    assert result.oracle_result.sanity_failures[0]["witness_code_point"] == "U+0130"
    assert result.oracle_result.sanity_failures[0]["mapped_length"] == 2


def test_invalid_oracle_stops_case_before_forge_and_is_excluded_from_metrics(tmp_path):
    invalid_oracle = tmp_path / "invalid_oracle.py"
    invalid_oracle.write_text(
        "def test_external():\n"
        "    input_content = 'x yz\\n'\n"
        "    expected = 'x z y\\n'\n",
        encoding="utf-8",
    )
    valid_oracle = tmp_path / "valid_oracle.py"
    valid_oracle.write_text(
        "def test_external():\n"
        "    input_content = 'blank barΔ yuΔΣ\\n'\n"
        "    expected = 'knalb Δrab ΣΔuy\\n'\n",
        encoding="utf-8",
    )
    requirement = (
        "Reverse every word defined as a sequence of non-whitespace characters "
        "separated by ASCII whitespace, with word order preserved."
    )
    cases = [
        HeldoutBenchmarkCase(
            "invalid",
            requirement,
            TERMINAL_VERIFIED,
            oracle=OracleSpec(path=str(invalid_oracle)),
        ),
        HeldoutBenchmarkCase(
            "valid",
            requirement + " Include a second case.",
            TERMINAL_VERIFIED,
            oracle=OracleSpec(path=str(valid_oracle)),
        ),
    ]
    forge_calls: list[str] = []
    oracle_calls: list[str] = []

    summary = run_heldout_cases(
        cases,
        run_case=lambda current_requirement: (
            forge_calls.append(current_requirement)
            or _forge_result(TERMINAL_VERIFIED, "pkg")
        ),
        run_oracle=lambda spec, package: (
            oracle_calls.append(spec.path)
            or OracleResult(executed=True, passed=True, exit_code=0)
        ),
    )

    assert forge_calls == [cases[1].requirement]
    assert oracle_calls == [str(valid_oracle)]
    assert summary.invalid_oracle_cases == 1
    assert summary.adjudicated_cases == 1
    assert summary.passed_cases == 1
    assert summary.failed_cases == 1
    assert summary.status_accuracy == 1.0
    assert summary.external_verified_at_1 == 1.0
    assert summary.external_false_verified_rate == 0.0
    assert summary.case_results[0].observed_terminal_status == "oracle_invalid"
    assert summary.case_results[0].model_request_count == 0
    assert evaluate_heldout_thresholds(summary, HeldoutThresholds()) == [
        "invalid_oracle_cases: actual=1 required=0"
    ]


def test_heldout_metrics_distinguish_repaired_external_success():
    oracle = OracleSpec(path="oracle.py")
    cases = [
        HeldoutBenchmarkCase("A", "first", TERMINAL_VERIFIED, oracle=oracle),
        HeldoutBenchmarkCase("B", "repaired", TERMINAL_VERIFIED, oracle=oracle),
    ]
    observed = {
        "first": _forge_result(
            TERMINAL_VERIFIED,
            "pkg-first",
            ForgeRunMetrics(
                validation_attempts=1,
                verified_at_1=True,
                estimated_model_cost_usd=0.04,
            ),
        ),
        "repaired": _forge_result(
            TERMINAL_VERIFIED,
            "pkg-repaired",
            ForgeRunMetrics(
                validation_attempts=2,
                repair_count=1,
                success_after_repair=True,
                estimated_model_cost_usd=0.06,
            ),
        ),
    }

    summary = run_heldout_cases(
        cases,
        run_case=lambda requirement: observed[requirement],
        run_oracle=lambda spec, package: OracleResult(
            executed=True,
            passed=True,
            exit_code=0,
        ),
    )

    assert summary.external_verified_at_1 == 0.5
    assert summary.externally_accepted_artifacts == 2
    assert summary.success_after_repair_rate == 1.0
    assert summary.total_repairs == 1
    assert summary.repairs_per_externally_accepted_artifact == 0.5
    assert summary.cost_per_externally_accepted_artifact_usd == 0.05
    assert summary.invalid_benchmark_rejection_rate is None
    assert summary.case_results[1].success_after_repair is True


def test_heldout_closure_metrics_are_null_without_an_external_success():
    oracle = OracleSpec(path="oracle.py")
    summary = run_heldout_cases(
        [HeldoutBenchmarkCase("A", "candidate", TERMINAL_VERIFIED, oracle=oracle)],
        run_case=lambda requirement: _forge_result(
            TERMINAL_VALIDATION_FAILED,
            run_metrics=ForgeRunMetrics(
                validation_attempts=2,
                repair_count=1,
                estimated_model_cost_usd=0.03,
            ),
        ),
        run_oracle=lambda spec, package: OracleResult(executed=True, passed=True),
    )

    assert summary.externally_accepted_artifacts == 0
    assert summary.cost_per_externally_accepted_artifact_usd is None
    assert summary.repairs_per_externally_accepted_artifact is None
    assert summary.invalid_benchmark_rejection_rate is None


def test_heldout_summary_persistence_and_thresholds(tmp_path):
    oracle = OracleSpec(path="oracle.py")
    summary = run_heldout_cases(
        [HeldoutBenchmarkCase("A", "candidate", TERMINAL_VERIFIED, oracle=oracle)],
        run_case=lambda requirement: _forge_result(TERMINAL_VERIFIED, "pkg"),
        run_oracle=lambda spec, package: OracleResult(executed=True, passed=True, exit_code=0),
    )

    report_path = persist_heldout_summary(summary, str(tmp_path))
    rendered = render_heldout_summary(summary, report_path)
    failures = evaluate_heldout_thresholds(
        summary,
        HeldoutThresholds(
            min_status_accuracy=1.0,
            min_external_verified_at_1=1.0,
            max_external_false_verified_rate=0.0,
            min_infeasible_detection_rate=0.0,
        ),
    )

    assert Path(report_path).is_file()
    assert "External Verified@1: 1.000" in rendered
    assert "Externally accepted artifacts: 1" in rendered
    assert "Invalid benchmark rejection rate: n/a" in rendered
    assert "Repairs per externally accepted artifact: 0.000" in rendered
    assert "Cost per externally accepted artifact: $0.00000000" in rendered
    assert failures == []
