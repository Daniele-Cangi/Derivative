from core.forge.execution import (
    ExecutionPolicy,
    SandboxProcessResult,
)
from core.forge.repair_support import (
    preflight_failed_paths,
    preflight_has_source_failure,
    pytest_failure_details,
    run_test_preflight,
    source_api_contracts,
)


class _RecordingExecutor:
    def __init__(self):
        self.policy = ExecutionPolicy(backend="local")
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return SandboxProcessResult(
            returncode=0,
            stdout="1 passed\n" if "pytest" in request.command else "",
            stderr="",
            backend="local",
            execution_time_seconds=0.01,
            isolation=self.policy.evidence(),
        )


def test_source_api_contracts_distinguish_click_commands_from_plain_functions():
    contracts = source_api_contracts(
        {
            "src/cli.py": (
                "import click\n\n"
                "def run() -> int:\n"
                "    return 0\n\n"
                "@click.command()\n"
                "@click.argument('input_path')\n"
                "def main(input_path: str) -> None:\n"
                "    raise SystemExit(run())\n"
            )
        }
    )

    functions = {
        function["name"]: function
        for function in contracts["src/cli.py"]["functions"]
    }
    assert functions["run"]["decorators"] == []
    assert functions["main"]["decorators"] == [
        "click.command()",
        "click.argument('input_path')",
    ]


def test_preflight_source_failure_detects_windows_and_relative_source_frames():
    assert preflight_has_source_failure(
        {
            "stdout": "C:\\Temp\\run\\src\\validator.py:28: AttributeError",
            "stderr": "",
        }
    )
    assert preflight_has_source_failure(
        {
            "stdout": "NameError in src/pipeline.py:14",
            "stderr": "",
        }
    )
    assert not preflight_has_source_failure(
        {
            "stdout": "tests/test_pipeline.py:20: AssertionError",
            "stderr": "",
        }
    )


def test_candidate_gate_reports_test_syntax_failure_before_pytest():
    result = run_test_preflight(
        {
            "src/component.py": "def run() -> int:\n    return 0\n",
            "tests/test_component.py": "def test_component(:\n    pass\n",
        },
        ["tests/test_component.py"],
        timeout_seconds=20,
    )

    assert result["phase"] == "syntax"
    assert result["ran"] is False
    assert result["failed_paths"] == ["tests/test_component.py"]
    assert result["source_failed_paths"] == []
    assert result["test_failed_paths"] == ["tests/test_component.py"]


def test_candidate_gate_reports_exact_source_import_failure():
    result = run_test_preflight(
        {
            "src/component.py": "import forge_dependency_that_does_not_exist\n",
            "tests/test_component.py": "def test_component():\n    assert 1 == 1\n",
        },
        ["tests/test_component.py"],
        timeout_seconds=20,
    )

    assert result["phase"] == "import"
    assert result["failed_paths"] == ["src/component.py"]
    assert result["source_failed_paths"] == ["src/component.py"]
    assert preflight_has_source_failure(result) is True


def test_candidate_gate_maps_pytest_failure_to_exact_test_file():
    result = run_test_preflight(
        {
            "src/component.py": "def run() -> int:\n    return 0\n",
            "tests/test_component.py": "def test_component():\n    assert 1 == 2\n",
            "tests/test_other.py": "def test_other():\n    assert 1 == 1\n",
        },
        ["tests/test_component.py", "tests/test_other.py"],
        timeout_seconds=20,
    )

    assert result["phase"] == "tests"
    assert result["ran"] is True
    assert result["passed"] is False
    assert result["test_failed_paths"] == ["tests/test_component.py"]
    assert preflight_failed_paths(result, prefix="tests/") == [
        "tests/test_component.py"
    ]
    assert pytest_failure_details(result) == [
        {
            "path": "tests/test_component.py",
            "node_id": "tests/test_component.py::test_component",
            "message": "assert 1 == 2",
        }
    ]


def test_candidate_gate_exposes_src_layout_to_nested_python_processes():
    result = run_test_preflight(
        {
            "src/tool.py": (
                "def main() -> int:\n"
                "    print('ready')\n"
                "    return 0\n\n"
                "if __name__ == '__main__':\n"
                "    raise SystemExit(main())\n"
            ),
            "tests/test_tool.py": (
                "import subprocess\n"
                "import sys\n\n"
                "def test_module_entrypoint():\n"
                "    completed = subprocess.run([sys.executable, '-m', 'tool'], capture_output=True, text=True)\n"
                "    assert completed.returncode == 0, completed.stderr\n"
                "    assert completed.stdout.strip() == 'ready'\n"
            ),
        },
        ["tests/test_tool.py"],
        timeout_seconds=20,
    )

    assert result["passed"] is True


def test_candidate_gate_uses_one_injected_executor_for_imports_and_tests():
    executor = _RecordingExecutor()

    result = run_test_preflight(
        {
            "src/component.py": "def run() -> int:\n    return 0\n",
            "tests/test_component.py": "from src.component import run\n\ndef test_run():\n    assert run() == 0\n",
        },
        ["tests/test_component.py"],
        timeout_seconds=20,
        executor=executor,
    )

    assert result["passed"] is True
    assert result["execution_policy"]["backend"] == "local"
    assert len(executor.requests) == 2
    assert executor.requests[0].command[:3] == ["python", "-B", "-c"]
    assert "pytest" in executor.requests[1].command
    assert executor.requests[1].environment["PYTHONPATH"] == "src"
    assert all(request.workspace == executor.requests[0].workspace for request in executor.requests)
