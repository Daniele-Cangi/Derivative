from core.forge.repair_support import (
    preflight_failed_paths,
    preflight_has_source_failure,
    run_test_preflight,
    source_api_contracts,
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
