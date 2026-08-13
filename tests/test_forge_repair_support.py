from core.forge.repair_support import preflight_has_source_failure, source_api_contracts


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
