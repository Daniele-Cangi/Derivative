from types import SimpleNamespace

from core.forge.cli_contract import cli_invocation_contract_failures
from core.forge.candidate_preflight import run_semantic_preflight
from core.forge.contracts import PlanInterface


def test_cli_contract_rejects_full_sys_argv_and_synthetic_program_argument():
    plan = SimpleNamespace(
        interfaces=[
            PlanInterface(
                name="main",
                interface_type="cli_entrypoint",
                module_path="reverse_chunks",
                explicit_argv_excludes_program_name=True,
                explicit_argv_count=2,
            )
        ],
        implementation_blueprint=SimpleNamespace(
            entrypoint_path="src/reverse_chunks.py"
        ),
    )
    files = {
        "src/reverse_chunks.py": (
            "import sys\n"
            "def main(argv=None):\n"
            "    if argv is None:\n"
            "        argv = sys.argv\n"
            "    if len(argv) != 3:\n"
            "        return 1\n"
            "    _prog, filename, size = argv\n"
            "    return 0\n"
        ),
        "tests/test_reverse_chunks.py": (
            "from reverse_chunks import main\n"
            "def test_valid_input():\n"
            "    args = ['prog', 'input.txt', '2']\n"
            "    assert main(args) == 0\n"
        ),
    }

    failures = cli_invocation_contract_failures(files, plan)

    reasons = {failure["reason"] for failure in failures}
    paths = {failure["path"] for failure in failures}
    assert reasons == {
        "explicit_argv_arity_mismatch",
        "explicit_argv_includes_program_name",
        "explicit_argv_uses_full_sys_argv",
    }
    assert paths == {
        "src/reverse_chunks.py",
        "tests/test_reverse_chunks.py",
    }


def test_cli_contract_accepts_user_only_argv_and_default_sys_argv_slice():
    plan = SimpleNamespace(
        interfaces=[
            PlanInterface(
                name="main",
                interface_type="cli_entrypoint",
                module_path="reverse_chunks",
                explicit_argv_excludes_program_name=True,
                explicit_argv_count=2,
            )
        ],
        implementation_blueprint=SimpleNamespace(
            entrypoint_path="src/reverse_chunks.py"
        ),
    )
    files = {
        "src/reverse_chunks.py": (
            "import sys\n"
            "def main(argv=None):\n"
            "    if argv is None:\n"
            "        argv = sys.argv[1:]\n"
            "    if len(argv) != 2:\n"
            "        return 1\n"
            "    filename, size = argv\n"
            "    return 0\n"
        ),
        "tests/test_reverse_chunks.py": (
            "from reverse_chunks import main\n"
            "def test_valid_input():\n"
            "    assert main(['input.txt', '2']) == 0\n"
        ),
    }

    assert cli_invocation_contract_failures(files, plan) == []


def test_cli_contract_failure_enters_semantic_preflight():
    plan = SimpleNamespace(
        build_spec=SimpleNamespace(
            normalized_requirement="Define a CLI reading argv[1] and argv[2].",
            requirement_atoms=[],
        ),
        required_tests=[],
        interfaces=[
            PlanInterface(
                name="main",
                interface_type="cli_entrypoint",
                module_path="reverse_chunks",
                explicit_argv_excludes_program_name=True,
                explicit_argv_count=2,
            )
        ],
        requirement_coverage={},
        implementation_blueprint=SimpleNamespace(
            entrypoint_path="src/reverse_chunks.py"
        ),
    )
    files = {
        "src/reverse_chunks.py": (
            "import sys\n"
            "def main(argv=None):\n"
            "    argv = sys.argv if argv is None else argv\n"
            "    if len(argv) != 3:\n"
            "        return 1\n"
            "    return 0\n"
        )
    }

    result = run_semantic_preflight(
        files,
        plan,
        {},
        {"ran": True, "passed": True, "phase": "tests", "failures": []},
    )

    assert result["passed"] is False
    assert result["phase"] == "semantic_contract"
    assert any(
        failure["kind"] == "cli_invocation_contract_failure"
        for failure in result["failures"]
    )
    assert any(
        "without an executable/program name" in item
        for item in result["correction_requirements"]
    )
