import pytest

from core.forge.semantic_contracts import non_semantic_test_paths
from core.forge.test_evidence import non_semantic_test_reasons


@pytest.mark.parametrize(
    "assertion",
    [
        "result == 0 or result != 0",
        "result is None or result is not None",
        "result in values or result not in values",
        "result or not result",
        "True",
        "1",
        "'always true'",
    ],
)
def test_shared_anti_stub_contract_rejects_tautologies(assertion):
    path = "tests/test_contract.py"
    content = (
        "def test_contract():\n"
        "    target = lambda: 0\n"
        "    result = target()\n"
        "    values = {0}\n"
        f"    assert {assertion}\n"
    )

    assert non_semantic_test_paths([path], {path: content}) == [path]


def test_shared_anti_stub_contract_accepts_observable_equality():
    path = "tests/test_contract.py"
    content = (
        "def test_contract():\n"
        "    target = lambda: 0\n"
        "    result = target()\n"
        "    assert result == 0\n"
    )

    assert non_semantic_test_paths([path], {path: content}) == []


def test_shared_anti_stub_contract_rejects_assertion_disconnected_from_target():
    path = "tests/test_contract.py"
    content = (
        "import cli\n"
        "\n"
        "def test_contract():\n"
        "    cli.main([])\n"
        "    unrelated = 2\n"
        "    assert unrelated == 2\n"
    )

    reasons = non_semantic_test_reasons(
        [path],
        {path: content},
        target_names={"main"},
        target_modules={"cli"},
    )

    assert reasons == {path: ["disconnected_assertion"]}


def test_shared_anti_stub_contract_accepts_target_return_value_assertion():
    path = "tests/test_contract.py"
    content = (
        "import cli\n"
        "\n"
        "def test_contract():\n"
        "    result = cli.main([])\n"
        "    assert result == 0\n"
    )

    reasons = non_semantic_test_reasons(
        [path],
        {path: content},
        target_names={"main"},
        target_modules={"cli"},
    )

    assert reasons == {}


def test_shared_anti_stub_contract_accepts_post_target_file_observation():
    path = "tests/test_contract.py"
    content = (
        "import cli\n"
        "\n"
        "def test_contract(output_path):\n"
        "    cli.main([str(output_path)])\n"
        "    output = output_path.read_text(encoding='utf-8')\n"
        "    assert output == 'expected'\n"
    )

    reasons = non_semantic_test_reasons(
        [path],
        {path: content},
        target_names={"main"},
        target_modules={"cli"},
    )

    assert reasons == {}
