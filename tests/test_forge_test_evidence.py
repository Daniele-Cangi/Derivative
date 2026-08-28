import pytest

from core.forge.semantic_contracts import non_semantic_test_paths
from core.forge.requirement_evidence import requirement_assertion_evidence
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


def test_shared_evidence_accepts_local_wrapper_that_invokes_target():
    path = "tests/test_word_freq_stats.py"
    content = (
        "import word_freq_stats as cli\n"
        "\n"
        "def run_cli(argv):\n"
        "    return cli.main(argv)\n"
        "\n"
        "def test_contract(tmp_path):\n"
        "    source = tmp_path / 'input.txt'\n"
        "    source.write_text('Alpha alpha', encoding='utf-8')\n"
        "    result = run_cli([str(source)])\n"
        "    assert result == 0\n"
    )

    reasons = non_semantic_test_reasons(
        [path],
        {path: content},
        target_names={"main"},
        target_modules={"word_freq_stats"},
    )
    report = requirement_assertion_evidence(
        {"R001": ["cli_entrypoint", "word_freq_stats"]},
        {"R001": [path]},
        {path: content},
        target_names={"main"},
        target_modules={"word_freq_stats"},
    )

    assert reasons == {}
    assert report["R001"]["passed"] is True


def test_shared_evidence_rejects_wrapper_with_only_unused_nested_target_call():
    path = "tests/test_word_freq_stats.py"
    content = (
        "import word_freq_stats as cli\n"
        "\n"
        "def run_cli(argv):\n"
        "    def unused():\n"
        "        return cli.main(argv)\n"
        "    return 0\n"
        "\n"
        "def test_contract():\n"
        "    result = run_cli([])\n"
        "    assert result == 0\n"
    )

    reasons = non_semantic_test_reasons(
        [path],
        {path: content},
        target_names={"main"},
        target_modules={"word_freq_stats"},
    )

    assert reasons == {path: ["missing_target_invocation"]}


def test_shared_evidence_accepts_literal_getattr_target_alias():
    path = "tests/test_public_api.py"
    content = (
        "import pyutfinvert\n"
        "\n"
        "def test_public_interface_only():\n"
        "    func = getattr(pyutfinvert, 'invert_case_preserve_nonletters', None)\n"
        "    result = func('AbC 123!')\n"
        "    assert result == 'aBc 123!'\n"
    )

    reasons = non_semantic_test_reasons(
        [path],
        {path: content},
        target_names={"invert_case_preserve_nonletters"},
        target_modules={"pyutfinvert"},
    )
    report = requirement_assertion_evidence(
        {"R012": ["invert_case_preserve_nonletters"]},
        {"R012": [path]},
        {path: content},
        target_names={"invert_case_preserve_nonletters"},
        target_modules={"pyutfinvert"},
    )

    assert reasons == {}
    assert report["R012"]["passed"] is True
    assert report["R012"]["assertions"][0]["expression"] == (
        "result == 'aBc 123!'"
    )


def test_shared_evidence_rejects_dynamic_getattr_alias():
    path = "tests/test_dynamic_target.py"
    content = (
        "import pyutfinvert\n"
        "\n"
        "def test_dynamic_target(target_name):\n"
        "    func = getattr(pyutfinvert, target_name, None)\n"
        "    result = func('AbC')\n"
        "    assert result == 'aBc'\n"
    )

    reasons = non_semantic_test_reasons(
        [path],
        {path: content},
        target_names={"invert_case_preserve_nonletters"},
        target_modules={"pyutfinvert"},
    )

    assert reasons == {path: ["missing_target_invocation"]}


def test_shared_evidence_accepts_static_target_capability_assertions():
    path = "tests/test_forbidden_capabilities.py"
    content = (
        "import pyenvlines\n"
        "\n"
        "def test_forbidden_capabilities():\n"
        "    names = getattr(pyenvlines.main, '__code__').co_names\n"
        "    assert 'socket' not in names\n"
        "    assert 'subprocess' not in names\n"
    )

    reasons = non_semantic_test_reasons(
        [path],
        {path: content},
        target_names={"main"},
        target_modules={"pyenvlines"},
    )
    report = requirement_assertion_evidence(
        {"R015": []},
        {"R015": [path]},
        {path: content},
        target_names={"main"},
        target_modules={"pyenvlines"},
    )

    assert reasons == {}
    assert report["R015"]["passed"] is True
    assert {
        assertion["kind"]
        for assertion in report["R015"]["assertions"]
    } == {"static_contract_assertion"}


def test_shared_evidence_rejects_vacuous_static_target_assertion():
    path = "tests/test_vacuous_static_contract.py"
    content = (
        "import pyenvlines\n"
        "\n"
        "def test_vacuous_static_contract():\n"
        "    names = getattr(pyenvlines.main, '__code__').co_names\n"
        "    assert len(names) >= 0\n"
    )

    reasons = non_semantic_test_reasons(
        [path],
        {path: content},
        target_names={"main"},
        target_modules={"pyenvlines"},
    )

    assert reasons == {path: ["missing_target_invocation"]}


def test_requirement_evidence_rejects_term_and_assertion_in_different_functions():
    path = "tests/test_contract.py"
    content = (
        "import cli\n"
        "\n"
        "def test_invalid_date_fixture_only():\n"
        "    invalid_date = 'not-a-date'\n"
        "    assert invalid_date == 'not-a-date'\n"
        "\n"
        "def test_exit_code_only():\n"
        "    result = cli.main([])\n"
        "    assert result == 0\n"
    )

    report = requirement_assertion_evidence(
        {"R001": ["invalid_dates"]},
        {"R001": [path]},
        {path: content},
        target_names={"main"},
        target_modules={"cli"},
    )

    assert report["R001"]["passed"] is False
    assert report["R001"]["missing_terms"] == ["invalid_dates"]
    assert report["R001"]["failure_reason"] == "missing_requirement_assertion_evidence"


def test_requirement_evidence_records_causal_assertion_location():
    path = "tests/test_contract.py"
    content = (
        "import cli\n"
        "\n"
        "def test_invalid_dates_are_rejected():\n"
        "    parsed = cli.parse_date('invalid-date')\n"
        "    assert parsed is None\n"
    )

    report = requirement_assertion_evidence(
        {"R001": ["invalid_dates"]},
        {"R001": [path]},
        {path: content},
        target_names={"parse_date"},
        target_modules={"cli"},
    )

    assert report["R001"]["passed"] is True
    assert report["R001"]["covered_terms"] == ["invalid_dates"]
    assert report["R001"]["assertions"] == [
        {
            "path": path,
            "function": "test_invalid_dates_are_rejected",
            "line": 5,
            "kind": "assert",
            "expression": "parsed is None",
            "evidence_terms": ["invalid_dates"],
        }
    ]


def test_requirement_evidence_keeps_assertions_separate_per_requirement():
    path = "tests/test_summary.py"
    content = (
        "import summary\n"
        "\n"
        "def test_totals():\n"
        "    totals = summary.compute([1, 2])\n"
        "    assert totals['total'] == 3\n"
        "\n"
        "def test_counts():\n"
        "    counts = summary.compute([1, 2])\n"
        "    assert counts['count'] == 2\n"
    )

    report = requirement_assertion_evidence(
        {"R001": ["totals"], "R002": ["counts"]},
        {"R001": [path], "R002": [path]},
        {path: content},
        target_names={"compute"},
        target_modules={"summary"},
    )

    assert report["R001"]["passed"] is True
    assert {item["function"] for item in report["R001"]["assertions"]} == {
        "test_totals"
    }
    assert report["R002"]["passed"] is True
    assert {item["function"] for item in report["R002"]["assertions"]} == {
        "test_counts"
    }
