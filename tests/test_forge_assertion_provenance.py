"""Causal evidence is about the observed binding, not labels or stale names."""

import pytest

from core.forge.candidate_preflight import run_semantic_preflight
from core.forge.contracts import (
    CodeArtifact,
    FeasiblePlan,
    GeneratedFile,
    PlanInterface,
    PlanTest,
)
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.requirement_evidence import requirement_assertion_evidence
from core.forge.semantic_contracts import behaviorally_evidences, semantic_term_present
from core.forge.test_evidence import analyze_test_functions, non_semantic_test_reasons
from core.forge.validation.adversarial import AdversarialValidationLayer
from core.forge.validation.obligations import ObligationValidationLayer


@pytest.mark.parametrize(
    "body,semantic",
    [
        (
            "result = cli.parse_date('2026-01-15')\nresult = None\nassert result is None",
            False,
        ),
        (
            "result = cli.parse_date('2026-01-15')\nresult: object = None\nassert result is None",
            False,
        ),
        (
            "result = cli.parse_date('2026-01-15')\nresult, other = None, 0\nassert result is None",
            False,
        ),
        (
            "result = cli.parse_date('2026-01-15')\nresult = None\nalias = result\nassert alias is None",
            False,
        ),
        (
            "result = None\nassert result is None\nresult = cli.parse_date('2026-01-15')",
            False,
        ),
        (
            "result = cli.parse_date('2026-01-15'); result = None; assert result is None",
            False,
        ),
        (
            "result = cli.parse_date('2026-01-15')\nassert result.year == 2026\nresult = None",
            True,
        ),
        (
            "result = cli.parse_date('2026-01-15')\nalias = result\nresult = None\nassert alias.year == 2026",
            True,
        ),
        (
            "result = cli.parse_date('2026-01-15')\nresult = None\nresult = cli.parse_date('2026-01-15')\nassert result.year == 2026",
            True,
        ),
        (
            "result = cli.parse_date('2026-01-15')\nyear = result.year\nresult = None\nassert year == 2026",
            True,
        ),
        (
            "result = cli.parse_date('2026-01-15'); assert result.year == 2026; result = None",
            True,
        ),
        (
            "input_value = []\ncli.parse_date(input_value)\ninput_value = []\nassert input_value == []",
            False,
        ),
        (
            "input_value = []\ncli.parse_date(input_value)\ninput_value = []\ncli.parse_date(input_value)\nassert input_value == []",
            True,
        ),
    ],
)
def test_return_provenance_is_evaluated_at_each_assertion(body, semantic):
    content = (
        "import cli\n\ndef test_contract():\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    function = analyze_test_functions(content, {"parse_date"}, {"cli"})[0]
    assert function["semantic"] is semantic
    assert bool(function["assertions"]) is semantic
    reasons = non_semantic_test_reasons(
        ["tests/test_contract.py"],
        {"tests/test_contract.py": content},
        {"parse_date"},
        {"cli"},
    )
    assert reasons == (
        {} if semantic else {"tests/test_contract.py": ["disconnected_assertion"]}
    )


def test_later_rebinding_does_not_rewrite_earlier_assertion_evidence():
    content = (
        "import cli\n\ndef test_contract():\n"
        "    result = cli.parse_date('2026-01-15')\n"
        "    assert result.year == 2026\n"
        "    result = None\n"
        "    assert result is None\n"
    )
    function = analyze_test_functions(content, {"parse_date"}, {"cli"})[0]
    assert function["assertions"] == [
        {"line": 5, "kind": "assert", "expression": "result.year == 2026"}
    ]


@pytest.mark.parametrize(
    "definition",
    [
        "    def unrelated_helper():\n        result = None\n        return result\n",
        "    async def unrelated_helper():\n        result = None\n        return result\n",
        "    class Unrelated:\n        result = None\n",
        "    unrelated_helper = lambda: (result := None)\n",
    ],
)
def test_nested_local_bindings_do_not_reassign_the_observed_outer_result(definition):
    content = (
        "import cli\n\ndef test_contract():\n"
        "    result = cli.parse_date('2026-01-15')\n"
        + definition
        + "    assert result.year == 2026\n"
    )
    function = analyze_test_functions(content, {"parse_date"}, {"cli"})[0]
    assert function["semantic"] is True


def test_a_deferred_call_does_not_supply_outer_test_provenance():
    content = (
        "import cli\n\ndef test_contract():\n"
        "    result = None\n"
        "    helper = lambda: cli.parse_date('2026-01-15')\n"
        "    assert result is None\n"
    )
    function = analyze_test_functions(content, {"parse_date"}, {"cli"})[0]
    assert function["semantic"] is False
    assert function["target_invoked"] is False


@pytest.mark.parametrize(
    "metadata",
    [
        "    # quarantine\n",
        '    """quarantine"""\n',
        "    'quarantine'\n",
    ],
)
@pytest.mark.parametrize("production_matcher", [False, True])
def test_documentation_cannot_supply_requirement_assertion_terms(
    metadata, production_matcher
):
    content = (
        "from telemetry_parser import parse_jsonl\n\ndef test_parser():\n"
        + metadata
        + "    result = parse_jsonl([])\n    assert result == []\n"
    )
    matcher = (
        (
            lambda term, text: semantic_term_present(term, text, is_test=True)
            or behaviorally_evidences(term, text, {"parse_jsonl"})
        )
        if production_matcher
        else None
    )
    report = requirement_assertion_evidence(
        {"R001": ["quarantine"]},
        {"R001": ["tests/test_parser.py"]},
        {"tests/test_parser.py": content},
        {"parse_jsonl"},
        {"telemetry_parser"},
        matcher,
    )["R001"]
    assert report["passed"] is False
    assert report["missing_terms"] == ["quarantine"]


def test_literal_values_in_actual_assertions_and_original_locations_are_preserved():
    content = (
        "from telemetry_parser import parse_jsonl\n\ndef test_parser():\n"
        '    """Documentation is not evidence."""\n'
        "    result = parse_jsonl(['invalid'])\n"
        "    assert result == {'quarantine': ['invalid']}\n"
    )
    report = requirement_assertion_evidence(
        {"R001": ["quarantine"]},
        {"R001": ["tests/test_parser.py"]},
        {"tests/test_parser.py": content},
        {"parse_jsonl"},
        {"telemetry_parser"},
    )["R001"]
    assert report["passed"] is True
    assert report["assertions"][0]["line"] == 6
    assert (
        report["assertions"][0]["expression"] == "result == {'quarantine': ['invalid']}"
    )


@pytest.mark.parametrize("case", ["empty", "comment", "docstring", "malformed"])
def test_shared_preflight_and_validator_gates_reject_documentation_only_coverage(
    case, tmp_path
):
    spec = RequirementCompiler().compile("Quarantine malformed records.")
    atom = spec.requirement_atoms[0]
    path = "tests/test_parser.py"
    plan = FeasiblePlan(
        plan_id="assertion-provenance",
        build_spec=spec,
        architecture_summary="Parser coverage regression.",
        interfaces=[PlanInterface(name="parse_jsonl", interface_type="function")],
        required_tests=[
            PlanTest(
                test_name="test_parser",
                objective=atom.text,
                requirement_ids=[atom.requirement_id],
            )
        ],
        requirement_coverage={
            atom.requirement_id: {
                "files": ["src/telemetry_parser.py"],
                "tests": ["test_parser"],
            }
        },
    )
    metadata = {
        "comment": "    # quarantine malformed_records\n",
        "docstring": '    """quarantine malformed_records"""\n',
    }.get(case, "")
    body = (
        "    malformed_records = ['invalid']\n    accepted, quarantine = parse_jsonl(malformed_records)\n    assert quarantine == malformed_records\n"
        if case == "malformed"
        else "    result = parse_jsonl([])\n    assert result == ([], [])\n"
    )
    files = {
        "src/telemetry_parser.py": "import json\ndef parse_jsonl(lines):\n    accepted, quarantine = [], []\n    for line in lines:\n        try:\n            accepted.append(json.loads(line))\n        except json.JSONDecodeError:\n            quarantine.append(line)\n    return accepted, quarantine\n",
        path: "from telemetry_parser import parse_jsonl\n\ndef test_parser():\n"
        + metadata
        + body,
    }
    artifact = CodeArtifact(
        artifact_id=case,
        plan_id=plan.plan_id,
        files=[
            GeneratedFile(
                name, text, "test" if name.startswith("tests/") else "python_module"
            )
            for name, text in files.items()
        ],
        test_paths=[path],
    )
    preflight = run_semantic_preflight(
        files,
        plan,
        {
            path: {
                "requirements": [
                    {"id": atom.requirement_id, "evidence_terms": atom.evidence_terms}
                ]
            }
        },
        {"ran": True, "passed": True, "failures": []},
    )
    _, _, evidence2 = ObligationValidationLayer.__new__(
        ObligationValidationLayer
    )._validate_requirement_semantics(spec, plan, artifact)
    target = tmp_path / "test_parser.py"
    target.write_text(files[path], encoding="utf-8")
    _, _, evidence3 = (
        AdversarialValidationLayer()._validate_semantic_requirement_test_coverage(
            spec, plan, {path}, [], {path: target}
        )
    )
    expected = case == "malformed"
    assert preflight["passed"] is expected
    assert (
        evidence2["requirements"][atom.requirement_id]["assertion_evidence"]["passed"]
        is expected
    )
    assert (
        evidence3["requirements"][atom.requirement_id]["assertion_evidence"]["passed"]
        is expected
    )
