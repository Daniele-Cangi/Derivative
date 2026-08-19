import pytest

from core.forge.candidate_preflight import run_fixture_oracle_preflight
from core.forge.contracts import BuildSpec, FeasiblePlan, RequirementAtom
from core.forge.fixture_oracle import (
    derive_fixture_output,
    fixture_oracle_capability,
    fixture_oracle_mismatches,
)


REQUIREMENT = (
    "Reverse every word in place, preserving word order. A word is a sequence of "
    "non-whitespace characters separated by ASCII whitespace: space, tab, carriage "
    "return, form feed, or a preserved line ending."
)


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("blank", "knalb"),
        ("foo barΔ\n", "oof Δrab\n"),
        ("yuΔΣ alpha\r", "ΣΔuy ahpla\r"),
        ("x yz\n", "x zy\n"),
    ],
)
def test_fixture_oracle_derives_exact_unicode_and_token_reversal(fixture, expected):
    assert derive_fixture_output(REQUIREMENT, fixture) == expected


def test_fixture_oracle_rejects_manual_blank_bar_delta_and_yu_delta_sigma_values():
    source = r'''def test_transform():
    content = "blank\nfoo barΔ\nyuΔΣ alpha\r"
    expected = "nkalb\noof rabΔ\nuyΔΣ ahpla\r"
    assert generated(content) == expected
'''

    mismatches = fixture_oracle_mismatches(source, REQUIREMENT)

    assert len(mismatches) == 1
    assert mismatches[0].declared_expected == repr(
        "nkalb\noof rabΔ\nuyΔΣ ahpla\r"
    )
    assert mismatches[0].derived_expected == repr(
        "knalb\noof Δrab\nΣΔuy ahpla\r"
    )


def test_fixture_oracle_detects_frozen_external_yz_contradiction():
    external_oracle_fragment = r'''def test_stdin_to_file():
    input_content = "x yz\n\rΑΒ γδ\n\n"
    expected = "x z y\n\rΒΑ δγ\n\n"
    assert output == expected
'''

    mismatches = fixture_oracle_mismatches(external_oracle_fragment, REQUIREMENT)

    assert len(mismatches) == 1
    assert mismatches[0].declared_expected == repr("x z y\n\rΒΑ δγ\n\n")
    assert mismatches[0].derived_expected == repr("x zy\n\rΒΑ δγ\n\n")


def test_fixture_oracle_accepts_source_independent_correct_expectation():
    source = r'''def test_transform():
    input_content = "blank\nfoo barΔ\nyuΔΣ alpha\r"
    expected = "knalb\noof Δrab\nΣΔuy ahpla\r"
    assert generated(input_content) == expected
'''

    assert fixture_oracle_mismatches(source, REQUIREMENT) == []


def test_fixture_oracle_preflight_fails_before_candidate_execution():
    spec = BuildSpec(
        build_id="build-fixture-oracle",
        raw_requirement=REQUIREMENT,
        normalized_requirement=REQUIREMENT,
        requirement_atoms=[
            RequirementAtom(
                requirement_id="R001",
                text=REQUIREMENT,
                category="functional",
                strength="hard",
                source_fragment=REQUIREMENT,
            )
        ],
    )
    plan = FeasiblePlan(
        plan_id="plan-fixture-oracle",
        build_spec=spec,
        architecture_summary="Python transformation function with deterministic tests.",
    )
    test_path = "tests/test_transform.py"
    files = {
        test_path: r'''def test_transform():
    content = b"foo bar\xce\x94\n"
    expected = b"oof rab\xce\x94\n"
    assert generated(content) == expected
'''
    }

    result = run_fixture_oracle_preflight(files, plan, {test_path: {}})

    assert result["passed"] is False
    assert result["phase"] == "fixture_oracle"
    assert result["failed_paths"] == [test_path]
    assert result["failures"][0]["kind"] == "fixture_oracle_mismatch"
    assert result["failures"][0]["requirement_id"] == "R001"
    assert "source-independent" in result["correction_requirements"][0]


def test_fixture_oracle_is_inactive_without_matching_semantics():
    unrelated = "Merge two JSON objects recursively and replace lists."

    assert fixture_oracle_capability(unrelated) is None
    assert derive_fixture_output(unrelated, "blank") is None
