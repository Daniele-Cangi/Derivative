import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import forge_blind_produce
from core.forge.blind_oracle import (
    oracle_preflight_error,
    oracle_preflight_failure_class,
)
from core.forge.blind_requirement import requirement_preflight_error
from core.forge.blind_producer import (
    BlindProducerConfig,
    _generate_oracle,
    _generate_requirement_case,
    produce_and_freeze_blind_bundle,
)
from core.model_provider import MissingTextOutputError


def _case_payload() -> dict:
    return {
        "cases": [
            {
                "requirement": (
                    "Build a Python library module code_policy exposing classify_code(code: str) -> str. "
                    "Return 'internal' for codes beginning INT-, return 'external' for EXT-, reject empty "
                    "or unknown codes with ValueError, preserve deterministic behavior, and include tests. "
                    "Public import contract: from code_policy import classify_code."
                ),
                "expected_terminal_status": "verified",
                "public_contract": {
                    "module": "code_policy",
                    "symbol": "classify_code",
                    "kind": "function",
                },
                "tags": ["library", "classification"],
            },
            {
                "requirement": (
                    "Build a production tool that identifies every meaningful risk in arbitrary customer "
                    "documents with perfect accuracy, without defining risk, document formats, or acceptance "
                    "criteria, and include exhaustive proof that every result is correct. "
                    "Public import contract: from risk_tool import analyze_risk."
                ),
                "expected_terminal_status": "validation_failed",
                "public_contract": {
                    "module": "risk_tool",
                    "symbol": "analyze_risk",
                    "kind": "function",
                },
                "tags": ["ambiguity", "universal"],
            },
            {
                "requirement": (
                    "Build a reversible encoder mapping every possible 24-bit input to one 8-bit output, "
                    "recover every original input exactly, and use no metadata, external state, rejection, "
                    "randomness, or additional storage under any circumstances. "
                    "Public import contract: from impossible_encoder import encode."
                ),
                "expected_terminal_status": "infeasible_proven",
                "public_contract": {
                    "module": "impossible_encoder",
                    "symbol": "encode",
                    "kind": "function",
                },
                "tags": ["contradiction", "information"],
            },
        ]
    }


def _oracle_payload() -> dict:
    return {
        "oracle_py": (
            "import pytest\n"
            "from code_policy import classify_code\n\n"
            "def test_internal_code():\n"
            "    assert classify_code('INT-42') == 'internal'\n\n"
            "def test_external_code():\n"
            "    assert classify_code('EXT-9') == 'external'\n\n"
            "def test_unknown_code_rejected():\n"
            "    with pytest.raises(ValueError):\n"
            "        classify_code('OTHER')\n"
        )
    }


class _RecordingGenerator:
    def __init__(
        self,
        oracle_payload: dict | None = None,
        review_payloads: list[dict] | None = None,
        requirement_review_payloads: list[dict] | None = None,
    ):
        self.calls = []
        self.oracle_payload = oracle_payload or _oracle_payload()
        self.review_payloads = list(
            review_payloads or [{"approved": True, "findings": []}]
        )
        self.requirement_review_payloads = list(
            requirement_review_payloads or [{"approved": True, "findings": []}]
        )

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["output_schema_name"].endswith("_requirements_review"):
            payload = (
                self.requirement_review_payloads.pop(0)
                if self.requirement_review_payloads
                else {"approved": True, "findings": []}
            )
            return json.dumps(payload)
        if kwargs["output_schema_name"].endswith("_requirements"):
            expected_status = next(
                item["expected_terminal_status"]
                for item in _case_payload()["cases"]
                if f"Required terminal status: {item['expected_terminal_status']}"
                in kwargs["input_text"]
            )
            item = next(
                case
                for case in _case_payload()["cases"]
                if case["expected_terminal_status"] == expected_status
            )
            return json.dumps(
                {
                    "case": {
                        "requirement": item["requirement"],
                        "public_contract": item["public_contract"],
                        "tags": item["tags"],
                    }
                }
            )
        if kwargs["output_schema_name"].endswith("_oracle_review"):
            payload = (
                self.review_payloads.pop(0)
                if self.review_payloads
                else {"approved": True, "findings": []}
            )
            return json.dumps(payload)
        return json.dumps(self.oracle_payload)


def test_one_shot_producer_separates_generation_and_freezes_before_publication(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "blind-v4-external"
    generator = _RecordingGenerator()

    bundle = produce_and_freeze_blind_bundle(
        output_root=output_root,
        repository_root=repository_root,
        config=BlindProducerConfig(
            bundle_id="blind-v4-test",
            benchmark_version="v4",
            verified_cases=1,
            validation_failed_cases=1,
            infeasible_cases=1,
        ),
        text_generator=generator,
        model="external-test-model",
    )

    assert output_root.is_dir()
    assert Path(bundle.manifest_path).is_file()
    assert len(bundle.cases) == 3
    assert b"\r\n" not in (output_root / "cases.json").read_bytes()
    assert b"\r\n" not in (
        output_root / "oracles" / "V4-001" / "oracle.py"
    ).read_bytes()
    assert [call["output_schema_name"] for call in generator.calls] == [
        "forge_blind_v4_requirements",
        "forge_blind_v4_requirements_review",
        "forge_blind_v4_requirements",
        "forge_blind_v4_requirements_review",
        "forge_blind_v4_requirements",
        "forge_blind_v4_requirements_review",
        "forge_blind_v4_oracle",
        "forge_blind_v4_oracle_review",
    ]
    requirement_review_request = generator.calls[1]["input_text"]
    assert _case_payload()["cases"][0]["requirement"] in requirement_review_request
    assert "never include confirming observations as findings" in generator.calls[1][
        "instructions"
    ]
    assert "logically satisfiable in principle" in generator.calls[2][
        "instructions"
    ]
    assert "importable main(argv: list[str] | None = None) -> int" in generator.calls[
        0
    ]["instructions"]
    oracle_request = generator.calls[6]["input_text"]
    assert _case_payload()["cases"][0]["requirement"] in oracle_request
    assert "core/forge" not in oracle_request
    assert "candidate implementation" not in oracle_request.lower()
    assert "lexically inside each test function" in generator.calls[6]["instructions"]
    review_request = generator.calls[7]["input_text"]
    assert _case_payload()["cases"][0]["requirement"] in review_request
    assert _oracle_payload()["oracle_py"] in review_request

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["independent_of_forge"] is True
    assert manifest["provenance"]["sealed_before_first_execution"] is True
    assert manifest["forge_baseline"]["sha256"] == bundle.baseline_sha256
    assert manifest["forge_baseline"]["digest_mode"] == "canonical_lf_v1"
    assert set(manifest["oracle_sha256"]) == {"V4-001"}
    cases = json.loads((output_root / "cases.json").read_text(encoding="utf-8"))
    assert cases[0]["tags"][0] == "blind-v4"
    assert cases[0]["public_contract"] == {
        "kind": "function",
        "module": "code_policy",
        "symbol": "classify_code",
    }
    assert cases[0]["requirement_validation"] == {
        "findings": [],
        "independent_review_passed": True,
        "review_model": "external-test-model",
        "static_checks_passed": True,
    }
    assert cases[0]["oracle_validation"] == {
        "findings": [],
        "independent_review_passed": True,
        "review_model": "external-test-model",
        "static_checks_passed": True,
    }


def test_one_shot_producer_refuses_existing_destination_without_model_call(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    generator = _RecordingGenerator()

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        produce_and_freeze_blind_bundle(
            output_root=destination,
            repository_root=Path(__file__).resolve().parents[1],
            config=BlindProducerConfig(
                bundle_id="blind-v3-test",
                verified_cases=1,
                validation_failed_cases=1,
                infeasible_cases=1,
            ),
            text_generator=generator,
            model="external-test-model",
        )

    assert generator.calls == []


def test_one_shot_producer_rejects_superficial_oracle_and_publishes_nothing(tmp_path):
    destination = tmp_path / "rejected"
    generator = _RecordingGenerator(
        oracle_payload={
            "oracle_py": (
                "def test_one():\n    assert True\n\n"
                "def test_two():\n    assert True\n\n"
                "def test_three():\n    assert True\n"
            )
        }
    )

    with pytest.raises(ValueError, match="Oracle producer failed validation"):
        produce_and_freeze_blind_bundle(
            output_root=destination,
            repository_root=Path(__file__).resolve().parents[1],
            config=BlindProducerConfig(
                bundle_id="blind-v3-test",
                verified_cases=1,
                validation_failed_cases=1,
                infeasible_cases=1,
                max_generation_attempts=2,
            ),
            text_generator=generator,
            model="external-test-model",
        )

    assert destination.exists() is False
    assert [call["output_schema_name"] for call in generator.calls].count(
        "forge_blind_v3_oracle"
    ) == 2


def test_one_shot_producer_regenerates_oracle_rejected_by_independent_review(tmp_path):
    destination = tmp_path / "reviewed"
    generator = _RecordingGenerator(
        review_payloads=[
            {
                "approved": False,
                "findings": ["The expected result is not stated by the requirement."],
            },
            {"approved": True, "findings": []},
        ]
    )

    bundle = produce_and_freeze_blind_bundle(
        output_root=destination,
        repository_root=Path(__file__).resolve().parents[1],
        config=BlindProducerConfig(
            bundle_id="blind-v4-reviewed",
            benchmark_version="v4",
            verified_cases=1,
            validation_failed_cases=1,
            infeasible_cases=1,
        ),
        text_generator=generator,
        model="external-test-model",
    )

    assert len(bundle.cases) == 3
    schemas = [call["output_schema_name"] for call in generator.calls]
    assert schemas.count("forge_blind_v4_oracle") == 2
    assert schemas.count("forge_blind_v4_oracle_review") == 2
    oracle_calls = [
        call
        for call in generator.calls
        if call["output_schema_name"] == "forge_blind_v4_oracle"
    ]
    assert "independent oracle review rejected" in oracle_calls[1]["input_text"]


def test_one_shot_producer_regenerates_case_set_rejected_by_requirement_review(tmp_path):
    destination = tmp_path / "requirement-reviewed"
    generator = _RecordingGenerator(
        requirement_review_payloads=[
            {
                "approved": False,
                "findings": ["Candidate 1 leaves output ordering unspecified."],
            },
            {"approved": True, "findings": []},
        ]
    )

    bundle = produce_and_freeze_blind_bundle(
        output_root=destination,
        repository_root=Path(__file__).resolve().parents[1],
        config=BlindProducerConfig(
            bundle_id="blind-v4-requirement-reviewed",
            benchmark_version="v4",
            verified_cases=1,
            validation_failed_cases=1,
            infeasible_cases=1,
        ),
        text_generator=generator,
        model="external-test-model",
    )

    assert len(bundle.cases) == 3
    schemas = [call["output_schema_name"] for call in generator.calls]
    assert schemas.count("forge_blind_v4_requirements") == 4
    assert schemas.count("forge_blind_v4_requirements_review") == 4
    requirement_calls = [
        call
        for call in generator.calls
        if call["output_schema_name"] == "forge_blind_v4_requirements"
    ]
    assert "independent requirement review rejected" in requirement_calls[1]["input_text"]


def test_requirement_producer_retries_incomplete_generation_and_review_output():
    calls: list[dict] = []
    requirement_attempts = 0
    review_attempts = 0
    source_case = _case_payload()["cases"][0]

    def generator(**kwargs):
        nonlocal requirement_attempts, review_attempts
        calls.append(kwargs)
        if kwargs["output_schema_name"].endswith("_requirements_review"):
            review_attempts += 1
            if review_attempts == 1:
                return "{"
            return json.dumps({"approved": True, "findings": []})
        requirement_attempts += 1
        if requirement_attempts == 1:
            raise MissingTextOutputError(
                "OpenAI response did not contain text output "
                "(status=incomplete, reason=max_output_tokens)."
            )
        return json.dumps(
            {
                "case": {
                    "requirement": source_case["requirement"],
                    "public_contract": source_case["public_contract"],
                    "tags": source_case["tags"],
                }
            }
        )

    candidate = _generate_requirement_case(
        generator=generator,
        model="external-test-model",
        config=BlindProducerConfig(
            bundle_id="blind-v8-retry-test",
            benchmark_version="v8",
            verified_cases=1,
            validation_failed_cases=1,
            infeasible_cases=1,
            max_generation_attempts=3,
        ),
        index=1,
        expected_status="verified",
        accepted_cases=[],
    )

    assert candidate["requirement"] == source_case["requirement"]
    assert requirement_attempts == 3
    assert review_attempts == 2
    assert "incomplete or invalid" in calls[1]["input_text"]
    assert "incomplete structured output" in calls[3]["input_text"]


def test_oracle_producer_retries_incomplete_generation_and_review_output():
    calls: list[dict] = []
    oracle_attempts = 0
    review_attempts = 0

    def generator(**kwargs):
        nonlocal oracle_attempts, review_attempts
        calls.append(kwargs)
        if kwargs["output_schema_name"].endswith("_oracle_review"):
            review_attempts += 1
            if review_attempts == 1:
                raise MissingTextOutputError(
                    "OpenAI response did not contain text output "
                    "(status=incomplete, reason=max_output_tokens)."
                )
            return json.dumps({"approved": True, "findings": []})
        oracle_attempts += 1
        if oracle_attempts == 1:
            return "{"
        return json.dumps(_oracle_payload())

    source, validation = _generate_oracle(
        generator=generator,
        model="external-test-model",
        case_id="V8-001",
        requirement=_case_payload()["cases"][0]["requirement"],
        max_attempts=3,
        schema_namespace="forge_blind_v8",
    )

    assert source == _oracle_payload()["oracle_py"]
    assert validation["independent_review_passed"] is True
    assert oracle_attempts == 3
    assert review_attempts == 2
    assert "incomplete or invalid" in calls[1]["input_text"]
    assert "incomplete structured output" in calls[3]["input_text"]


def test_requirement_producer_does_not_retry_non_structured_generator_errors():
    calls = 0

    def generator(**_kwargs):
        nonlocal calls
        calls += 1
        raise ValueError(
            "OpenAI response did not contain text output "
            "(status=incomplete, reason=max_output_tokens)."
        )

    with pytest.raises(ValueError, match="OpenAI response did not contain text output"):
        _generate_requirement_case(
            generator=generator,
            model="external-test-model",
            config=BlindProducerConfig(
                bundle_id="blind-v8-nonretryable-test",
                benchmark_version="v8",
                verified_cases=1,
                validation_failed_cases=1,
                infeasible_cases=1,
                max_generation_attempts=3,
            ),
            index=1,
            expected_status="verified",
            accepted_cases=[],
        )

    assert calls == 1


def test_oracle_preflight_rejects_discarded_entrypoint_return_value():
    source = (
        "from dupfilter import main\n\n"
        "def test_one(tmp_path):\n"
        "    main()\n"
        "    assert (tmp_path / 'one').exists()\n\n"
        "def test_two(tmp_path):\n"
        "    main()\n"
        "    assert (tmp_path / 'two').exists()\n\n"
        "def test_three(tmp_path):\n"
        "    main()\n"
        "    assert (tmp_path / 'three').exists()\n"
    )

    assert oracle_preflight_error(source) == (
        "oracle discards the return value of public entrypoint main; "
        "assert the returned exit code explicitly"
    )


def test_oracle_preflight_rejects_deterministic_fixture_contradiction():
    requirement = (
        "Reverse every word defined as a sequence of non-whitespace characters separated "
        "by ASCII whitespace, with word order preserved."
    )
    source = _reverse_words_oracle(
        blank="nkalb",
        bar_delta="rabΔ",
        yu_delta_sigma="uyΔΣ",
    )

    error = oracle_preflight_error(source, requirement)

    assert error is not None
    assert "fixture expectation contradicts the requirement" in error
    assert "expected declares 'nkalb'" in error
    assert "independently derived value is 'knalb'" in error
    assert oracle_preflight_failure_class(error) == "fixture_oracle_mismatch"


def test_oracle_preflight_rejects_injected_cli_name_in_main_argv():
    requirement = (
        "Implement a verified CLI utility named 'pycolmask'. "
        "The main(argv: list[str] | None = None) -> int contract must be importable."
    )
    source = (
        "from pycolmask import main\n\n"
        "def test_one():\n"
        "    argv = ['pycolmask', 'one.csv', '--mask=1']\n"
        "    rc = main(argv)\n"
        "    assert rc == 0\n\n"
        "def test_two():\n"
        "    argv = ['pycolmask', 'two.csv', '--mask=0']\n"
        "    rc = main(argv)\n"
        "    assert rc == 0\n\n"
        "def test_three():\n"
        "    argv = ['pycolmask', 'three.csv', '--mask=2']\n"
        "    rc = main(argv)\n"
        "    assert rc == 0\n"
    )

    error = oracle_preflight_error(source, requirement)

    assert error is not None
    assert "invocation contract contradicts the requirement" in error
    assert "passes declared CLI name 'pycolmask' as argv[0]" in error
    assert oracle_preflight_failure_class(error) == "oracle_contract_mismatch"


def test_oracle_preflight_rejects_fixture_that_contradicts_explicit_regex():
    requirement = (
        "Implement a CLI named 'pyenvlines'. A line must match the regex: "
        "'^[A-Z_][A-Z0-9_]*=[^\\n]*$'. The main(argv) contract is importable."
    )
    source = (
        "from pyenvlines import main\n\n"
        "INVALID_LINES = ['FOO=bar extra\\n']\n\n"
        "def test_one():\n"
        "    rc = main([])\n"
        "    assert rc == 1\n\n"
        "def test_two():\n"
        "    rc = main([])\n"
        "    assert rc == 1\n\n"
        "def test_three():\n"
        "    rc = main([])\n"
        "    assert rc == 1\n"
    )

    error = oracle_preflight_error(source, requirement)

    assert error is not None
    assert "explicit pattern contract contradicts" in error
    assert "FOO=bar extra" in error
    assert oracle_preflight_failure_class(error) == "explicit_pattern_mismatch"


def test_requirement_preflight_rejects_verified_unicode_cardinality_conflict():
    requirement = (
        "Implement a function returning a string of the same length as input where "
        "each Unicode letter has its case inverted."
    )

    error = requirement_preflight_error(requirement, "verified")

    assert error is not None
    assert "finite witness contradiction" in error
    assert "U+0130" in error
    assert requirement_preflight_error(requirement, "infeasible_proven") is None


def test_requirement_preflight_rejects_same_length_behavioral_example_conflict():
    requirement = (
        "Return a new list of the same length as inputs after processing adjacent "
        "values. Behavioral tests: ['a','a','b','B','b'] with mode='strict' "
        "returns ['a','b','B','b']."
    )

    error = requirement_preflight_error(requirement, "validation_failed")

    assert error is not None
    assert "behavioral example contradiction" in error
    assert "5-item input returns a 4-item list" in error
    assert requirement_preflight_error(requirement, "infeasible_proven") is None


def test_oracle_preflight_uses_argv_value_preceding_each_call():
    requirement = (
        "Implement a verified CLI utility named 'pycolmask'. "
        "The main(argv: list[str] | None = None) -> int contract must be importable."
    )
    tests = []
    for index in range(3):
        tests.append(
            f"def test_{index}():\n"
            f"    argv = ['input-{index}.csv', '--mask=1']\n"
            "    rc = main(argv)\n"
            "    argv = ['pycolmask', 'unused.csv', '--mask=1']\n"
            "    assert rc == 0\n"
        )
    source = "from pycolmask import main\n\n" + "\n".join(tests)

    assert oracle_preflight_error(source, requirement) is None


def test_oracle_producer_regenerates_before_review_on_fixture_contradiction():
    requirement = (
        "Reverse every word defined as a sequence of non-whitespace characters separated "
        "by ASCII whitespace, with word order preserved."
    )
    sources = [
        _reverse_words_oracle(
            blank="nkalb",
            bar_delta="rabΔ",
            yu_delta_sigma="uyΔΣ",
        ),
        _reverse_words_oracle(
            blank="knalb",
            bar_delta="Δrab",
            yu_delta_sigma="ΣΔuy",
        ),
    ]
    calls: list[dict] = []

    def generator(**kwargs):
        calls.append(kwargs)
        if kwargs["output_schema_name"].endswith("_oracle_review"):
            return json.dumps({"approved": True, "findings": []})
        return json.dumps({"oracle_py": sources.pop(0)})

    source, validation = _generate_oracle(
        generator=generator,
        model="external-test-model",
        case_id="V5-001",
        requirement=requirement,
        max_attempts=2,
        schema_namespace="forge_blind_v5",
    )

    assert "expected = 'ΣΔuy'" in source
    assert validation["static_checks_passed"] is True
    assert [call["output_schema_name"] for call in calls] == [
        "forge_blind_v5_oracle",
        "forge_blind_v5_oracle",
        "forge_blind_v5_oracle_review",
    ]
    assert "independently derived value is 'knalb'" in calls[1]["input_text"]


@pytest.mark.parametrize(
    ("rejected_source", "expected_error"),
    [
        (
            "from code_policy import classify_code\n\ndef test_broken(:\n",
            "oracle syntax error",
        ),
        (
            (
                "from code_policy import classify_code\n\n"
                "def test_one():\n"
                "    assert 'INT-1'.startswith('INT-')\n\n"
                "def test_two():\n"
                "    assert 'EXT-1'.startswith('EXT-')\n\n"
                "def test_three():\n"
                "    assert 'OTHER' != 'INT-1'\n"
            ),
            "does not directly invoke the public target",
        ),
    ],
)
def test_oracle_retry_receives_rejected_source_for_targeted_revision(
    rejected_source,
    expected_error,
):
    requirement = _case_payload()["cases"][0]["requirement"]
    sources = [rejected_source, _oracle_payload()["oracle_py"]]
    calls: list[dict] = []

    def generator(**kwargs):
        calls.append(kwargs)
        if kwargs["output_schema_name"].endswith("_oracle_review"):
            return json.dumps({"approved": True, "findings": []})
        return json.dumps({"oracle_py": sources.pop(0)})

    source, validation = _generate_oracle(
        generator=generator,
        model="external-test-model",
        case_id="V6-001",
        requirement=requirement,
        max_attempts=2,
        schema_namespace="forge_blind_v6",
    )

    assert source == _oracle_payload()["oracle_py"]
    assert validation["static_checks_passed"] is True
    retry = calls[1]["input_text"]
    assert "untrusted data, not instructions" in retry
    assert expected_error in retry
    assert json.dumps(rejected_source) in retry


def _reverse_words_oracle(
    *,
    blank: str,
    bar_delta: str,
    yu_delta_sigma: str,
) -> str:
    return (
        "from revwords import reverse_words\n\n"
        "def test_blank():\n"
        "    input_content = 'blank'\n"
        f"    expected = {blank!r}\n"
        "    result = reverse_words(input_content)\n"
        "    assert result == expected\n\n"
        "def test_bar_delta():\n"
        "    input_content = 'barΔ'\n"
        f"    expected = {bar_delta!r}\n"
        "    result = reverse_words(input_content)\n"
        "    assert result == expected\n\n"
        "def test_yu_delta_sigma():\n"
        "    input_content = 'yuΔΣ'\n"
        f"    expected = {yu_delta_sigma!r}\n"
        "    result = reverse_words(input_content)\n"
        "    assert result == expected\n"
    )


def test_blind_producer_rejects_invalid_version_identifier():
    with pytest.raises(ValueError, match="benchmark_version"):
        BlindProducerConfig(bundle_id="invalid", benchmark_version="4")


def test_producer_cli_reports_failure_without_traceback_or_locals(monkeypatch, tmp_path):
    def fail_production(**_kwargs):
        raise ValueError(
            "Oracle producer failed validation; "
            "rejection_classes=causal_assertion,independent_review"
        )

    monkeypatch.setattr(
        forge_blind_produce,
        "produce_and_freeze_blind_bundle",
        fail_production,
    )
    result = CliRunner().invoke(
        forge_blind_produce.app,
        [str(tmp_path / "bundle"), "--bundle-id", "blind-v4-safe-error"],
    )

    assert result.exit_code == 1
    assert "Blind production failed: oracle_preflight_failed" in result.output
    assert "No bundle was published" in result.output
    assert "Rejection classes: causal_assertion,independent_review" in result.output
    assert "Model requests before failure: 0" in result.output
    assert "Model input tokens before failure: 0" in result.output
    assert "Model output tokens before failure: 0" in result.output
    assert "Model tokens before failure: 0" in result.output
    assert "Estimated producer cost before failure: $0.00000000" in result.output
    assert "Pricing source: no_model_calls" in result.output
    assert "Oracle producer failed validation" not in result.output
    assert "Traceback" not in result.output
    assert "raw_cases" not in result.output


def test_oracle_preflight_failure_classes_do_not_expose_source():
    assert (
        oracle_preflight_failure_class(
            "oracle test test_output has no causal behavioral assertion"
        )
        == "causal_assertion"
    )


def test_frozen_bundle_transport_disables_git_text_normalization():
    repository_root = Path(__file__).resolve().parents[1]
    attributes = (repository_root / ".gitattributes").read_text(encoding="utf-8")

    for version in range(3, 10):
        assert f"benchmarks/blind_v{version}/external_*/** -text" in attributes
