from types import SimpleNamespace

import pytest

from core.forge.telemetry import track_model_usage
from core.model_provider import (
    DEFAULT_OPENAI_MODEL,
    MissingTextOutputError,
    create_openai_client,
    generate_text,
    is_live_openai_key,
    resolve_openai_api_key,
    resolve_openai_model,
)


class _Responses:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.response


class _FailingResponses:
    def create(self, **kwargs):
        raise RuntimeError("provider unavailable")


def test_openai_client_creation_fails_explicitly_when_runtime_is_unavailable(monkeypatch):
    original_import = __import__

    def reject_openai(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ImportError("openai unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", reject_openai)

    with pytest.raises(RuntimeError, match="OpenAI runtime is required"):
        create_openai_client("sk-test")


def test_generate_text_uses_openai_responses_api():
    responses = _Responses(SimpleNamespace(output_text="result"))
    client = SimpleNamespace(responses=responses)

    text = generate_text(
        client,
        instructions="Return text.",
        input_text="input",
        max_output_tokens=100,
        model="gpt-4.1-mini",
    )

    assert text == "result"
    assert responses.request == {
        "model": "gpt-4.1-mini",
        "instructions": "Return text.",
        "input": "input",
        "max_output_tokens": 100,
    }


def test_generate_text_falls_back_to_structured_output_content():
    response = SimpleNamespace(
        output_text="",
        output=[
            SimpleNamespace(
                content=[SimpleNamespace(text="part one"), SimpleNamespace(text="part two")]
            )
        ],
    )
    client = SimpleNamespace(responses=_Responses(response))

    text = generate_text(
        client,
        instructions="Return text.",
        input_text="input",
        max_output_tokens=100,
        model="gpt-4.1-mini",
    )

    assert text == "part one\npart two"


def test_generate_text_passes_strict_json_schema_to_responses_api():
    responses = _Responses(SimpleNamespace(output_text='{"status":"candidate"}'))
    client = SimpleNamespace(responses=responses)
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
        "additionalProperties": False,
    }

    result = generate_text(
        client,
        instructions="Return JSON.",
        input_text="input",
        max_output_tokens=100,
        model="gpt-4.1-mini",
        output_schema=schema,
        output_schema_name="test_schema",
    )

    assert result == '{"status":"candidate"}'
    assert responses.request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "test_schema",
            "schema": schema,
            "strict": True,
        }
    }


def test_generate_text_reports_incomplete_response_without_payload_content():
    response = SimpleNamespace(
        output_text="",
        output=[],
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    client = SimpleNamespace(responses=_Responses(response))

    with pytest.raises(
        MissingTextOutputError,
        match=r"status=incomplete, reason=max_output_tokens",
    ):
        generate_text(
            client,
            instructions="Return JSON.",
            input_text="sensitive payload must not appear",
            max_output_tokens=100,
            model="gpt-4.1-mini",
        )


def test_openai_configuration_is_deterministic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert resolve_openai_api_key() == ""
    assert resolve_openai_model() == DEFAULT_OPENAI_MODEL
    assert is_live_openai_key("") is False
    assert is_live_openai_key("dummy_key_for_testing") is False
    assert is_live_openai_key("sk-live") is True

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")
    assert resolve_openai_api_key() == "sk-env"
    assert resolve_openai_model() == "gpt-4.1"


def test_generate_text_records_response_usage_without_inventing_cost(monkeypatch):
    monkeypatch.delenv("OPENAI_INPUT_COST_PER_1M_TOKENS", raising=False)
    monkeypatch.delenv("OPENAI_OUTPUT_COST_PER_1M_TOKENS", raising=False)
    response = SimpleNamespace(
        output_text="result",
        usage=SimpleNamespace(input_tokens=125, output_tokens=25, total_tokens=150),
    )
    client = SimpleNamespace(responses=_Responses(response))

    with track_model_usage() as usage:
        generate_text(
            client,
            instructions="Return text.",
            input_text="input",
            max_output_tokens=100,
            model="gpt-test",
        )

    assert usage.request_count == 1
    assert usage.input_tokens == 125
    assert usage.output_tokens == 25
    assert usage.total_tokens == 150
    assert usage.models == {"gpt-test": 1}
    assert usage.estimated_cost() == (None, "unconfigured")


def test_model_usage_cost_requires_explicit_rates(monkeypatch):
    monkeypatch.setenv("OPENAI_INPUT_COST_PER_1M_TOKENS", "2.0")
    monkeypatch.setenv("OPENAI_OUTPUT_COST_PER_1M_TOKENS", "8.0")
    response = SimpleNamespace(
        output_text="result",
        usage=SimpleNamespace(
            input_tokens=1_000_000,
            output_tokens=500_000,
            total_tokens=1_500_000,
        ),
    )
    client = SimpleNamespace(responses=_Responses(response))

    with track_model_usage() as usage:
        generate_text(
            client,
            instructions="Return text.",
            input_text="input",
            max_output_tokens=100,
            model="gpt-test",
        )

    assert usage.estimated_cost() == (6.0, "environment")


def test_model_usage_counts_failed_provider_requests():
    client = SimpleNamespace(responses=_FailingResponses())

    with track_model_usage() as usage:
        with pytest.raises(RuntimeError, match="provider unavailable"):
            generate_text(
                client,
                instructions="Return text.",
                input_text="input",
                max_output_tokens=100,
                model="gpt-test",
            )

    assert usage.request_count == 1
    assert usage.total_tokens == 0
    assert usage.models == {"gpt-test": 1}
