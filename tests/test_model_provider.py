from types import SimpleNamespace

import pytest

from core.model_provider import (
    DEFAULT_OPENAI_MODEL,
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
        ValueError,
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
