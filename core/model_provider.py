import os
from typing import Any

from openai import OpenAI


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
_NON_LIVE_KEYS = {"", "dummy_key_for_testing", "your-api-key-here"}


def resolve_openai_api_key(api_key: str | None = None) -> str:
    return (api_key or os.getenv("OPENAI_API_KEY") or "").strip()


def resolve_openai_model(model: str | None = None) -> str:
    return (model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip()


def is_live_openai_key(api_key: str) -> bool:
    return api_key not in _NON_LIVE_KEYS


def create_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def generate_text(
    client: Any,
    *,
    instructions: str,
    input_text: str,
    max_output_tokens: int,
    model: str,
    output_schema: dict[str, Any] | None = None,
    output_schema_name: str = "structured_response",
) -> str:
    request: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "max_output_tokens": max_output_tokens,
    }
    if output_schema is not None:
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": output_schema_name,
                "schema": output_schema,
                "strict": True,
            }
        }
    response = client.responses.create(
        **request,
    )
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    fragments: list[str] = []
    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            text = getattr(content_item, "text", None)
            if isinstance(text, str) and text:
                fragments.append(text)
    if fragments:
        return "\n".join(fragments)
    status = str(getattr(response, "status", "unknown") or "unknown")
    incomplete_details = getattr(response, "incomplete_details", None)
    incomplete_reason = getattr(incomplete_details, "reason", None)
    reason_suffix = f", reason={incomplete_reason}" if incomplete_reason else ""
    raise ValueError(
        f"OpenAI response did not contain text output (status={status}{reason_suffix})."
    )
