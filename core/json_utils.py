import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No JSON content supplied.")

    normalized = text.strip()
    if normalized.startswith("```"):
        for block in normalized.split("```"):
            candidate = block.strip()
            if not candidate:
                continue
            if candidate.startswith("json"):
                candidate = candidate[4:].lstrip()
            if "{" in candidate and "}" in candidate:
                normalized = candidate
                break

    payload = _find_first_object(normalized)
    for strict in (True, False):
        try:
            data = json.loads(payload, strict=strict)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
        raise ValueError("JSON payload is not an object.")

    raise ValueError("Unable to parse JSON object.")


def ensure_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def clamp_float(
    value: Any,
    default: float = 0.0,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(minimum, min(maximum, numeric))


def _find_first_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Unbalanced JSON object.")
