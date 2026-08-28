import ast
import re
from dataclasses import dataclass
from typing import Mapping


_EXACT_OUTPUT_PATTERN = re.compile(
    r"\b(?:output|outputs|write|writes|emit|emits)\s+exactly\s+"
    r"(?P<quoted>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s+"
    r"to\s+(?P<stream>stderr|stdout)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExactOutputContract:
    stream: str
    expected: str
    source_fragment: str


def extract_exact_output_contracts(requirement: str) -> list[ExactOutputContract]:
    contracts: list[ExactOutputContract] = []
    for match in _EXACT_OUTPUT_PATTERN.finditer(requirement):
        try:
            expected = ast.literal_eval(match.group("quoted"))
        except (SyntaxError, ValueError):
            continue
        if not isinstance(expected, str):
            continue
        contracts.append(
            ExactOutputContract(
                stream=match.group("stream").lower(),
                expected=expected,
                source_fragment=match.group(0),
            )
        )
    return contracts


def exact_output_contract_evidence(
    requirement: str,
    source_files: Mapping[str, str],
) -> list[dict[str, object]]:
    observations = _stream_literal_observations(source_files)
    evidence: list[dict[str, object]] = []
    for contract in extract_exact_output_contracts(requirement):
        observed = [
            item for item in observations if item["stream"] == contract.stream
        ]
        observed_values = list(
            dict.fromkeys(str(item["value"]) for item in observed)
        )
        evidence.append(
            {
                "stream": contract.stream,
                "expected": contract.expected,
                "source_fragment": contract.source_fragment,
                "observed": observed_values,
                "paths": list(
                    dict.fromkeys(str(item["path"]) for item in observed)
                ),
                "passed": contract.expected in observed_values,
                "failure_reason": (
                    ""
                    if contract.expected in observed_values
                    else "exact_output_mismatch"
                    if observed_values
                    else "exact_output_unproven"
                ),
            }
        )
    return evidence


def _stream_literal_observations(
    source_files: Mapping[str, str],
) -> list[dict[str, str]]:
    observations: list[dict[str, str]] = []
    for path, content in source_files.items():
        if not path.replace("\\", "/").startswith("src/"):
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            observed = _literal_stream_write(node)
            if observed is None:
                continue
            stream, value = observed
            observations.append({"path": path, "stream": stream, "value": value})
    return observations


def _literal_stream_write(node: ast.Call) -> tuple[str, str] | None:
    function_name = _expression_name(node.func)
    for stream in ("stderr", "stdout"):
        if function_name in {f"sys.{stream}.write", f"{stream}.write"}:
            value = _literal_text(node.args[0]) if node.args else None
            return (stream, value) if value is not None else None

    if function_name != "print":
        return None
    stream = "stdout"
    separator = " "
    ending = "\n"
    for keyword in node.keywords:
        if keyword.arg == "file":
            candidate = _expression_name(keyword.value)
            if candidate in {"sys.stderr", "stderr"}:
                stream = "stderr"
            elif candidate not in {"sys.stdout", "stdout"}:
                return None
        elif keyword.arg == "sep":
            separator_value = _literal_text(keyword.value)
            if separator_value is None:
                return None
            separator = separator_value
        elif keyword.arg == "end":
            ending_value = _literal_text(keyword.value)
            if ending_value is None:
                return None
            ending = ending_value
    values = [_literal_text(argument) for argument in node.args]
    if any(value is None for value in values):
        return None
    return stream, separator.join(value for value in values if value is not None) + ending


def _literal_text(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if isinstance(node.value, bytes):
            try:
                return node.value.decode("utf-8")
            except UnicodeDecodeError:
                return None
    return None


def _expression_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expression_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""
