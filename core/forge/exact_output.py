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
    precondition: str


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
                precondition=_extract_precondition(requirement, match.start()),
            )
        )
    return contracts


def exact_output_contract_evidence(
    requirement: str,
    source_files: Mapping[str, str],
    target_names: set[str] | None = None,
) -> list[dict[str, object]]:
    observations, calls = _source_observations(source_files)
    evidence: list[dict[str, object]] = []
    for contract in extract_exact_output_contracts(requirement):
        stream_observations = [
            item for item in observations if item["stream"] == contract.stream
        ]
        observed = [
            item
            for item in stream_observations
            if _observation_matches_precondition(
                item,
                contract.precondition,
                calls,
                target_names,
            )
        ]
        observed_values = list(
            dict.fromkeys(str(item["value"]) for item in observed)
        )
        unbound_values = list(
            dict.fromkeys(
                str(item["value"])
                for item in stream_observations
                if item not in observed
            )
        )
        evidence.append(
            {
                "stream": contract.stream,
                "expected": contract.expected,
                "source_fragment": contract.source_fragment,
                "precondition": contract.precondition,
                "observed": observed_values,
                "unbound_observed": unbound_values,
                "paths": list(
                    dict.fromkeys(
                        str(item["path"])
                        for item in (observed or stream_observations)
                    )
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


def _source_observations(
    source_files: Mapping[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    observations: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []
    for path, content in source_files.items():
        if not path.replace("\\", "/").startswith("src/"):
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        visitor = _StreamObservationVisitor(path)
        visitor.visit(tree)
        observations.extend(visitor.observations)
        calls.extend(visitor.calls)
    return observations, calls


class _StreamObservationVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.functions: list[str] = []
        self.conditions: list[str] = []
        self.observations: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        condition = ast.unparse(node.test)
        truth = _static_truth_value(node.test)
        if truth is not False:
            self.conditions.append(condition)
            for statement in node.body:
                self.visit(statement)
            self.conditions.pop()
        if node.orelse and truth is not True:
            self.conditions.append(f"not ({condition})")
            for statement in node.orelse:
                self.visit(statement)
            self.conditions.pop()

    def visit_Try(self, node: ast.Try) -> None:
        for statement in node.body:
            self.visit(statement)
        for handler in node.handlers:
            exception_name = _expression_name(handler.type) if handler.type else "exception"
            self.conditions.append(f"except {exception_name}")
            for statement in handler.body:
                self.visit(statement)
            self.conditions.pop()
        for statement in [*node.orelse, *node.finalbody]:
            self.visit(statement)

    def visit_Call(self, node: ast.Call) -> None:
        function_name = _expression_name(node.func)
        if self.functions and function_name:
            self.calls.append(
                {
                    "path": self.path,
                    "caller": self.functions[-1],
                    "callee": function_name.rsplit(".", 1)[-1],
                    "conditions": tuple(self.conditions),
                }
            )
        observed = _literal_stream_write(node)
        if observed is not None:
            stream, value = observed
            self.observations.append(
                {
                    "path": self.path,
                    "stream": stream,
                    "value": value,
                    "function": self.functions[-1] if self.functions else "",
                    "conditions": tuple(self.conditions),
                    "line": getattr(node, "lineno", 0),
                }
            )
        self.generic_visit(node)


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
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_precondition(requirement: str, output_start: int) -> str:
    clause_start = max(
        requirement.rfind(delimiter, 0, output_start)
        for delimiter in ".!?;"
    )
    prefix = requirement[clause_start + 1 : output_start]
    match = re.search(
        r"\b(?:if|when|on|for)\s+(?P<condition>[^,;:.!?]+?)(?:,|\bthe\s+tool\b|$)",
        prefix,
        re.IGNORECASE,
    )
    return " ".join(match.group("condition").split()) if match else ""


def _observation_matches_precondition(
    observation: Mapping[str, object],
    precondition: str,
    calls: list[dict[str, object]],
    target_names: set[str] | None,
) -> bool:
    if not precondition:
        return True
    required_features = _semantic_features(precondition)
    function_name = str(observation.get("function", ""))
    local_context = " ".join(
        str(value) for value in observation.get("conditions", ())
    )
    if not target_names:
        return bool(
            required_features
            & _semantic_features(f"{function_name} {local_context}")
        )

    path = str(observation.get("path", ""))
    reachable_context = _reachable_function_contexts(path, calls, target_names)
    if function_name not in reachable_context:
        return False
    context_features = reachable_context[function_name] | _semantic_features(local_context)
    return bool(required_features & context_features)


def _reachable_function_contexts(
    path: str,
    calls: list[dict[str, object]],
    target_names: set[str],
) -> dict[str, set[str]]:
    path_calls = [item for item in calls if item.get("path") == path]
    contexts = {
        name: _semantic_features(name)
        for name in target_names
    }
    changed = True
    while changed:
        changed = False
        for call in path_calls:
            caller = str(call.get("caller", ""))
            callee = str(call.get("callee", ""))
            if caller not in contexts or not callee:
                continue
            propagated = contexts[caller] | _semantic_features(
                " ".join(str(value) for value in call.get("conditions", ()))
            )
            existing = contexts.setdefault(callee, set())
            if not propagated.issubset(existing):
                existing.update(propagated)
                changed = True
    return contexts


def _semantic_features(value: str) -> set[str]:
    split_identifiers = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    lowered = re.sub(r"[_-]+", " ", split_identifiers.lower())
    features = {
        token
        for token in re.findall(r"[a-z0-9]+", lowered)
        if token
        not in {
            "a",
            "an",
            "data",
            "for",
            "if",
            "input",
            "is",
            "on",
            "the",
            "when",
        }
    }
    invalid_aliases = {"error", "exception", "fail", "failed", "failure", "invalid", "reject"}
    if features & invalid_aliases or re.search(r"\bnot\s+valid\b", lowered):
        features.add("invalid")
    if features & {"absent", "empty", "missing", "none"}:
        features.add("missing")
    if features & {"bad", "malformed"}:
        features.add("malformed")
    return features


def _static_truth_value(node: ast.expr) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int)):
        return bool(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _static_truth_value(node.operand)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_static_truth_value(value) for value in node.values]
        if isinstance(node.op, ast.And) and False in values:
            return False
        if isinstance(node.op, ast.Or) and all(value is False for value in values):
            return False
    return None


def _expression_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expression_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""
