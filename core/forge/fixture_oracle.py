import ast
import re
from dataclasses import asdict, dataclass
from typing import Any


_TOKEN_PATTERN = re.compile(r"[^ \t\r\f\n]+")
_INPUT_NAMES = {
    "content",
    "contents",
    "fixture",
    "input_content",
    "source",
    "source_text",
    "testin",
    "text",
}


@dataclass(frozen=True)
class FixtureOracleMismatch:
    capability_id: str
    function: str
    input_name: str
    expected_name: str
    input_line: int
    expected_line: int
    declared_expected: str
    derived_expected: str

    def to_evidence(self) -> dict[str, Any]:
        return asdict(self)


def fixture_oracle_capability(requirement: str) -> str | None:
    normalized = " ".join(requirement.lower().replace("-", " ").split())
    reverses_each_word = any(
        signal in normalized
        for signal in (
            "reverse each word",
            "reverse every word",
            "each word reversed",
            "every word reversed",
            "reversed in place",
            "word's character order reversed",
            "word’s character order reversed",
        )
    )
    if (
        reverses_each_word
        and "ascii whitespace" in normalized
        and any(
            signal in normalized
            for signal in ("word order preserved", "preserving word order")
        )
    ):
        return "reverse_ascii_whitespace_tokens"
    return None


def derive_fixture_output(requirement: str, value: str | bytes) -> str | bytes | None:
    if fixture_oracle_capability(requirement) != "reverse_ascii_whitespace_tokens":
        return None
    is_bytes = isinstance(value, bytes)
    try:
        text = value.decode("utf-8") if is_bytes else value
    except UnicodeDecodeError:
        return None
    transformed = _TOKEN_PATTERN.sub(lambda match: match.group(0)[::-1], text)
    return transformed.encode("utf-8") if is_bytes else transformed


def fixture_oracle_mismatches(
    source: str,
    requirement: str,
) -> list[FixtureOracleMismatch]:
    capability_id = fixture_oracle_capability(requirement)
    if capability_id is None:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    mismatches: list[FixtureOracleMismatch] = []
    for function in (
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ):
        environment: dict[str, str | bytes] = {}
        latest_input: tuple[str, str | bytes, int] | None = None
        for statement in function.body:
            assignment = _simple_assignment(statement)
            if assignment is None:
                continue
            name, expression = assignment
            value = _static_value(expression, environment)
            if not isinstance(value, (str, bytes)):
                continue
            environment[name] = value
            if name in _INPUT_NAMES or name.startswith("input_"):
                latest_input = (name, value, statement.lineno)
                continue
            if not _is_expected_name(name) or latest_input is None:
                continue
            input_name, input_value, input_line = latest_input
            derived = derive_fixture_output(requirement, input_value)
            if derived is None:
                continue
            comparable = _coerce_expected_type(derived, value)
            if comparable is None or comparable == value:
                continue
            mismatches.append(
                FixtureOracleMismatch(
                    capability_id=capability_id,
                    function=function.name,
                    input_name=input_name,
                    expected_name=name,
                    input_line=input_line,
                    expected_line=statement.lineno,
                    declared_expected=_bounded_repr(value),
                    derived_expected=_bounded_repr(comparable),
                )
            )
    return mismatches


def _simple_assignment(statement: ast.stmt) -> tuple[str, ast.expr] | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            return target.id, statement.value
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        if statement.value is not None:
            return statement.target.id, statement.value
    return None


def _static_value(
    expression: ast.expr,
    environment: dict[str, str | bytes],
) -> str | bytes | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, (str, bytes)):
        return expression.value
    if isinstance(expression, ast.Name):
        return environment.get(expression.id)
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _static_value(expression.left, environment)
        right = _static_value(expression.right, environment)
        if type(left) is type(right) and isinstance(left, (str, bytes)):
            return left + right
        return None
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Attribute):
        return None
    base = _static_value(expression.func.value, environment)
    arguments = [_static_value(argument, environment) for argument in expression.args]
    if expression.func.attr == "encode" and isinstance(base, str):
        encoding = arguments[0] if arguments else "utf-8"
        if encoding == "utf-8":
            return base.encode("utf-8")
    if expression.func.attr == "decode" and isinstance(base, bytes):
        encoding = arguments[0] if arguments else "utf-8"
        if encoding == "utf-8":
            try:
                return base.decode("utf-8")
            except UnicodeDecodeError:
                return None
    if (
        expression.func.attr == "replace"
        and isinstance(base, (str, bytes))
        and len(arguments) == 2
        and type(arguments[0]) is type(base)
        and type(arguments[1]) is type(base)
    ):
        return base.replace(arguments[0], arguments[1])
    return None


def _is_expected_name(name: str) -> bool:
    return name == "want" or name.startswith("expected") or name.startswith("desired")


def _coerce_expected_type(
    derived: str | bytes,
    declared: str | bytes,
) -> str | bytes | None:
    if type(derived) is type(declared):
        return derived
    if isinstance(derived, str) and isinstance(declared, bytes):
        return derived.encode("utf-8")
    if isinstance(derived, bytes) and isinstance(declared, str):
        try:
            return derived.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def _bounded_repr(value: str | bytes, limit: int = 320) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."
