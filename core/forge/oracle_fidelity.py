from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OracleOutputFidelityMismatch:
    function: str
    input_name: str
    expected_name: str
    input_line_endings: int
    expected_line_endings: int


@dataclass(frozen=True)
class _SequenceInfo:
    source_input: str
    item_count: int
    line_local: bool


def oracle_output_fidelity_mismatches(
    source: str,
    requirement: str,
) -> list[OracleOutputFidelityMismatch]:
    if not _requires_exact_line_endings(requirement):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    helpers = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("test_")
    }
    mismatches: list[OracleOutputFidelityMismatch] = []
    for function in (
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ):
        assignments = _assignments(function)
        sequences = _sequence_info(function, assignments, helpers)
        for expected_name, expected_expression in assignments.items():
            if not _is_expected_name(expected_name):
                continue
            if not _expected_is_asserted(function, expected_name):
                continue
            expected_count = _expression_line_ending_count(
                expected_expression,
                assignments,
                sequences,
            )
            input_name = _source_input_name(
                expected_expression,
                assignments,
                sequences,
            )
            input_value = _assigned_literal(input_name, assignments)
            if expected_count is None or input_value is None:
                continue
            input_count = _line_ending_count(input_value)
            if input_count != expected_count:
                mismatches.append(
                    OracleOutputFidelityMismatch(
                        function=function.name,
                        input_name=input_name,
                        expected_name=expected_name,
                        input_line_endings=input_count,
                        expected_line_endings=expected_count,
                    )
                )
    return mismatches


def _requires_exact_line_endings(requirement: str) -> bool:
    normalized = " ".join(requirement.lower().replace("-", " ").split())
    mentions_boundary = any(
        token in normalized for token in ("newline", "line ending", "line count")
    )
    requires_preservation = any(
        token in normalized for token in ("preserv", "identical", "must match", "same")
    )
    exact_or_counted = (
        "exact" in normalized
        or "trailing newline" in normalized
        or "line count" in normalized
    )
    return mentions_boundary and requires_preservation and exact_or_counted


def _assignments(function: ast.AST) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            value = node.value
        else:
            continue
        if isinstance(target, ast.Name):
            result[target.id] = value
    return result


def _sequence_info(
    function: ast.AST,
    assignments: dict[str, ast.expr],
    helpers: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> dict[str, _SequenceInfo]:
    sequences: dict[str, _SequenceInfo] = {}
    for name, expression in assignments.items():
        input_name = _splitlines_source(expression)
        input_value = _assigned_literal(input_name, assignments)
        if input_name and input_value is not None:
            sequences[name] = _SequenceInfo(
                source_input=input_name,
                item_count=len(input_value.splitlines()),
                line_local=True,
            )

    for loop in (
        node
        for node in ast.walk(function)
        if isinstance(node, (ast.For, ast.AsyncFor))
    ):
        input_name = _splitlines_source(loop.iter)
        input_value = _assigned_literal(input_name, assignments)
        if (
            not input_name
            or input_value is None
            or not isinstance(loop.target, ast.Name)
        ):
            continue
        for call in (
            node
            for statement in loop.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
        ):
            if (
                not isinstance(call.func, ast.Attribute)
                or call.func.attr != "append"
                or not isinstance(call.func.value, ast.Name)
                or not call.args
            ):
                continue
            list_name = call.func.value.id
            if not _is_empty_sequence(assignments.get(list_name)):
                continue
            sequences[list_name] = _SequenceInfo(
                source_input=input_name,
                item_count=len(input_value.splitlines()),
                line_local=_line_local_append(
                    call.args[0],
                    loop.target.id,
                    helpers,
                ),
            )
    return sequences


def _expression_line_ending_count(
    expression: ast.expr,
    assignments: dict[str, ast.expr],
    sequences: dict[str, _SequenceInfo],
    seen: frozenset[str] = frozenset(),
) -> int | None:
    literal = _constant_text(expression)
    if literal is not None:
        return _line_ending_count(literal)
    if isinstance(expression, ast.Name):
        if expression.id in seen or expression.id not in assignments:
            return None
        return _expression_line_ending_count(
            assignments[expression.id],
            assignments,
            sequences,
            seen | {expression.id},
        )
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _expression_line_ending_count(
            expression.left,
            assignments,
            sequences,
            seen,
        )
        right = _expression_line_ending_count(
            expression.right,
            assignments,
            sequences,
            seen,
        )
        return left + right if left is not None and right is not None else None
    if not _is_join_call(expression) or not isinstance(expression.args[0], ast.Name):
        return None
    sequence = sequences.get(expression.args[0].id)
    separator = _literal_text(expression.func.value, assignments)
    if sequence is None or not sequence.line_local or separator is None:
        return None
    return max(0, sequence.item_count - 1) * _line_ending_count(separator)


def _source_input_name(
    expression: ast.expr,
    assignments: dict[str, ast.expr],
    sequences: dict[str, _SequenceInfo],
    seen: frozenset[str] = frozenset(),
) -> str:
    if isinstance(expression, ast.Name):
        sequence = sequences.get(expression.id)
        if sequence is not None:
            return sequence.source_input
        if expression.id in seen or expression.id not in assignments:
            return ""
        return _source_input_name(
            assignments[expression.id],
            assignments,
            sequences,
            seen | {expression.id},
        )
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        return _source_input_name(
            expression.left, assignments, sequences, seen
        ) or _source_input_name(expression.right, assignments, sequences, seen)
    if _is_join_call(expression) and isinstance(expression.args[0], ast.Name):
        sequence = sequences.get(expression.args[0].id)
        return sequence.source_input if sequence is not None else ""
    return ""


def _literal_text(
    expression: ast.expr,
    assignments: dict[str, ast.expr],
    seen: frozenset[str] = frozenset(),
) -> str | bytes | None:
    literal = _constant_text(expression)
    if literal is not None:
        return literal
    if not isinstance(expression, ast.Name):
        return None
    if expression.id in seen or expression.id not in assignments:
        return None
    return _literal_text(
        assignments[expression.id],
        assignments,
        seen | {expression.id},
    )


def _assigned_literal(
    name: str,
    assignments: dict[str, ast.expr],
) -> str | bytes | None:
    return _literal_text(assignments[name], assignments) if name in assignments else None


def _splitlines_source(expression: ast.expr) -> str:
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "splitlines"
        and isinstance(expression.func.value, ast.Name)
    ):
        return expression.func.value.id
    return ""


def _line_local_append(
    expression: ast.expr,
    loop_name: str,
    helpers: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> bool:
    if isinstance(expression, ast.Name):
        return expression.id == loop_name
    if not isinstance(expression, ast.Call):
        return False
    helper = helpers.get(_call_name(expression.func))
    if helper is None:
        return False
    line_parameters = _line_derived_parameters(expression, helper, loop_name)
    return bool(line_parameters) and _helper_is_line_local(helper, line_parameters)


def _line_derived_parameters(
    call: ast.Call,
    helper: ast.FunctionDef | ast.AsyncFunctionDef,
    loop_name: str,
) -> set[str]:
    positional = [*helper.args.posonlyargs, *helper.args.args]
    result = {
        parameter.arg
        for parameter, argument in zip(positional, call.args)
        if isinstance(argument, ast.Name) and argument.id == loop_name
    }
    result.update(
        keyword.arg
        for keyword in call.keywords
        if keyword.arg is not None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == loop_name
    )
    return result


def _helper_is_line_local(
    function: ast.AST,
    line_parameters: set[str],
) -> bool:
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    return bool(returns) and all(
        node.value is not None
        and _return_is_line_local(node.value, line_parameters, function)
        for node in returns
    )


def _return_is_line_local(
    expression: ast.expr,
    parameters: set[str],
    function: ast.AST,
) -> bool:
    literal = _constant_text(expression)
    if literal is not None:
        return _line_ending_count(literal) == 0
    if isinstance(expression, ast.Name):
        return expression.id in parameters
    if not _is_join_call(expression):
        return False
    separator = _constant_text(expression.func.value)
    joined = expression.args[0]
    return (
        separator is not None
        and _line_ending_count(separator) == 0
        and isinstance(joined, ast.Name)
        and _helper_sequence_is_line_local(function, joined.id, parameters)
    )


def _helper_sequence_is_line_local(
    function: ast.AST,
    sequence_name: str,
    parameters: set[str],
) -> bool:
    initializers = [
        expression
        for name, expression in _named_assignments(function)
        if name == sequence_name
    ]
    if len(initializers) != 1 or not _is_empty_sequence(initializers[0]):
        return False

    appends: list[ast.expr] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != sequence_name:
            continue
        if node.func.attr != "append" or len(node.args) != 1:
            return False
        appends.append(node.args[0])
    if not appends:
        return False

    assignments = dict(_named_assignments(function))
    loop_sources = _loop_value_sources(function)
    return all(
        _helper_value_is_line_local(
            expression,
            parameters,
            assignments,
            loop_sources,
        )
        for expression in appends
    )


def _helper_value_is_line_local(
    expression: ast.expr,
    parameters: set[str],
    assignments: dict[str, ast.expr],
    loop_sources: dict[str, ast.expr],
    seen: frozenset[str] = frozenset(),
) -> bool:
    literal = _constant_text(expression)
    if literal is not None:
        return _line_ending_count(literal) == 0
    if not isinstance(expression, ast.Name):
        return False
    if expression.id in parameters:
        return True
    if expression.id in seen:
        return False
    source = loop_sources.get(expression.id) or assignments.get(expression.id)
    if source is None:
        return False
    if _is_safe_split_call(source):
        base = source.func.value
        return _helper_value_is_line_local(
            base,
            parameters,
            assignments,
            loop_sources,
            seen | {expression.id},
        ) and all(
            (value := _constant_text(argument)) is not None
            and _line_ending_count(value) == 0
            for argument in source.args
        )
    return _helper_value_is_line_local(
        source,
        parameters,
        assignments,
        loop_sources,
        seen | {expression.id},
    )


def _named_assignments(function: ast.AST) -> list[tuple[str, ast.expr]]:
    result: list[tuple[str, ast.expr]] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            value = node.value
        else:
            continue
        if isinstance(target, ast.Name):
            result.append((target.id, value))
    return result


def _loop_value_sources(function: ast.AST) -> dict[str, ast.expr]:
    return {
        node.target.id: node.iter
        for node in ast.walk(function)
        if isinstance(node, (ast.For, ast.AsyncFor))
        and isinstance(node.target, ast.Name)
    }


def _is_safe_split_call(expression: ast.expr) -> bool:
    return (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr in {"split", "splitlines"}
        and not expression.keywords
    )


def _expected_is_asserted(function: ast.AST, expected_name: str) -> bool:
    return any(
        isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(item, ast.Name) and item.id == expected_name
            for item in ast.walk(node.test)
        )
        for node in ast.walk(function)
    )


def _is_join_call(expression: ast.expr) -> bool:
    return (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr == "join"
        and len(expression.args) == 1
    )


def _is_empty_sequence(expression: ast.expr | None) -> bool:
    return isinstance(expression, (ast.List, ast.Tuple)) and not expression.elts


def _is_expected_name(name: str) -> bool:
    normalized = name.lower()
    return normalized == "want" or normalized.startswith(("expected", "desired"))


def _constant_text(expression: ast.expr) -> str | bytes | None:
    return (
        expression.value
        if isinstance(expression, ast.Constant)
        and isinstance(expression.value, (str, bytes))
        else None
    )


def _line_ending_count(value: str | bytes) -> int:
    pattern = rb"\r\n|\r|\n" if isinstance(value, bytes) else r"\r\n|\r|\n"
    return len(re.findall(pattern, value))


def _call_name(expression: ast.expr) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return expression.attr
    return ""
