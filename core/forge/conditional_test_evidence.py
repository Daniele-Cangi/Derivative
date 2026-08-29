import ast
import re
from typing import Any, Iterable, Mapping

from core.forge.contracts import BuildSpec, ConditionalObligation, FeasiblePlan


_LOSSY_METHODS = frozenset(
    {
        "casefold",
        "lower",
        "lstrip",
        "replace",
        "rstrip",
        "sort",
        "splitlines",
        "strip",
        "upper",
    }
)
_LOSSY_FUNCTIONS = frozenset({"set", "sorted"})

def analyze_test_expectations(
    contents: Mapping[str, str],
    build_spec: BuildSpec,
    plan: FeasiblePlan,
) -> list[dict[str, Any]]:
    obligations_by_id = {
        obligation.obligation_id: obligation
        for obligation in build_spec.conditional_obligations
    }
    planned_by_path = {
        f"tests/{test.test_name}.py": test
        for test in plan.required_tests
    }
    evidence: list[dict[str, Any]] = []
    for path, content in sorted(contents.items()):
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        planned = planned_by_path.get(path)
        mapped_ids = set(planned.conditional_obligation_ids if planned else [])
        for function in _test_functions(tree):
            source = ast.get_source_segment(content, function) or ast.unparse(function)
            witness_classes = _infer_witness_classes(function, source, plan)
            exercised = [
                obligation
                for obligation in obligations_by_id.values()
                if (
                    obligation.witness_class in witness_classes
                    if witness_classes
                    else obligation.obligation_id in mapped_ids
                )
            ]
            observations = _asserted_observations(function)
            contradictions: list[dict[str, Any]] = []
            for obligation in exercised:
                matching = [
                    item
                    for item in observations
                    if item["channel"] == obligation.observable_channel
                ]
                for item in matching:
                    if _expectation_contradicts(obligation, item):
                        contradictions.append(
                            {
                                "obligation_id": obligation.obligation_id,
                                "expected_relation": obligation.comparison_relation,
                                "expected_value": obligation.expected_value,
                                "asserted_relation": item["relation"],
                                "asserted_value": item["value"],
                                "line": item["line"],
                            }
                        )
            evidence.append(
                {
                    "path": path,
                    "function": function.name,
                    "witness_classes": sorted(witness_classes),
                    "exercised_obligation_ids": sorted(
                        obligation.obligation_id for obligation in exercised
                    ),
                    "observations": observations,
                    "contradictions": contradictions,
                }
            )
    return evidence


def analyze_observation_fidelity(
    contents: Mapping[str, str],
    build_spec: BuildSpec,
    plan: FeasiblePlan,
) -> list[dict[str, Any]]:
    exact_obligations = {
        obligation.obligation_id: obligation
        for obligation in build_spec.conditional_obligations
        if obligation.observation_fidelity in {"exact_text", "exact_bytes"}
    }
    planned_by_path = {
        f"tests/{test.test_name}.py": test
        for test in plan.required_tests
    }
    evidence: list[dict[str, Any]] = []
    for path, content in sorted(contents.items()):
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        planned = planned_by_path.get(path)
        mapped_ids = set(planned.conditional_obligation_ids if planned else [])
        for function in _test_functions(tree):
            source = ast.get_source_segment(content, function) or ast.unparse(function)
            witnesses = _infer_witness_classes(function, source, plan)
            applicable = [
                obligation
                for obligation in exact_obligations.values()
                if (
                    obligation.witness_class in witnesses
                    if witnesses
                    else obligation.obligation_id in mapped_ids
                )
            ]
            lossy: list[dict[str, Any]] = []
            for assertion in (node for node in ast.walk(function) if isinstance(node, ast.Assert)):
                transformations = _lossy_transformations(assertion.test)
                if not transformations:
                    continue
                channels = _observed_channels(assertion.test)
                for obligation in applicable:
                    if obligation.observable_channel not in channels:
                        continue
                    lossy.append(
                        {
                            "obligation_id": obligation.obligation_id,
                            "channel": obligation.observable_channel,
                            "line": getattr(assertion, "lineno", 0),
                            "transformations": sorted(transformations),
                        }
                    )
            evidence.append(
                {
                    "path": path,
                    "function": function.name,
                    "exact_obligation_ids": sorted(
                        obligation.obligation_id for obligation in applicable
                    ),
                    "lossy_observations": lossy,
                }
            )
    return evidence


def _test_functions(tree: ast.AST) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    return (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def _infer_witness_classes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    plan: FeasiblePlan,
) -> set[str]:
    lowered = f"{function.name} {source}".lower().replace("-", "_")
    witnesses: set[str] = set()
    patterns = (
        ("empty_input", r"empty_(?:file|input)|write_(?:text|bytes)\s*\(\s*(?:b)?[\"']{2}"),
        ("numeric_argument_exceeds_input_length", r"(?:greater|exceeds?|larger|longer)_than_(?:input|length)|size_exceeds"),
        ("invalid_positive_integer", r"invalid_(?:chunk_)?size|non_integer|not_integer|negative_size|zero_size"),
        ("invalid_argument_count", r"argument_count|missing_argument|too_(?:many|few)_arguments"),
        ("utf8_decode_failure", r"invalid_utf_?8|decode_failure"),
        ("file_read_failure", r"read_failure|missing_file|nonexistent_file"),
        ("no_separator_output", r"no_separator|separator"),
    )
    for witness, pattern in patterns:
        if re.search(pattern, lowered):
            witnesses.add(witness)

    writes = _literal_file_writes(function)
    calls = _literal_main_calls(function)
    if any(value in {"", b""} for value in writes):
        witnesses.add("empty_input")
    for args in calls:
        numeric = [int(value) for value in args if isinstance(value, str) and re.fullmatch(r"\d+", value)]
        text_lengths = [len(value) for value in writes if isinstance(value, str) and value]
        if numeric and text_lengths and max(numeric) > min(text_lengths):
            witnesses.add("numeric_argument_exceeds_input_length")
        if any(
            isinstance(value, str)
            and (not re.fullmatch(r"\d+", value) or int(value) <= 0)
            for value in args[1:]
        ):
            witnesses.add("invalid_positive_integer")
    expected_count = next(
        (
            interface.explicit_argv_count
            for interface in plan.interfaces
            if interface.interface_type == "cli_entrypoint"
            and interface.explicit_argv_count is not None
        ),
        None,
    )
    if expected_count is not None and any(len(args) != expected_count for args in calls):
        witnesses.add("invalid_argument_count")
    if any(isinstance(value, bytes) and _invalid_utf8(value) for value in writes):
        witnesses.add("utf8_decode_failure")
    return witnesses


def _literal_file_writes(function: ast.AST) -> list[str | bytes]:
    values: list[str | bytes] = []
    for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
        name = _call_name(call.func)
        if name not in {"write_text", "write_bytes"} or not call.args:
            continue
        value = _literal_value(call.args[0])
        if isinstance(value, (str, bytes)):
            values.append(value)
    return values


def _literal_main_calls(function: ast.AST) -> list[list[Any]]:
    calls: list[list[Any]] = []
    for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
        if _call_name(call.func) not in {"main", "run"} or not call.args:
            continue
        argument = call.args[0]
        if isinstance(argument, (ast.List, ast.Tuple)):
            calls.append([_literal_value(item) for item in argument.elts])
    return calls


def _asserted_observations(function: ast.AST) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for assertion in (node for node in ast.walk(function) if isinstance(node, ast.Assert)):
        expression = assertion.test
        if not isinstance(expression, ast.Compare) or len(expression.ops) != 1 or len(expression.comparators) != 1:
            continue
        left, right = expression.left, expression.comparators[0]
        channel = _expression_channel(left) or _expression_channel(right)
        if channel is None:
            continue
        literal = _literal_value(right if _expression_channel(left) else left)
        if literal is _UNSET:
            continue
        operator = expression.ops[0]
        relation = {
            ast.Eq: "equals",
            ast.NotEq: "not_equals",
            ast.In: "contains",
            ast.NotIn: "not_contains",
        }.get(type(operator))
        if relation is None:
            continue
        if isinstance(operator, (ast.In, ast.NotIn)) and _expression_channel(right):
            value = _literal_value(left)
        else:
            value = literal
        observations.append(
            {
                "channel": channel,
                "relation": relation,
                "value": value,
                "line": getattr(assertion, "lineno", 0),
            }
        )
    return observations


def _expectation_contradicts(
    obligation: ConditionalObligation,
    asserted: Mapping[str, Any],
) -> bool:
    expected_relation = obligation.comparison_relation
    asserted_relation = asserted["relation"]
    expected_value = obligation.expected_value
    asserted_value = asserted["value"]
    if expected_relation == "equals":
        return (
            asserted_relation == "equals" and asserted_value != expected_value
        ) or (
            asserted_relation == "not_equals" and asserted_value == expected_value
        )
    if expected_relation == "not_equals":
        return asserted_relation == "equals" and asserted_value == expected_value
    if expected_relation == "contains":
        return asserted_relation == "not_contains" and asserted_value == expected_value
    if expected_relation == "not_contains":
        return asserted_relation == "contains" and asserted_value == expected_value
    if expected_relation == "raises":
        return asserted_relation == "equals" and asserted_value != expected_value
    return False


def _expression_channel(expression: ast.AST) -> str | None:
    names = {
        name.lower()
        for node in ast.walk(expression)
        for name in [_node_name(node)]
        if name
    }
    if names & {"stdout", "out"}:
        return "stdout"
    if names & {"stderr", "err"}:
        return "stderr"
    if names & {"exception", "exc", "error_type"}:
        return "exception"
    if names & {"returncode", "exit_code", "status", "code", "result"}:
        return "exit_code"
    return None


def _observed_channels(expression: ast.AST) -> set[str]:
    return {
        channel
        for node in ast.walk(expression)
        for channel in [_expression_channel(node)]
        if channel
    }


def _lossy_transformations(expression: ast.AST) -> set[str]:
    transformations: set[str] = set()
    for node in ast.walk(expression):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in _LOSSY_METHODS or name in _LOSSY_FUNCTIONS:
            transformations.add(name)
    return transformations


def _call_name(expression: ast.AST) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return expression.attr
    return ""


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


class _Unset:
    pass


_UNSET = _Unset()


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return _UNSET


def _invalid_utf8(value: bytes) -> bool:
    try:
        value.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False
