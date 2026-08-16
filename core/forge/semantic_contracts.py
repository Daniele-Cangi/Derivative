import ast
from typing import Any, Iterable, Mapping


def non_semantic_test_paths(
    test_paths: Iterable[str],
    file_contents: Mapping[str, str],
    target_names: Iterable[str] = (),
    target_modules: Iterable[str] = (),
) -> list[str]:
    """Return test files that do not causally assert generated behavior."""
    from core.forge.test_evidence import non_semantic_test_reasons

    return sorted(
        non_semantic_test_reasons(
            test_paths,
            file_contents,
            target_names=target_names,
            target_modules=target_modules,
        )
    )

def _expression_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expression_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def structurally_evidences(
    term: str,
    source_content: str,
    interfaces: Iterable[Any],
) -> bool:
    if term != "cli_entrypoint":
        return False
    expected_names = {
        interface.name
        for interface in interfaces
        if getattr(interface, "interface_type", "") == "cli_entrypoint"
        and getattr(interface, "name", "").isidentifier()
    }
    if not expected_names:
        return False
    try:
        tree = ast.parse(source_content)
    except SyntaxError:
        return False
    defined_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return bool(expected_names & defined_names)


def behaviorally_evidences(
    term: str,
    content: str,
    public_interface_names: set[str] | None = None,
) -> bool:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    if term in {
        "malformed_records",
        "malformed_rows",
        "missing_fields",
        "invalid_dates",
        "invalid_timestamp",
    }:
        return has_expected_exception_assertion(content, public_interface_names)
    if term == "duplicate_ids":
        return has_duplicate_id_rejection_test(content, public_interface_names)
    if term == "json_object_root_validation":
        normalized = content.lower().replace("-", "_").replace(" ", "_")
        constants = {
            node.value.strip()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        has_non_object_fixture = bool(
            constants & {"[]", "null", "true", "false", '"value"', "1"}
        ) or any(
            marker in normalized
            for marker in (
                "non_object_root",
                "rejects_non_object",
                "root_must_be_an_object",
                "json_root_must_be_an_object",
            )
        )
        return (
            has_non_object_fixture
            and has_expected_exception_assertion(
                content,
                public_interface_names or {"main", "run", "merge"},
            )
        )
    if term != "json_list_replacement":
        return False
    list_literals = [node for node in ast.walk(tree) if isinstance(node, (ast.List, ast.Tuple))]
    equality_assertions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Compare)
        and any(isinstance(operator, (ast.Eq, ast.NotEq)) for operator in node.test.ops)
        and any(
            isinstance(value, (ast.List, ast.Tuple))
            for value in [node.test.left, *node.test.comparators]
        )
    ]
    return len(list_literals) >= 2 and bool(equality_assertions)


def has_json_lines_processing(content: str) -> bool:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            target_names = _assigned_names(node.target)
            if target_names and _body_loads_json_from_names(node.body, target_names):
                return True
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            for generator in node.generators:
                target_names = _assigned_names(generator.target)
                if target_names and _expression_loads_json_from_names(node.elt, target_names):
                    return True
        if isinstance(node, ast.DictComp):
            for generator in node.generators:
                target_names = _assigned_names(generator.target)
                if target_names and (
                    _expression_loads_json_from_names(node.key, target_names)
                    or _expression_loads_json_from_names(node.value, target_names)
                ):
                    return True
    return False


def has_duplicate_id_rejection_test(
    content: str,
    target_names: set[str] | None = None,
) -> bool:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False

    id_values: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "id"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                id_values.append(value.value)
    has_duplicate_fixture = len(id_values) != len(set(id_values))
    return has_duplicate_fixture and has_expected_exception_assertion(
        content,
        target_names,
    )


def has_expected_exception_assertion(
    content: str,
    target_names: set[str] | None = None,
) -> bool:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                expression = item.context_expr
                if not isinstance(expression, ast.Call):
                    continue
                function = expression.func
                if isinstance(function, ast.Attribute) and function.attr in {"raises", "assertRaises"}:
                    if target_names is None or _contains_target_call(node.body, target_names):
                        return True
        if isinstance(node, ast.Try):
            handled = {
                name
                for handler in node.handlers
                for name in _exception_names(handler.type)
            }
            if handled & {"ValueError", "TypeError", "SystemExit"} and (
                target_names is None or _contains_target_call(node.body, target_names)
            ):
                return True
    return False


def has_canonicalized_deduplication_assertion(content: str) -> bool:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False

    literals: dict[str, list[str]] = {}
    calls: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        literal = _string_list(node.value, literals)
        if literal is not None:
            literals[target.id] = literal
            continue
        call_input = _deduplication_call_input(node.value, literals)
        if call_input is not None:
            calls[target.id] = call_input

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        if not any(isinstance(operator, ast.Eq) for operator in node.test.ops):
            continue
        expressions = [node.test.left, *node.test.comparators]
        for index, expression in enumerate(expressions):
            call_input = _deduplication_call_input(expression, literals)
            if call_input is None and isinstance(expression, ast.Name):
                call_input = calls.get(expression.id)
            if call_input is None:
                continue
            for expected_expression in expressions[:index] + expressions[index + 1 :]:
                expected = _string_list(expected_expression, literals)
                if expected is not None and _is_canonical_deduplication(call_input, expected):
                    return True
    return False


def interface_parameter_is_exercised(
    term: str,
    test_content: str,
    source_content: str,
    public_interface_names: set[str],
) -> bool:
    if not term.isidentifier() or not public_interface_names:
        return False
    try:
        source_tree = ast.parse(source_content)
        test_tree = ast.parse(test_content)
    except SyntaxError:
        return False

    public_parameters = {
        argument.arg
        for node in ast.walk(source_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in public_interface_names
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    }
    if term not in public_parameters:
        return False

    for node in ast.walk(test_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        invokes_public_interface = any(
            isinstance(child, ast.Call)
            and _call_name(child) in public_interface_names
            for child in ast.walk(node)
        )
        has_assertion = any(isinstance(child, ast.Assert) for child in ast.walk(node))
        if invokes_public_interface and has_assertion:
            return True
    return False


def has_end_to_end_file_workflow_test(
    content: str,
    public_interface_names: set[str] | None = None,
) -> bool:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    interface_names = public_interface_names or {"main", "run"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
        invokes_interface = any(_call_name(call) in interface_names for call in calls)
        writes_fixture = any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr in {"write_text", "write_bytes", "writerow", "writerows"}
            for call in calls
        )
        reads_result = any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr in {"read_text", "read_bytes", "exists"}
            for call in calls
        )
        assertion_count = sum(
            1 for child in ast.walk(node) if isinstance(child, ast.Assert)
        )
        if invokes_interface and writes_fixture and reads_result and assertion_count >= 2:
            return True
    return False


def _contains_target_call(statements: list[ast.stmt], target_names: set[str]) -> bool:
    return any(
        isinstance(node, ast.Call) and _call_name(node) in target_names
        for statement in statements
        for node in ast.walk(statement)
    )


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _assigned_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    }


def _body_loads_json_from_names(statements: list[ast.stmt], names: set[str]) -> bool:
    return any(
        _is_json_loads_call(node) and _call_references_names(node, names)
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    )


def _expression_loads_json_from_names(expression: ast.AST, names: set[str]) -> bool:
    return any(
        _is_json_loads_call(node) and _call_references_names(node, names)
        for node in ast.walk(expression)
        if isinstance(node, ast.Call)
    )


def _is_json_loads_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "loads"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "json"
    )


def _call_references_names(node: ast.Call, names: set[str]) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id in names
        for argument in node.args
        for child in ast.walk(argument)
    )


def _string_list(node: ast.AST, literals: dict[str, list[str]]) -> list[str] | None:
    if isinstance(node, ast.Name):
        return literals.get(node.id)
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        values.append(item.value)
    return values


def _deduplication_call_input(
    node: ast.AST,
    literals: dict[str, list[str]],
) -> list[str] | None:
    if not isinstance(node, ast.Call) or _call_name(node) != "deduplicate_emails" or not node.args:
        return None
    return _string_list(node.args[0], literals)


def _is_canonical_deduplication(values: list[str], expected: list[str]) -> bool:
    if not values or not any(value != value.strip().lower() for value in values):
        return False
    canonical: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            canonical.append(normalized)
    return expected == canonical


def _exception_names(node: ast.expr | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Tuple):
        return [name for item in node.elts for name in _exception_names(item)]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return []
