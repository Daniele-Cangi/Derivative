import ast
from typing import Iterable, Mapping


_OBSERVER_CALL_NAMES = frozenset(
    {
        "DictReader",
        "execute",
        "exists",
        "fetchall",
        "fetchone",
        "getvalue",
        "load",
        "loads",
        "open",
        "read",
        "read_bytes",
        "read_text",
        "readouterr",
    }
)


def non_semantic_test_reasons(
    test_paths: Iterable[str],
    file_contents: Mapping[str, str],
    target_names: Iterable[str] = (),
    target_modules: Iterable[str] = (),
) -> dict[str, list[str]]:
    reasons_by_path: dict[str, list[str]] = {}
    expected_names = {name for name in target_names if name}
    expected_modules = {name for name in target_modules if name}
    target_contract_declared = bool(expected_names or expected_modules)
    for test_path in sorted(set(test_paths)):
        content = file_contents.get(test_path)
        if content is None:
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        test_functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ]
        if not test_functions:
            reasons_by_path[test_path] = ["missing_test_function"]
            continue
        if any(
            _is_tautological_assertion(node)
            for function in test_functions
            for node in ast.walk(function)
            if isinstance(node, ast.Assert)
        ):
            reasons_by_path[test_path] = ["tautological_assertion"]
            continue
        module_aliases, imported_target_names = _target_import_context(
            tree,
            expected_names,
            expected_modules,
        )
        states = [
            _test_function_semantic_state(
                function,
                target_names=expected_names | imported_target_names,
                target_module_aliases=module_aliases,
                target_contract_declared=target_contract_declared,
            )
            for function in test_functions
        ]
        if any(state["semantic"] for state in states):
            continue
        if target_contract_declared and not any(
            state["target_invoked"] for state in states
        ):
            reasons_by_path[test_path] = ["missing_target_invocation"]
        elif any(state["has_assertion"] for state in states):
            reasons_by_path[test_path] = ["disconnected_assertion"]
        else:
            reasons_by_path[test_path] = ["missing_behavioral_assertion"]
    return reasons_by_path


def analyze_test_functions(
    content: str,
    target_names: Iterable[str] = (),
    target_modules: Iterable[str] = (),
) -> list[dict[str, object]]:
    expected_names = {name for name in target_names if name}
    expected_modules = {name for name in target_modules if name}
    target_contract_declared = bool(expected_names or expected_modules)
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    module_aliases, imported_target_names = _target_import_context(
        tree,
        expected_names,
        expected_modules,
    )
    evidence: list[dict[str, object]] = []
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ):
        state = _test_function_semantic_state(
            function,
            target_names=expected_names | imported_target_names,
            target_module_aliases=module_aliases,
            target_contract_declared=target_contract_declared,
        )
        evidence.append(
            {
                **state,
                "function": function.name,
                "source": ast.get_source_segment(content, function)
                or ast.unparse(function),
            }
        )
    return evidence


def source_module_names(paths: Iterable[str]) -> set[str]:
    modules: set[str] = set()
    for path in paths:
        normalized = path.replace("\\", "/")
        if not normalized.startswith("src/") or not normalized.endswith(".py"):
            continue
        module_parts = normalized[4:-3].split("/")
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        if not module_parts:
            continue
        dotted = ".".join(module_parts)
        modules.update({dotted, module_parts[0], module_parts[-1]})
    return modules


def _test_function_semantic_state(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    target_names: set[str],
    target_module_aliases: set[str],
    target_contract_declared: bool,
) -> dict[str, object]:
    if function.name in {"test_acceptance_requirement", "test_stub"}:
        return {
            "semantic": False,
            "target_invoked": False,
            "has_assertion": False,
            "assertions": [],
        }
    target_names = target_names | _static_target_callable_aliases(
        function,
        target_names,
        target_module_aliases,
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    target_calls = [
        call
        for call in calls
        if _call_matches_target(call, target_names, target_module_aliases)
    ]
    if not target_contract_declared:
        target_calls = calls
    has_assertion = any(isinstance(node, ast.Assert) for node in ast.walk(function))
    target_invoked = bool(target_calls)
    if not target_invoked:
        return {
            "semantic": False,
            "target_invoked": False,
            "has_assertion": has_assertion,
            "assertions": [],
        }
    exception_assertions = _target_exception_assertion_records(
        function,
        target_names,
        target_module_aliases,
        target_contract_declared,
    )
    if exception_assertions:
        return {
            "semantic": True,
            "target_invoked": True,
            "has_assertion": True,
            "assertions": exception_assertions,
        }
    observable_names = _observable_names(
        function,
        target_calls,
        target_names,
        target_module_aliases,
    )
    observable_assertions = [
        {
            "line": getattr(node, "lineno", 0),
            "kind": "assert",
            "expression": ast.unparse(node.test),
        }
        for node in ast.walk(function)
        if isinstance(node, ast.Assert) and _is_semantic_assertion(node)
        and _assertion_observes_target(
            node,
            target_calls,
            observable_names,
            target_names,
            target_module_aliases,
        )
    ]
    return {
        "semantic": bool(observable_assertions),
        "target_invoked": True,
        "has_assertion": has_assertion,
        "assertions": observable_assertions,
    }


def _static_target_callable_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    target_names: set[str],
    target_module_aliases: set[str],
) -> set[str]:
    aliases: set[str] = set()
    assignments = sorted(
        (
            (node, value, targets)
            for node in ast.walk(function)
            for value, targets in [_assignment_value_and_targets(node)]
            if value is not None and targets
        ),
        key=lambda item: getattr(item[0], "lineno", 0),
    )
    changed = True
    while changed:
        changed = False
        known_names = target_names | aliases
        for _node, value, targets in assignments:
            if not _expression_is_static_target_callable(
                value,
                known_names,
                target_module_aliases,
            ):
                continue
            new_aliases = targets - aliases
            if new_aliases:
                aliases.update(new_aliases)
                changed = True
    return aliases


def _expression_is_static_target_callable(
    expression: ast.expr,
    target_names: set[str],
    target_module_aliases: set[str],
) -> bool:
    name = _expression_name(expression)
    if name:
        tail = name.rsplit(".", 1)[-1]
        root = name.split(".", 1)[0]
        if name in target_names or tail in target_names:
            return isinstance(expression, ast.Name) or root in target_module_aliases
    if not isinstance(expression, ast.Call):
        return False
    if _expression_name(expression.func) != "getattr" or len(expression.args) < 2:
        return False
    module_name = _expression_name(expression.args[0]).split(".", 1)[0]
    attribute = expression.args[1]
    return (
        module_name in target_module_aliases
        and isinstance(attribute, ast.Constant)
        and isinstance(attribute.value, str)
        and attribute.value in target_names
    )


def _target_import_context(
    tree: ast.Module,
    target_names: set[str],
    target_modules: set[str],
) -> tuple[set[str], set[str]]:
    module_aliases = set(target_modules)
    callable_aliases = set(target_names)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_matches(alias.name, target_modules):
                    module_aliases.add(alias.asname or alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom) and node.module:
            if not _module_matches(node.module, target_modules):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                callable_aliases.add(alias.asname or alias.name)
    return module_aliases, callable_aliases


def _module_matches(module_name: str, target_modules: set[str]) -> bool:
    return any(
        module_name == candidate
        or module_name.endswith(f".{candidate}")
        or candidate.endswith(f".{module_name}")
        for candidate in target_modules
    )


def _call_matches_target(
    call: ast.Call,
    target_names: set[str],
    target_module_aliases: set[str],
) -> bool:
    name = _expression_name(call.func)
    if not name:
        return False
    if name in target_names or name.rsplit(".", 1)[-1] in target_names:
        return True
    root = name.split(".", 1)[0]
    return root in target_module_aliases


def _observable_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    target_calls: list[ast.Call],
    target_names: set[str],
    target_module_aliases: set[str],
) -> set[str]:
    target_call_ids = {id(call) for call in target_calls}
    observable = {
        name
        for call in target_calls
        for expression in [*call.args, *(keyword.value for keyword in call.keywords)]
        for name in _loaded_names(expression)
    }
    first_target_line = min(getattr(call, "lineno", 0) for call in target_calls)
    assignments = sorted(
        (
            (node, value, targets)
            for node in ast.walk(function)
            for value, targets in [_assignment_value_and_targets(node)]
            if value is not None and targets
        ),
        key=lambda item: getattr(item[0], "lineno", 0),
    )
    changed = True
    while changed:
        changed = False
        for node, value, targets in assignments:
            derives_from_target = any(
                id(call) in target_call_ids
                or _call_matches_target(call, target_names, target_module_aliases)
                for call in ast.walk(value)
                if isinstance(call, ast.Call)
            )
            derives_from_observable = bool(_loaded_names(value) & observable)
            observes_side_effect = (
                getattr(node, "lineno", 0) >= first_target_line
                and _contains_observer_call(value)
            )
            if not (
                derives_from_target
                or derives_from_observable
                or observes_side_effect
            ):
                continue
            new_names = targets - observable
            if new_names:
                observable.update(new_names)
                changed = True
    return observable


def _assignment_value_and_targets(
    node: ast.AST,
) -> tuple[ast.expr | None, set[str]]:
    if isinstance(node, ast.Assign):
        return node.value, {
            name
            for target in node.targets
            for name in _assigned_names(target)
        }
    if isinstance(node, ast.AnnAssign):
        return node.value, _assigned_names(node.target)
    if isinstance(node, ast.NamedExpr):
        return node.value, _assigned_names(node.target)
    return None, set()


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _contains_observer_call(node: ast.AST) -> bool:
    return any(
        _expression_name(call.func).rsplit(".", 1)[-1] in _OBSERVER_CALL_NAMES
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    )


def _assertion_observes_target(
    assertion: ast.Assert,
    target_calls: list[ast.Call],
    observable_names: set[str],
    target_names: set[str],
    target_module_aliases: set[str],
) -> bool:
    if any(
        _call_matches_target(call, target_names, target_module_aliases)
        for call in ast.walk(assertion.test)
        if isinstance(call, ast.Call)
    ):
        return True
    if _loaded_names(assertion.test) & observable_names:
        return True
    if _expression_references_target_module(assertion.test, target_module_aliases):
        return True
    first_target_line = min(getattr(call, "lineno", 0) for call in target_calls)
    return (
        getattr(assertion, "lineno", 0) >= first_target_line
        and _contains_observer_call(assertion.test)
    )


def _expression_references_target_module(
    node: ast.AST,
    target_module_aliases: set[str],
) -> bool:
    return any(
        _expression_name(child).split(".", 1)[0] in target_module_aliases
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
    )


def _target_exception_assertion_records(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    target_names: set[str],
    target_module_aliases: set[str],
    target_contract_declared: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for node in ast.walk(function):
        if isinstance(node, (ast.With, ast.AsyncWith)) and _is_pytest_raises_context(node):
            if _statements_contain_target_call(
                node.body,
                target_names,
                target_module_aliases,
                target_contract_declared,
            ):
                records.append(
                    {
                        "line": getattr(node, "lineno", 0),
                        "kind": "expected_exception",
                        "expression": ast.unparse(node.items[0].context_expr),
                    }
                )
            continue
        if not isinstance(node, ast.Try):
            continue
        handled = {
            name
            for handler in node.handlers
            for name in _exception_names(handler.type)
        }
        if not handled & {"ValueError", "TypeError", "SystemExit"}:
            continue
        if not _statements_contain_target_call(
            node.body,
            target_names,
            target_module_aliases,
            target_contract_declared,
        ):
            continue
        if _has_explicit_try_exception_assertion(function):
            records.append(
                {
                    "line": getattr(node, "lineno", 0),
                    "kind": "expected_exception",
                    "expression": "try/except rejects invalid input",
                }
            )
    return records


def _statements_contain_target_call(
    statements: list[ast.stmt],
    target_names: set[str],
    target_module_aliases: set[str],
    target_contract_declared: bool,
) -> bool:
    calls = [
        call
        for statement in statements
        for call in ast.walk(statement)
        if isinstance(call, ast.Call)
    ]
    if not target_contract_declared:
        return bool(calls)
    return any(
        _call_matches_target(call, target_names, target_module_aliases)
        for call in calls
    )


def _is_semantic_assertion(node: ast.Assert) -> bool:
    if _is_tautological_assertion(node):
        return False
    test = node.test
    if isinstance(test, ast.Constant):
        return test.value is not True
    if isinstance(test, ast.Call):
        function_name = _expression_name(test.func)
        if function_name in {"callable", "hasattr", "isinstance", "issubclass"}:
            return False
    if isinstance(test, ast.Compare):
        values = [test.left, *test.comparators]
        if any(
            isinstance(value, ast.Call)
            and _expression_name(value.func)
            in {"callable", "hasattr", "isinstance", "issubclass"}
            for value in values
        ):
            return False
        if any(isinstance(value, ast.Name) and value.id == "target" for value in values):
            return False
    return True


def _is_tautological_assertion(node: ast.Assert) -> bool:
    return _expression_is_tautological(node.test)


def _expression_is_tautological(node: ast.expr) -> bool:
    literal_truth = _literal_truthiness(node)
    if literal_truth is not None:
        return literal_truth
    comparison = _single_comparison(node)
    if comparison is not None:
        left, operator, right = comparison
        if isinstance(operator, (ast.Eq, ast.Is)) and _same_expression(left, right):
            return True
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
        return False
    expressions = _flatten_boolean_or(node)
    if any(_expression_is_tautological(expression) for expression in expressions):
        return True
    return any(
        _expressions_are_complements(left, right)
        for index, left in enumerate(expressions)
        for right in expressions[index + 1 :]
    )


def _flatten_boolean_or(node: ast.expr) -> list[ast.expr]:
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
        return [node]
    return [
        expression
        for value in node.values
        for expression in _flatten_boolean_or(value)
    ]


def _expressions_are_complements(left: ast.expr, right: ast.expr) -> bool:
    if isinstance(left, ast.UnaryOp) and isinstance(left.op, ast.Not):
        return _same_expression(left.operand, right)
    if isinstance(right, ast.UnaryOp) and isinstance(right.op, ast.Not):
        return _same_expression(right.operand, left)
    left_comparison = _single_comparison(left)
    right_comparison = _single_comparison(right)
    if left_comparison is None or right_comparison is None:
        return False
    left_operand, left_operator, left_comparator = left_comparison
    right_operand, right_operator, right_comparator = right_comparison
    complementary_operators = (
        (ast.Eq, ast.NotEq),
        (ast.NotEq, ast.Eq),
        (ast.Is, ast.IsNot),
        (ast.IsNot, ast.Is),
        (ast.In, ast.NotIn),
        (ast.NotIn, ast.In),
    )
    return (
        any(
            isinstance(left_operator, left_type)
            and isinstance(right_operator, right_type)
            for left_type, right_type in complementary_operators
        )
        and _same_expression(left_operand, right_operand)
        and _same_expression(left_comparator, right_comparator)
    )


def _single_comparison(
    node: ast.expr,
) -> tuple[ast.expr, ast.cmpop, ast.expr] | None:
    if (
        not isinstance(node, ast.Compare)
        or len(node.ops) != 1
        or len(node.comparators) != 1
    ):
        return None
    return node.left, node.ops[0], node.comparators[0]


def _same_expression(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right,
        include_attributes=False,
    )


def _literal_truthiness(node: ast.expr) -> bool | None:
    try:
        return bool(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _is_pytest_raises_context(node: ast.With | ast.AsyncWith) -> bool:
    return any(
        isinstance(item.context_expr, ast.Call)
        and _expression_name(item.context_expr.func) in {"pytest.raises", "raises"}
        for item in node.items
    )


def _has_explicit_try_exception_assertion(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    expected_exceptions = {"ValueError", "TypeError", "SystemExit"}
    catches_expected = any(
        expected_exceptions & set(_exception_names(handler.type))
        for node in ast.walk(function)
        if isinstance(node, ast.Try)
        for handler in node.handlers
    )
    raises_failure = any(
        isinstance(node, ast.Raise)
        and node.exc is not None
        and (
            _expression_name(node.exc.func) == "AssertionError"
            if isinstance(node.exc, ast.Call)
            else _expression_name(node.exc) == "AssertionError"
        )
        for node in ast.walk(function)
    )
    return catches_expected and raises_failure


def _expression_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expression_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _assigned_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    }


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
