import ast
import re
from dataclasses import asdict, dataclass
from typing import Any


_CLI_NAME_PATTERNS = (
    re.compile(
        r"\bcli\s+(?:utility|tool|application|command)\s+"
        r"(?:named|called)\s+[`'\"]?([A-Za-z][\w.-]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcommand(?:-line)?\s+(?:utility|tool|application)\s+"
        r"(?:named|called)\s+[`'\"]?([A-Za-z][\w.-]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bexecutable\s+(?:named|called)\s+[`'\"]?([A-Za-z][\w.-]*)",
        re.IGNORECASE,
    ),
)

_ARGV0_ALLOWANCE_PATTERNS = (
    re.compile(r"\bargv\s*\[\s*0\s*\]", re.IGNORECASE),
    re.compile(r"\bfull\s+sys\.argv\b", re.IGNORECASE),
    re.compile(r"\bargv0\b", re.IGNORECASE),
    re.compile(
        r"\bargv\b.{0,40}\b(?:includes?|contains?)\b.{0,30}"
        r"\b(?:program|executable|command)(?:\s+name)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bincluding\b.{0,30}\b(?:program|executable|command)\s+name\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class OracleContractMismatch:
    contract_id: str
    function: str
    call_line: int
    declared_cli_name: str
    argument_name: str
    first_argument: str
    message: str

    def to_evidence(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OraclePatternMismatch:
    contract_id: str
    function: str
    fixture_name: str
    fixture_line: int
    declared_pattern: str
    sample: str
    oracle_classification: str
    derived_classification: str
    message: str

    def to_evidence(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OracleHarnessMismatch:
    contract_id: str
    function: str
    class_name: str
    context_line: int
    enter_line: int
    bound_name: str
    message: str

    def to_evidence(self) -> dict[str, Any]:
        return asdict(self)


def oracle_contract_mismatches(
    source: str,
    requirement: str,
) -> list[OracleContractMismatch | OraclePatternMismatch | OracleHarnessMismatch]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    mismatches: list[
        OracleContractMismatch | OraclePatternMismatch | OracleHarnessMismatch
    ] = [
        *explicit_pattern_fixture_mismatches(source, requirement, tree=tree),
        *context_manager_binding_mismatches(tree),
    ]
    cli_name = _declared_cli_name(requirement)
    if (
        cli_name is None
        or re.search(r"\bmain\s*\(\s*argv\b", requirement, re.IGNORECASE) is None
        or _requirement_allows_argv0(requirement)
    ):
        return mismatches

    direct_main_names, module_aliases = _main_import_context(tree, cli_name)
    if not direct_main_names and not module_aliases:
        return mismatches

    for function in (
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ):
        sequence_first_values = _sequence_first_values(function)
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not _is_main_call(node.func, direct_main_names, module_aliases):
                continue
            first_value = _first_sequence_value(
                node.args[0], sequence_first_values, node.lineno
            )
            if first_value != cli_name:
                continue
            argument_name = (
                node.args[0].id if isinstance(node.args[0], ast.Name) else "<literal>"
            )
            mismatches.append(
                OracleContractMismatch(
                    contract_id="in_process_main_argv",
                    function=function.name,
                    call_line=node.lineno,
                    declared_cli_name=cli_name,
                    argument_name=argument_name,
                    first_argument=first_value,
                    message=(
                        "oracle injects the declared CLI name as argv[0], but the "
                        "requirement does not define main(argv) as full sys.argv"
                    ),
                )
            )
    return mismatches


def context_manager_binding_mismatches(
    tree: ast.Module,
) -> list[OracleHarnessMismatch]:
    invalid_context_managers: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for class_node in (
        node for node in tree.body if isinstance(node, ast.ClassDef)
    ):
        enter = next(
            (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in {"__enter__", "__aenter__"}
            ),
            None,
        )
        if enter is not None and not _returns_non_none_value(enter):
            invalid_context_managers[class_node.name] = enter

    mismatches: list[OracleHarnessMismatch] = []
    if not invalid_context_managers:
        return mismatches
    for function in (
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ):
        for node in ast.walk(function):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            for item in node.items:
                call = item.context_expr
                bound = item.optional_vars
                if (
                    not isinstance(call, ast.Call)
                    or not isinstance(call.func, ast.Name)
                    or call.func.id not in invalid_context_managers
                    or not isinstance(bound, ast.Name)
                    or not _bound_attribute_is_used(function, bound.id)
                ):
                    continue
                enter = invalid_context_managers[call.func.id]
                mismatches.append(
                    OracleHarnessMismatch(
                        contract_id="context_manager_binding",
                        function=function.name,
                        class_name=call.func.id,
                        context_line=node.lineno,
                        enter_line=enter.lineno,
                        bound_name=bound.id,
                        message=(
                            f"{call.func.id}.__enter__ does not return the object "
                            f"bound as {bound.id}, but the oracle dereferences it"
                        ),
                    )
                )
    return mismatches


def _returns_non_none_value(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return any(
        _statement_returns_non_none_value(statement) for statement in function.body
    )


def _statement_returns_non_none_value(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Return):
        return statement.value is not None and not (
            isinstance(statement.value, ast.Constant) and statement.value.value is None
        )
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    return any(
        _statement_returns_non_none_value(child)
        for child in ast.iter_child_nodes(statement)
        if isinstance(child, ast.stmt)
    )


def _bound_attribute_is_used(function: ast.AST, bound_name: str) -> bool:
    return any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == bound_name
        for node in ast.walk(function)
    )


def explicit_pattern_fixture_mismatches(
    source: str,
    requirement: str,
    *,
    tree: ast.Module | None = None,
) -> list[OraclePatternMismatch]:
    declared = _declared_explicit_pattern(requirement)
    if declared is None:
        return []
    try:
        compiled = re.compile(declared)
    except re.error:
        return []
    if tree is None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

    parent_by_node = {
        child: node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }
    mismatches: list[OraclePatternMismatch] = []
    for node in ast.walk(tree):
        assignment = _named_assignment(node)
        if assignment is None:
            continue
        fixture_name, expression = assignment
        oracle_classification = _fixture_classification(fixture_name)
        if oracle_classification is None:
            continue
        for sample in _literal_fixture_samples(expression):
            body = _without_one_line_ending(sample)
            derived_classification = (
                "valid" if compiled.fullmatch(body) is not None else "invalid"
            )
            if derived_classification == oracle_classification:
                continue
            mismatches.append(
                OraclePatternMismatch(
                    contract_id="explicit_regex_fixture",
                    function=_containing_function(node, parent_by_node),
                    fixture_name=fixture_name,
                    fixture_line=getattr(node, "lineno", 0),
                    declared_pattern=declared,
                    sample=sample,
                    oracle_classification=oracle_classification,
                    derived_classification=derived_classification,
                    message=(
                        f"fixture {fixture_name} classifies the sample as "
                        f"{oracle_classification}, but the requirement's explicit "
                        f"pattern classifies it as {derived_classification}"
                    ),
                )
            )
    return mismatches


def _declared_explicit_pattern(requirement: str) -> str | None:
    patterns = (
        re.compile(
            r"\b(?:regex|regular\s+expression)\s*:\s*([`'\"])(.+?)\1",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\bmatches?\s+(?:the\s+)?pattern\s*([`'\"])(.+?)\1",
            re.IGNORECASE | re.DOTALL,
        ),
    )
    for pattern in patterns:
        match = pattern.search(requirement)
        if match:
            return match.group(2)
    return None


def _named_assignment(node: ast.AST) -> tuple[str, ast.expr] | None:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
        return node.targets[0].id, node.value
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ):
        return node.target.id, node.value
    return None


def _fixture_classification(name: str) -> str | None:
    normalized = name.casefold()
    if re.search(r"(?:^|_)(?:invalid|bad)(?:_|$)", normalized):
        return "invalid"
    if re.search(r"(?:^|_)(?:valid|good)(?:_|$)", normalized):
        return "valid"
    return None


def _literal_fixture_samples(expression: ast.expr) -> list[str]:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value.splitlines(keepends=True) or [expression.value]
    if not isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        return []
    samples: list[str] = []
    for item in expression.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            continue
        samples.extend(item.value.splitlines(keepends=True) or [item.value])
    return samples


def _without_one_line_ending(value: str) -> str:
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith(("\n", "\r")):
        return value[:-1]
    return value


def _containing_function(
    node: ast.AST,
    parent_by_node: dict[ast.AST, ast.AST],
) -> str:
    current = node
    while current in parent_by_node:
        current = parent_by_node[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return "<module>"


def _declared_cli_name(requirement: str) -> str | None:
    for pattern in _CLI_NAME_PATTERNS:
        match = pattern.search(requirement)
        if match:
            return match.group(1)
    return None


def _requirement_allows_argv0(requirement: str) -> bool:
    return any(pattern.search(requirement) for pattern in _ARGV0_ALLOWANCE_PATTERNS)


def _main_import_context(tree: ast.Module, cli_name: str) -> tuple[set[str], set[str]]:
    direct_main_names: set[str] = set()
    module_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not _module_matches_cli(module, cli_name):
                continue
            direct_main_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "main"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _module_matches_cli(alias.name, cli_name):
                    module_aliases.add(alias.asname or alias.name.split(".", 1)[0])
    return direct_main_names, module_aliases


def _module_matches_cli(module: str, cli_name: str) -> bool:
    expected = cli_name.replace("-", "_").casefold()
    parts = [part.casefold() for part in module.split(".") if part]
    return bool(parts) and (parts[0] == expected or parts[-1] == expected)


def _sequence_first_values(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, list[tuple[int, str]]]:
    values: dict[str, list[tuple[int, str]]] = {}
    assignments = sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda node: node.lineno,
    )
    for assignment in assignments:
        if isinstance(assignment, ast.Assign):
            if len(assignment.targets) != 1 or not isinstance(
                assignment.targets[0], ast.Name
            ):
                continue
            name = assignment.targets[0].id
            expression = assignment.value
        else:
            if not isinstance(assignment.target, ast.Name) or assignment.value is None:
                continue
            name = assignment.target.id
            expression = assignment.value
        first = _literal_sequence_first(expression)
        if first is not None:
            values.setdefault(name, []).append((assignment.lineno, first))
    return values


def _first_sequence_value(
    expression: ast.expr,
    values: dict[str, list[tuple[int, str]]],
    call_line: int,
) -> str | None:
    if isinstance(expression, ast.Name):
        prior_values = values.get(expression.id, [])
        return next(
            (value for line, value in reversed(prior_values) if line < call_line),
            None,
        )
    return _literal_sequence_first(expression)


def _literal_sequence_first(expression: ast.expr) -> str | None:
    if not isinstance(expression, (ast.List, ast.Tuple)) or not expression.elts:
        return None
    first = expression.elts[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _is_main_call(
    expression: ast.expr,
    direct_main_names: set[str],
    module_aliases: set[str],
) -> bool:
    if isinstance(expression, ast.Name):
        return expression.id in direct_main_names
    return (
        isinstance(expression, ast.Attribute)
        and expression.attr == "main"
        and isinstance(expression.value, ast.Name)
        and expression.value.id in module_aliases
    )
