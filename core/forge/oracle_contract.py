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


def oracle_contract_mismatches(
    source: str,
    requirement: str,
) -> list[OracleContractMismatch]:
    cli_name = _declared_cli_name(requirement)
    if (
        cli_name is None
        or re.search(r"\bmain\s*\(\s*argv\b", requirement, re.IGNORECASE) is None
        or _requirement_allows_argv0(requirement)
    ):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    direct_main_names, module_aliases = _main_import_context(tree, cli_name)
    if not direct_main_names and not module_aliases:
        return []

    mismatches: list[OracleContractMismatch] = []
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
