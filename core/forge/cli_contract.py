from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from core.forge.contracts import FeasiblePlan, PlanInterface


def cli_invocation_contract_failures(
    file_contents: Mapping[str, str],
    plan: FeasiblePlan,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for interface in plan.interfaces:
        if (
            interface.interface_type != "cli_entrypoint"
            or not getattr(interface, "explicit_argv_excludes_program_name", False)
        ):
            continue
        source_path = _interface_source_path(interface, plan, file_contents)
        source = file_contents.get(source_path, "") if source_path else ""
        if source:
            failures.extend(_source_failures(source_path, source, interface))
        failures.extend(_test_failures(file_contents, interface))
    return _deduplicate_failures(failures)


def _interface_source_path(
    interface: PlanInterface,
    plan: FeasiblePlan,
    file_contents: Mapping[str, str],
) -> str:
    if interface.module_path:
        return f"src/{interface.module_path.replace('.', '/')}.py"
    entrypoint_path = plan.implementation_blueprint.entrypoint_path
    if entrypoint_path in file_contents:
        return entrypoint_path
    for path in sorted(file_contents):
        if not path.startswith("src/") or not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(file_contents[path])
        except SyntaxError:
            continue
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == interface.name
            for node in tree.body
        ):
            return path
    return ""


def _source_failures(
    path: str,
    source: str,
    interface: PlanInterface,
) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == interface.name
        ),
        None,
    )
    if function is None or not function.args.args:
        return []
    argv_name = function.args.args[0].arg
    failures: list[dict[str, Any]] = []
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is not None and any(
                isinstance(target, ast.Name) and target.id == argv_name
                for target in targets
            ) and _is_unsliced_sys_argv(value):
                failures.append(
                    _failure(
                        path,
                        interface,
                        "explicit_argv_uses_full_sys_argv",
                        node.lineno,
                    )
                )
            if (
                value is not None
                and isinstance(value, ast.Name)
                and value.id == argv_name
                and interface.explicit_argv_count is not None
            ):
                for target in targets:
                    if not isinstance(target, (ast.Tuple, ast.List)):
                        continue
                    observed = len(target.elts)
                    if observed == interface.explicit_argv_count + 1:
                        failures.append(
                            _failure(
                                path,
                                interface,
                                "explicit_argv_arity_mismatch",
                                node.lineno,
                                observed_argv_count=observed,
                            )
                        )
        if interface.explicit_argv_count is None or not isinstance(node, ast.Compare):
            continue
        observed = _compared_argv_length(node, argv_name)
        if observed == interface.explicit_argv_count + 1:
            failures.append(
                _failure(
                    path,
                    interface,
                    "explicit_argv_arity_mismatch",
                    node.lineno,
                    observed_argv_count=observed,
                )
            )
    return failures


def _test_failures(
    file_contents: Mapping[str, str],
    interface: PlanInterface,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for path, source in sorted(file_contents.items()):
        if not path.startswith("tests/") or not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for function in (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ):
            sequences = _assigned_sequences(function)
            for node in ast.walk(function):
                if (
                    not isinstance(node, ast.Call)
                    or not node.args
                    or _call_name(node.func) != interface.name
                ):
                    continue
                sequence = _sequence_value(node.args[0], sequences)
                if not sequence or not _looks_like_program_name(sequence[0], interface):
                    continue
                if (
                    interface.explicit_argv_count is None
                    or len(sequence) == interface.explicit_argv_count + 1
                ):
                    failures.append(
                        _failure(
                            path,
                            interface,
                            "explicit_argv_includes_program_name",
                            node.lineno,
                            observed_argv_count=len(sequence),
                        )
                    )
    return failures


def _failure(
    path: str,
    interface: PlanInterface,
    reason: str,
    line: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "path": path,
        "kind": "cli_invocation_contract_failure",
        "reason": reason,
        "interface": interface.name,
        "line": line,
        "explicit_argv_excludes_program_name": True,
        "expected_argv_count": interface.explicit_argv_count,
        **extra,
    }


def _is_unsliced_sys_argv(node: ast.expr) -> bool:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr == "argv"
    ):
        return True
    if isinstance(node, ast.IfExp):
        return _is_unsliced_sys_argv(node.body) or _is_unsliced_sys_argv(node.orelse)
    return (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and _is_unsliced_sys_argv(node.args[0])
    )


def _compared_argv_length(node: ast.Compare, argv_name: str) -> int | None:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    pairs = ((node.left, node.comparators[0]), (node.comparators[0], node.left))
    for possible_len, possible_count in pairs:
        if (
            isinstance(possible_len, ast.Call)
            and isinstance(possible_len.func, ast.Name)
            and possible_len.func.id == "len"
            and len(possible_len.args) == 1
            and isinstance(possible_len.args[0], ast.Name)
            and possible_len.args[0].id == argv_name
            and isinstance(possible_count, ast.Constant)
            and isinstance(possible_count.value, int)
        ):
            return possible_count.value
    return None


def _assigned_sequences(function: ast.AST) -> dict[str, list[object]]:
    sequences: dict[str, list[object]] = {}
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        values = _literal_sequence(node.value)
        if values is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                sequences[target.id] = values
    return sequences


def _sequence_value(
    node: ast.expr,
    assigned: Mapping[str, list[object]],
) -> list[object] | None:
    if isinstance(node, ast.Name):
        return assigned.get(node.id)
    return _literal_sequence(node)


def _literal_sequence(node: ast.expr | None) -> list[object] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    return [
        item.value if isinstance(item, ast.Constant) else None
        for item in node.elts
    ]


def _looks_like_program_name(value: object, interface: PlanInterface) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower().replace("\\", "/").rsplit("/", 1)[-1]
    module_leaf = interface.module_path.rsplit(".", 1)[-1].lower()
    return lowered in {
        "prog",
        "program",
        "cli",
        "cli.py",
        interface.name.lower(),
        f"{interface.name.lower()}.py",
        module_leaf,
        f"{module_leaf}.py",
    } or lowered.endswith(".exe")


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _deduplicate_failures(
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for failure in failures:
        key = (
            failure.get("path"),
            failure.get("reason"),
            failure.get("line"),
            failure.get("interface"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(failure)
    return unique
