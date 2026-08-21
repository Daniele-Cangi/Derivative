import ast
import re
from dataclasses import asdict, dataclass
from typing import Any


PUBLIC_IMPORT_KINDS = frozenset({"callable", "cli_entrypoint", "function"})
_MODULE_PATTERN = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
_SYMBOL_PATTERN = r"[A-Za-z_]\w*"
_PUBLIC_IMPORT_PATTERN = re.compile(
    rf"\bPublic\s+import\s+contract\s*:\s*from\s+"
    rf"(?P<module>{_MODULE_PATTERN})\s+import\s+"
    rf"(?P<symbol>{_SYMBOL_PATTERN})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PublicImportContract:
    module: str
    symbol: str
    kind: str

    def to_payload(self) -> dict[str, str]:
        return asdict(self)


def load_public_import_contract(
    payload: object,
    *,
    label: str,
    required: bool,
) -> PublicImportContract | None:
    if payload is None:
        if required:
            raise ValueError(f"{label} requires a public_contract object.")
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{label} public_contract must be an object.")

    module = str(payload.get("module", "")).strip()
    symbol = str(payload.get("symbol", "")).strip()
    kind = str(payload.get("kind", "")).strip()
    if re.fullmatch(_MODULE_PATTERN, module) is None:
        raise ValueError(f"{label} public_contract module is not importable: {module!r}.")
    if re.fullmatch(_SYMBOL_PATTERN, symbol) is None:
        raise ValueError(f"{label} public_contract symbol is invalid: {symbol!r}.")
    if kind not in PUBLIC_IMPORT_KINDS:
        raise ValueError(
            f"{label} public_contract kind must be one of "
            f"{sorted(PUBLIC_IMPORT_KINDS)}."
        )
    return PublicImportContract(module=module, symbol=symbol, kind=kind)


def extract_public_import_contract(
    requirement: str,
    *,
    kind: str = "callable",
) -> PublicImportContract | None:
    match = _PUBLIC_IMPORT_PATTERN.search(requirement)
    if match is None:
        return None
    return PublicImportContract(
        module=match.group("module"),
        symbol=match.group("symbol"),
        kind=kind,
    )


def requirement_public_import_error(
    requirement: str,
    contract: PublicImportContract,
) -> str | None:
    declared = extract_public_import_contract(requirement, kind=contract.kind)
    if declared is None:
        return (
            "requirement must declare the exact public import using "
            "'Public import contract: from <module> import <symbol>'"
        )
    if (declared.module, declared.symbol) != (contract.module, contract.symbol):
        return (
            "public_contract does not match the requirement declaration: "
            f"payload=from {contract.module} import {contract.symbol}, "
            f"requirement=from {declared.module} import {declared.symbol}"
        )
    return None


def oracle_public_import_error(
    source: str,
    contract: PublicImportContract,
) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"oracle syntax error at line {exc.lineno}: {exc.msg}"

    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 0:
            continue
        if node.module != contract.module:
            continue
        if any(alias.name == contract.symbol for alias in node.names):
            return None
    return (
        "oracle must import the declared public target exactly with "
        f"'from {contract.module} import {contract.symbol}'"
    )


def public_import_contract_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "module": {"type": "string"},
            "symbol": {"type": "string"},
            "kind": {
                "type": "string",
                "enum": sorted(PUBLIC_IMPORT_KINDS),
            },
        },
        "required": ["module", "symbol", "kind"],
        "additionalProperties": False,
    }
