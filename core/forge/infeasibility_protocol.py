"""Bounded, exact certificates for explicitly authored finite output contracts.

This is a benchmark admission checker, not a natural-language formalizer and not
a Forge terminal-status detector. Never infer a formal obligation from prose.
"""

import hashlib
import itertools
import json
import re

from core.forge.public_contract import (
    PublicImportContract,
    requirement_public_import_error,
)


PROTOCOL = "finite-linear-v1"
MARKER = "[BEGIN FINITE-LINEAR-V1]"
END_MARKER = "[END FINITE-LINEAR-V1]"
MAX_ASSIGNMENTS = 4096
MAX_VARIABLES = 8
MAX_CONSTRAINTS = 16
MAX_DOMAIN_WIDTH = 16
MAX_INTEGER = 1_000_000


def _object(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{PROTOCOL}: invalid {label} fields")
    return value


def _integer(value: object) -> bool:
    return type(value) is int and abs(value) <= MAX_INTEGER


def validate_obligation(obligation: object) -> int:
    payload = _object(obligation, {"protocol", "variables", "constraints"}, "obligation")
    if payload["protocol"] != PROTOCOL:
        raise ValueError(f"{PROTOCOL}: unsupported protocol")
    variables = payload["variables"]
    constraints = payload["constraints"]
    if not isinstance(variables, list) or not 1 <= len(variables) <= MAX_VARIABLES:
        raise ValueError(f"{PROTOCOL}: invalid variable count")
    if not isinstance(constraints, list) or not 1 <= len(constraints) <= MAX_CONSTRAINTS:
        raise ValueError(f"{PROTOCOL}: invalid constraint count")
    names: set[str] = set()
    count = 1
    for variable in variables:
        item = _object(variable, {"name", "lower", "upper"}, "variable")
        name, lower, upper = item["name"], item["lower"], item["upper"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name) is None
            or name in names
        ):
            raise ValueError(f"{PROTOCOL}: invalid or duplicate variable name")
        names.add(name)
        if not _integer(lower) or not _integer(upper):
            raise ValueError(f"{PROTOCOL}: bounds must be bounded integers, not booleans")
        if not 1 <= upper - lower + 1 <= MAX_DOMAIN_WIDTH:
            raise ValueError(f"{PROTOCOL}: domains must be nonempty and bounded")
        count *= upper - lower + 1
    if count > MAX_ASSIGNMENTS:
        raise ValueError(f"{PROTOCOL}: assignment budget exceeded")
    for constraint in constraints:
        item = _object(constraint, {"coefficients", "relation", "rhs"}, "constraint")
        coefficients = item["coefficients"]
        if (
            not isinstance(coefficients, list)
            or len(coefficients) != len(variables)
            or not all(_integer(value) for value in coefficients)
        ):
            raise ValueError(f"{PROTOCOL}: invalid coefficient vector")
        if item["relation"] not in ("le", "eq", "ne") or not _integer(item["rhs"]):
            raise ValueError(f"{PROTOCOL}: invalid relation or right-hand side")
    return count


def obligation_schema() -> dict:
    def obj(properties: dict) -> dict:
        return {
            "type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False,
        }

    integer = {"type": "integer", "minimum": -MAX_INTEGER, "maximum": MAX_INTEGER}
    return obj({
        "protocol": {"type": "string", "enum": [PROTOCOL]},
        "variables": {
            "type": "array", "minItems": 1, "maxItems": MAX_VARIABLES,
            "items": obj({
                "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$"},
                "lower": integer, "upper": integer,
            }),
        },
        "constraints": {
            "type": "array", "minItems": 1, "maxItems": MAX_CONSTRAINTS,
            "items": obj({
                "coefficients": {
                    "type": "array", "items": integer,
                    "minItems": 1, "maxItems": MAX_VARIABLES,
                },
                "relation": {"type": "string", "enum": ["le", "eq", "ne"]},
                "rhs": integer,
            }),
        },
    })


def render_obligation(obligation: object, public_contract: PublicImportContract) -> str:
    validate_obligation(obligation)
    if public_contract.kind not in ("function", "callable"):
        raise ValueError(f"{PROTOCOL}: requires a no-argument callable, not a CLI")
    return (
        f"{MARKER}\n"
        "Normative output contract (unconditional; not an example or a hypothetical).\n"
        f"Calling {public_contract.module}.{public_contract.symbol}() with no arguments "
        "must terminate normally and return a dict with exactly the variable names below "
        "as keys and Python int values (bool is excluded). Each value must lie between "
        "its lower and upper bounds, inclusive. All constraints must hold simultaneously. "
        "For each constraint, sum(coefficients[i] * value_of_variable_i) must have the "
        "stated relation to rhs: le means <=, eq means ==, ne means !=. Variable order "
        "is the listed order. Raising, returning None, reporting impossibility, skipping "
        "the call, or satisfying only a subset is not permitted. This obligation is "
        "mandatory regardless of any other prose; no exception or fallback overrides it.\n"
        + json.dumps(obligation, sort_keys=True, separators=(",", ":"))
        + f"\n{END_MARKER}"
    )


def has_protocol_marker(requirement: str) -> bool:
    # Reserve the protocol identifier too, so damaged/partial blocks fail closed.
    return PROTOCOL in requirement.lower()


def bind_obligation(
    requirement: str, obligation: object, public_contract: PublicImportContract,
) -> str:
    if has_protocol_marker(requirement):
        raise ValueError(f"{PROTOCOL}: producer prose must not contain a formal block")
    return requirement.strip() + "\n" + render_obligation(obligation, public_contract)


def _check_binding(
    requirement: str, obligation: object, public_contract: PublicImportContract,
) -> None:
    block = render_obligation(obligation, public_contract)
    if (
        not requirement.endswith("\n" + block)
        or requirement.lower().count(PROTOCOL) != 3
        or requirement_public_import_error(requirement, public_contract) is not None
    ):
        raise ValueError(f"{PROTOCOL}: normative requirement binding mismatch")


def _prove_unsatisfiable(obligation: dict) -> int:
    total = validate_obligation(obligation)
    domains = [range(v["lower"], v["upper"] + 1) for v in obligation["variables"]]
    for values in itertools.product(*domains):
        for constraint in obligation["constraints"]:
            lhs = sum(a * x for a, x in zip(constraint["coefficients"], values))
            rhs, relation = constraint["rhs"], constraint["relation"]
            satisfied = (lhs <= rhs if relation == "le" else
                         lhs == rhs if relation == "eq" else lhs != rhs)
            if not satisfied:
                break
        else:
            raise ValueError(f"{PROTOCOL}: satisfiable obligation; infeasibility unproven")
    return total


def certify_infeasibility(
    requirement: str, obligation: object, public_contract: PublicImportContract,
) -> dict:
    _check_binding(requirement, obligation, public_contract)
    count = _prove_unsatisfiable(obligation)
    return {
        "protocol": PROTOCOL,
        "method": "exhaustive-integer-enumeration",
        "requirement_sha256": hashlib.sha256(requirement.encode("utf-8")).hexdigest(),
        "assignments_checked": count,
    }


def verify_infeasibility_certificate(
    requirement: str, expected_status: str, obligation: object,
    certificate: object, public_contract: PublicImportContract | None,
) -> None:
    if expected_status != "infeasible_proven" or public_contract is None:
        raise ValueError(f"{PROTOCOL}: certificate requires infeasible label and public contract")
    payload = _object(
        certificate,
        {"protocol", "method", "requirement_sha256", "assignments_checked"},
        "certificate",
    )
    if type(payload["assignments_checked"]) is not int:
        raise ValueError(f"{PROTOCOL}: invalid assignment count")
    # Recompute the proof, not merely the digest or the claimed result.
    if payload != certify_infeasibility(requirement, obligation, public_contract):
        raise ValueError(f"{PROTOCOL}: certificate mismatch")
