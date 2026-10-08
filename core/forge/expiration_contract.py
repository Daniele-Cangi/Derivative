"""Numeric obligations for the existing contract-expiration CLI domain."""
import re
from typing import Any, Iterable

from core.forge.contracts import BuildSpec, RequirementAtom
from core.forge.requirement_text import mask_quoted_literals


def compile_expiration_horizon(atoms: Iterable[RequirementAtom]) -> dict[str, Any] | None:
    matches = []
    for atom in atoms:
        match = re.match(
            r"^flags?\s+(?:(?:a|the)\s+)?contracts?\s+"
            r"(?:expiring|(?:that|which)\s+expires?)\s+in\s+less\s+than\s+(-?\d+)\s+days?\b",
            mask_quoted_literals(atom.text), re.IGNORECASE,
        )
        if match and atom.strength in {"hard", "universal"}:
            matches.append((int(match.group(1)), atom.requirement_id))
    if not matches:
        return None
    values = {value for value, _ in matches}
    return {
        "observable_field": "is_expiring_within_horizon",
        "comparison_relation": "less_than",
        "threshold_days": next(iter(values)) if len(values) == 1 else None,
        "requirement_ids": [requirement_id for _, requirement_id in matches],
    }


def expiration_horizon_contract(spec: BuildSpec) -> dict[str, Any] | None:
    contract = spec.obligation_contract
    return contract.context.get("expiration_horizon") if contract is not None else None


def expiration_horizon_days(spec: BuildSpec) -> int | None:
    contract = expiration_horizon_contract(spec)
    if contract is None:
        return 90
    value = contract.get("threshold_days")
    if value is None:
        # Retain unresolved policy through planning/generation; validation owns
        # the material-ambiguity failure. Never substitute the legacy default.
        return None
    if type(value) is not int:
        raise ValueError("The expiration horizon is materially unspecified.")
    return value
