"""Numeric obligations for the existing contract-expiration CLI domain."""
import re
from typing import Any, Iterable

from core.forge.contracts import BuildSpec, RequirementAtom
from core.forge.requirement_text import mask_quoted_literals, prose_mention_is_negated


# Detection is deliberately broader than extraction: a nearby numeric duration
# and expiration predicate are enough to require interpretation, not to prove a
# value or comparison. Keep unsupported policies on the existing unresolved path.
_EXPIRATION = r"\b(?:expires?|expired|expiring|expiration)\b"
_DURATION = r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?:\s+|-)days?\b"
# An "or" inside a comparison does not introduce a separate property.
_BRIDGE = (
    r"(?:(?![.,;!?]|\b(?:and|but|however|whereas|then)\b|"
    r"\bor\b(?!\s+(?:before|after|below|above|equal)\b)).){0,100}?"
)
_NUMERIC_EXPIRATION = re.compile(
    rf"(?:{_EXPIRATION}{_BRIDGE}{_DURATION}|{_DURATION}{_BRIDGE}{_EXPIRATION})",
    re.IGNORECASE,
)


def _has_numeric_expiration_policy(text: str) -> bool:
    # Negative-looking comparatives are not rejected obligations.
    # Mask it only for polarity analysis, preserving recognizer offsets.
    polarity_text = re.sub(
        r"\b(?:no\s+(?:more|less|fewer)\s+than|does\s+not\s+exceed)\b",
        lambda match: " " * len(match.group()), text, flags=re.IGNORECASE,
    )
    return any(
        not prose_mention_is_negated(polarity_text, match.start(), match.end())
        and not prose_mention_is_negated(polarity_text, match.end(), match.end())
        for match in _NUMERIC_EXPIRATION.finditer(text)
    )


def compile_expiration_horizon(atoms: Iterable[RequirementAtom]) -> dict[str, Any] | None:
    matches = []
    unresolved_ids = []
    requirement_ids = []
    for atom in atoms:
        if atom.strength not in {"hard", "universal"}:
            continue
        text = mask_quoted_literals(atom.text)
        match = re.match(
            r"^flags?\s+(?:(?:a|the)\s+)?contracts?\s+"
            r"(?:expiring|(?:that|which)\s+expires?)\s+in\s+less\s+than\s+(-?\d+)\s+days?\b",
            text, re.IGNORECASE,
        )
        if match:
            matches.append((int(match.group(1)), atom.requirement_id))
            requirement_ids.append(atom.requirement_id)
        elif _has_numeric_expiration_policy(text):
            unresolved_ids.append(atom.requirement_id)
            requirement_ids.append(atom.requirement_id)
    if not matches and not unresolved_ids:
        return None
    values = {value for value, _ in matches}
    contract = {
        "observable_field": "is_expiring_within_horizon",
        "comparison_relation": None if unresolved_ids else "less_than",
        "threshold_days": next(iter(values)) if len(values) == 1 and not unresolved_ids else None,
        "requirement_ids": requirement_ids,
    }
    if unresolved_ids:
        contract["unresolved_reason"] = "unsupported_numeric_expiration_wording"
    return contract


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
