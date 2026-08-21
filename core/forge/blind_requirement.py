import ast
import re
from typing import Any

from core.constraint_witnesses import finite_witness_contradictions


def requirement_preflight_error(
    requirement: str,
    expected_terminal_status: str,
) -> str | None:
    example_error = _same_length_example_error(requirement)
    if example_error is not None and expected_terminal_status != "infeasible_proven":
        return (
            f"expected {expected_terminal_status} requirement has a behavioral "
            f"example contradiction: {example_error}"
        )
    contradictions = finite_witness_contradictions(requirement)
    if not contradictions or expected_terminal_status == "infeasible_proven":
        return None
    return (
        f"expected {expected_terminal_status} requirement has a finite witness "
        f"contradiction: {contradictions[0].message}"
    )


def requirement_preflight_failure_class(error: str) -> str:
    if "behavioral example contradiction" in error:
        return "requirement_behavioral_example"
    return "requirement_finite_witness"


def _same_length_example_error(requirement: str) -> str | None:
    if re.search(r"\bsame\s+length\b", requirement, re.IGNORECASE) is None:
        return None
    literals = _list_literals(requirement)
    for match in re.finditer(r"\breturns?\b", requirement, re.IGNORECASE):
        preceding = [
            item
            for item in literals
            if item[1] <= match.start() and match.start() - item[1] <= 180
        ]
        following = [
            item
            for item in literals
            if item[0] >= match.end() and item[0] - match.end() <= 180
        ]
        if not preceding or not following:
            continue
        source = max(preceding, key=lambda item: item[1])
        result = min(following, key=lambda item: item[0])
        if len(source[2]) != len(result[2]):
            return (
                f"a {len(source[2])}-item input returns a {len(result[2])}-item "
                "list despite the same-length contract"
            )
    return None


def _list_literals(text: str) -> list[tuple[int, int, list[Any]]]:
    literals: list[tuple[int, int, list[Any]]] = []
    for start, character in enumerate(text):
        if character != "[":
            continue
        end = _balanced_list_end(text, start)
        if end is None:
            continue
        try:
            value = ast.literal_eval(text[start:end])
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, list):
            literals.append((start, end, value))
    return literals


def _balanced_list_end(text: str, start: int) -> int | None:
    depth = 0
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = ""
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def requirement_review_error(review: dict[str, Any]) -> str | None:
    approved = review.get("approved")
    findings = review.get("findings")
    if not isinstance(approved, bool):
        return "independent requirement review omitted a boolean approval"
    if not isinstance(findings, list) or not all(
        isinstance(item, str) and item.strip() for item in findings
    ):
        return "independent requirement review returned invalid findings"
    if approved and findings:
        return "independent requirement review was internally inconsistent"
    if approved:
        return None
    detail = "; ".join(_bounded_finding(item) for item in findings[:4])
    return "independent requirement review rejected the case set" + (
        f": {detail}" if detail else ""
    )


def requirement_reviewer_instructions() -> str:
    return """You are an independent blind-benchmark requirement reviewer. You receive only candidate requirement definitions and no
Forge source, generated implementation, prior benchmark case, or oracle. Approve only if every expected verified case defines an exact
public Python module and interface, deterministic observable behavior, representative edge cases, and explicit failure behavior without
contradictions or unspecified algorithms. An expected validation_failed case must remain logically satisfiable in principle but contain a
material ambiguity or universal claim that prevents objective certification as written. An expected infeasible_proven case must contain a
precise mathematical or finite constraint contradiction independent of platform limitations, dependencies, or implementation difficulty.
Reject incorrect terminal labels, overlapping or repeated tasks, hidden implementation hints, underspecified verified contracts, and
requirements whose stated examples conflict with their rules. Do not reject a correctly labeled ambiguous or contradictory case merely
because it has the intended property, and never include confirming observations as findings. Findings must identify only actual defects,
must identify the candidate index, and must remain empty when approved. Reject any verified CLI without an importable
main(argv: list[str] | None = None) -> int interface that can be tested in-process, and reject any verified service contract that requires
a live server, socket, subprocess, or HTTP client instead of a callable module interface. For universal character transformations with
fixed output length, check finite witnesses whose case mapping expands to multiple code points. Return only the requested structured object."""


def _bounded_finding(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= 240 else normalized[:237] + "..."
