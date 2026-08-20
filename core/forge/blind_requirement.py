from typing import Any

from core.constraint_witnesses import finite_witness_contradictions


def requirement_preflight_error(
    requirement: str,
    expected_terminal_status: str,
) -> str | None:
    contradictions = finite_witness_contradictions(requirement)
    if not contradictions or expected_terminal_status == "infeasible_proven":
        return None
    return (
        f"expected {expected_terminal_status} requirement has a finite witness "
        f"contradiction: {contradictions[0].message}"
    )


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
