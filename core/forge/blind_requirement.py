from typing import Any


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
contradictions or unspecified algorithms. Expected validation_failed cases must contain a material ambiguity or universal claim that
prevents objective certification as written. Expected infeasible_proven cases must contain a precise contradiction with a short formal
basis, not merely a difficult implementation. Reject incorrect terminal labels, overlapping or repeated tasks, hidden implementation
hints, underspecified verified contracts, and requirements whose stated examples conflict with their rules. Findings must identify the
candidate index and remain empty when approved. Return only the requested structured object."""


def _bounded_finding(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= 240 else normalized[:237] + "..."
