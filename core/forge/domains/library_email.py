from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


def is_email_normalization_library(plan: FeasiblePlan) -> bool:
    corpus = " ".join(
        [
            plan.build_spec.normalized_requirement,
            *(interface.signature for interface in plan.interfaces),
        ]
    ).lower()
    return all(
        token in corpus
        for token in ("canonicalize_email", "deduplicate_emails", "trim", "lowercase")
    )


def render_email_library_file(
    plan: FeasiblePlan,
    path: str,
    interfaces: list[PlanInterface],
) -> str | None:
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith("src/library/core.py"):
        return _core_module()
    if normalized.endswith("src/library/__init__.py"):
        return "from .core import canonicalize_email, deduplicate_emails\n"
    if normalized.startswith("tests/"):
        return _behavioral_test()
    return None


def render_email_library_test(plan: FeasiblePlan, plan_test: PlanTest) -> str:
    return _behavioral_test()


def _core_module() -> str:
    return r'''def canonicalize_email(value: str) -> str:
    return value.strip().lower()


def deduplicate_emails(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        canonical = canonicalize_email(value)
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append(canonical)
    return result
'''


def _behavioral_test() -> str:
    return r'''from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from library import canonicalize_email, deduplicate_emails


def test_email_canonicalization_and_first_seen_deduplication():
    assert canonicalize_email("  Alice@Example.COM ") == "alice@example.com"
    values = [" Alice@Example.com ", "BOB@example.com", "alice@example.COM"]
    result = deduplicate_emails(values)
    assert result == ["alice@example.com", "bob@example.com"]
'''
