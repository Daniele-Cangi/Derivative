from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


def is_semver_library(plan: FeasiblePlan) -> bool:
    corpus = " ".join(
        [
            plan.build_spec.normalized_requirement,
            *(interface.signature for interface in plan.interfaces),
        ]
    ).lower()
    return "compare_versions" in corpus and (
        "semantic versioning" in corpus or "semver" in corpus
    )


def render_semver_library_file(
    plan: FeasiblePlan,
    path: str,
    interfaces: list[PlanInterface],
) -> str | None:
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith("src/library/core.py"):
        return _core_module()
    if normalized.endswith("src/library/__init__.py"):
        return "from .core import compare_versions\n"
    if normalized.startswith("tests/"):
        return _behavioral_test()
    return None


def render_semver_library_test(plan: FeasiblePlan, plan_test: PlanTest) -> str:
    return _behavioral_test()


def _core_module() -> str:
    return r'''import re


_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


def _parse_version(value: str) -> tuple[tuple[int, int, int], list[str] | None]:
    if not isinstance(value, str):
        raise ValueError("version must be a string")
    match = _SEMVER_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid Semantic Version: {value!r}")
    core = tuple(int(match.group(index)) for index in range(1, 4))
    prerelease = match.group(4)
    return core, prerelease.split(".") if prerelease is not None else None


def _compare_prerelease(left: list[str] | None, right: list[str] | None) -> int:
    if left is None or right is None:
        if left is right:
            return 0
        return 1 if left is None else -1
    for left_id, right_id in zip(left, right):
        if left_id == right_id:
            continue
        left_numeric = left_id.isdigit()
        right_numeric = right_id.isdigit()
        if left_numeric and right_numeric:
            return -1 if int(left_id) < int(right_id) else 1
        if left_numeric != right_numeric:
            return -1 if left_numeric else 1
        return -1 if left_id < right_id else 1
    return (len(left) > len(right)) - (len(left) < len(right))


def compare_versions(left: str, right: str) -> int:
    left_core, left_prerelease = _parse_version(left)
    right_core, right_prerelease = _parse_version(right)
    if left_core != right_core:
        return -1 if left_core < right_core else 1
    return _compare_prerelease(left_prerelease, right_prerelease)
'''


def _behavioral_test() -> str:
    return r'''from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from library import compare_versions


def test_semver_precedence_and_build_metadata():
    versions = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
    ]
    assert all(compare_versions(a, b) == -1 for a, b in zip(versions, versions[1:]))
    assert compare_versions("2.0.0", "1.99.99") == 1
    assert compare_versions("1.2.3+build.1", "1.2.3+build.99") == 0
    assert compare_versions("1.0.0-alpha01", "1.0.0-alpha02") == -1


@pytest.mark.parametrize("value", ["1.0", "01.0.0", "1.0.0-01", "1.0.0-", "1.0.0+build+meta"])
def test_invalid_semver_is_rejected(value):
    with pytest.raises(ValueError):
        compare_versions(value, "1.0.0")
'''
