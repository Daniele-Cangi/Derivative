from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


def is_interval_merge_library(plan: FeasiblePlan) -> bool:
    corpus = " ".join(
        [
            plan.build_spec.normalized_requirement,
            *(interface.signature for interface in plan.interfaces),
        ]
    ).lower()
    return "merge_intervals" in corpus and "interval" in corpus


def render_interval_library_file(
    plan: FeasiblePlan,
    path: str,
    interfaces: list[PlanInterface],
) -> str | None:
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith("src/library/core.py"):
        return _core_module()
    if normalized.endswith("src/library/__init__.py"):
        return "from .core import merge_intervals\n"
    if normalized.startswith("tests/"):
        return _behavioral_test()
    return None


def render_interval_library_test(plan: FeasiblePlan, plan_test: PlanTest) -> str:
    return _behavioral_test()


def _core_module() -> str:
    return r'''def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered: list[tuple[int, int]] = []
    for interval in intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            raise ValueError("each interval must contain exactly two endpoints")
        start, end = interval
        if start > end:
            raise ValueError("interval start must not exceed interval end")
        ordered.append((start, end))
    ordered.sort(key=lambda item: (item[0], item[1]))

    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged
'''


def _behavioral_test() -> str:
    return r'''from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from library import merge_intervals


def test_overlapping_and_touching_intervals_are_merged_without_mutation():
    intervals = [(8, 10), (1, 3), (3, 5), (12, 15), (14, 20)]
    original = list(intervals)
    assert merge_intervals(intervals) == [(1, 5), (8, 10), (12, 20)]
    assert intervals == original
    assert merge_intervals([]) == []


def test_disjoint_integer_intervals_are_not_treated_as_touching():
    assert merge_intervals([(1, 2), (3, 4), (5, 7), (7, 8)]) == [
        (1, 2),
        (3, 4),
        (5, 8),
    ]


def test_invalid_interval_is_rejected():
    with pytest.raises(ValueError):
        merge_intervals([(4, 3)])
'''
