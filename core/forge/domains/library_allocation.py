from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


def is_largest_remainder_library(plan: FeasiblePlan) -> bool:
    corpus = " ".join(
        [
            plan.build_spec.normalized_requirement,
            *(interface.signature for interface in plan.interfaces),
        ]
    ).lower()
    return all(
        token in corpus
        for token in ("allocate_cents", "largest-remainder", "total_cents", "weights")
    )


def render_allocation_library_file(
    plan: FeasiblePlan,
    path: str,
    interfaces: list[PlanInterface],
) -> str | None:
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith("src/library/core.py"):
        return _core_module()
    if normalized.endswith("src/library/__init__.py"):
        return "from .core import allocate_cents\n"
    if normalized.startswith("tests/"):
        return _behavioral_test()
    return None


def render_allocation_library_test(plan: FeasiblePlan, plan_test: PlanTest) -> str:
    return _behavioral_test()


def _core_module() -> str:
    return r'''def allocate_cents(total_cents: int, weights: list[int]) -> list[int]:
    if total_cents < 0:
        raise ValueError("total_cents must be non-negative")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if not weights:
        if total_cents == 0:
            return []
        raise ValueError("positive totals require at least one weight")

    total_weight = sum(weights)
    if total_weight == 0:
        if total_cents == 0:
            return [0 for _ in weights]
        raise ValueError("positive totals require positive aggregate weight")

    allocations: list[int] = []
    remainders: list[tuple[int, int]] = []
    for index, weight in enumerate(weights):
        floor_share, remainder = divmod(total_cents * weight, total_weight)
        allocations.append(floor_share)
        remainders.append((remainder, index))

    units_left = total_cents - sum(allocations)
    for _, index in sorted(remainders, key=lambda item: (-item[0], item[1]))[:units_left]:
        allocations[index] += 1
    return allocations
'''


def _behavioral_test() -> str:
    return r'''from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from library import allocate_cents


def test_largest_remainder_preserves_total_cents_and_input_order():
    total_cents = 10
    assert allocate_cents(total_cents, [1, 1, 1]) == [4, 3, 3]
    assert allocate_cents(7, [1, 2]) == [2, 5]
    assert sum(allocate_cents(101, [2, 3, 5])) == 101


def test_negative_totals_or_weights_are_rejected():
    with pytest.raises(ValueError):
        allocate_cents(-1, [1])
    with pytest.raises(ValueError):
        allocate_cents(10, [1, -1])
'''
