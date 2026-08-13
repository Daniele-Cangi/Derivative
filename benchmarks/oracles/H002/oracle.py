import pytest

from library.core import allocate_cents


def test_largest_remainder_contract():
    assert allocate_cents(10, [1, 1, 1]) == [4, 3, 3]
    assert allocate_cents(7, [1, 2]) == [2, 5]
    assert sum(allocate_cents(101, [2, 3, 5])) == 101


def test_invalid_allocations_are_rejected():
    with pytest.raises(ValueError):
        allocate_cents(-1, [1])
    with pytest.raises(ValueError):
        allocate_cents(10, [1, -1])
