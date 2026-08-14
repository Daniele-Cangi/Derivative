import pytest

from library.core import merge_intervals


def test_overlapping_and_touching_intervals_are_merged_without_mutation():
    intervals = [(8, 10), (1, 3), (3, 5), (12, 15), (14, 20)]
    original = list(intervals)
    assert merge_intervals(intervals) == [(1, 5), (8, 10), (12, 20)]
    assert intervals == original
    assert merge_intervals([]) == []


def test_invalid_interval_is_rejected():
    with pytest.raises(ValueError):
        merge_intervals([(4, 3)])
