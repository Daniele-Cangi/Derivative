# V5-005: Acceptance tests for pygroupbyrun.groupby_runs
import pytest
from pygroupbyrun import groupby_runs

# Helper functions to derive expected grouping reference, independent of implementation
import itertools

def reference_groupby_runs(iterable, key_func):
    """
    Reference implementation using itertools.groupby, guaranteed cross-platform, key param,
    but groupby requires consecutive keys; exactly what we want. Returns list of (key, run list) tuples.
    """
    result = []
    for k, group in itertools.groupby(iterable, key=key_func):
        run = list(group)
        result.append((k, run))
    return result

# --- Test Cases ---

def test_groupby_runs_empty_iterable():
    res = groupby_runs([])
    assert res == [], "Should return an empty list for empty input"
    # For generator input as well
    res2 = groupby_runs((x for x in []))
    assert res2 == []


def test_groupby_runs_all_unique_keys():
    data = [10, 20, 30]
    # Each element should start a new run since keys never repeat consecutively
    expected = reference_groupby_runs(data, lambda x: x)
    out = groupby_runs(data)
    assert out == expected


def test_groupby_runs_repeated_consecutive_keys():
    data = [1, 1, 1, 2, 2, 1, 1, 3]
    expected = reference_groupby_runs(data, lambda x: x)
    out = groupby_runs(data)
    assert out == expected


def test_groupby_runs_unhashable_elements():
    # List of lists, which are unhashable, but equality compares their contents
    data = [[1], [1], [2], [2], [1]]
    expected = reference_groupby_runs(data, lambda x: x)
    out = groupby_runs(data)
    assert out == expected


def test_groupby_runs_custom_key_function():
    data = [10, 11, 12, 20, 21, 30]
    # Group by the tens digit
    key_fn = lambda x: x // 10
    expected = reference_groupby_runs(data, key_fn)
    out = groupby_runs(data, key=key_fn)
    assert out == expected


def test_groupby_runs_generator_input():
    # Provide a generator input
    def mygen():
        for val in [1, 2, 2, 3, 1, 1]:
            yield val
    gen = mygen()
    expected = reference_groupby_runs([1,2,2,3,1,1], lambda x: x)
    out = groupby_runs(gen)
    assert out == expected


def test_groupby_runs_unhashable_key_values():
    # Key function returns an unhashable (list)
    data = ['abc', 'abd', 'aef', 'xyz']
    # Group by first two characters as list
    key_fn = lambda s: list(s)[:2]
    expected = reference_groupby_runs(data, key_fn)
    out = groupby_runs(data, key=key_fn)
    # The group key is a list; compare equality per requirement
    assert len(out) == len(expected)
    for (group_key, group_run), (exp_key, exp_run) in zip(out, expected):
        assert group_key == exp_key
        assert group_run == exp_run


def test_groupby_runs_non_iterable_input_typeerror():
    # int is not iterable
    with pytest.raises(TypeError):
        groupby_runs(5)
    # None is not iterable
    with pytest.raises(TypeError):
        groupby_runs(None)


def test_groupby_runs_non_callable_key_typeerror():
    data = [1, 2]
    with pytest.raises(TypeError):
        groupby_runs(data, key='not a function')
    with pytest.raises(TypeError):
        groupby_runs(data, key=None)


def test_groupby_runs_input_not_modified():
    data = [1, 2, 2, 3]
    data_copy = list(data)
    groupby_runs(data)
    assert data == data_copy, "Input data must not be modified"
    # Also test for generator: no side effects, exhausted
    def gen():
        for i in [1,2,2,3]:
            yield i
    g = gen()
    list(g)
    # No change to the generator itself, but it's exhausted, which is expected when iterated
