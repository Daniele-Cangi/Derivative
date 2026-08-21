import pytest
from stable_digit_group_sum import stable_digit_group_sum

import copy

def test_basic_behavioral_cases():
    """Requirement behavioral tests (1), (2), (3) and custom mix."""
    # Basic behavioral requirements
    inp1 = [[1, 2], [3], [], [-4, 4, 0]]
    result1 = stable_digit_group_sum(copy.deepcopy(inp1))
    assert result1 == [sum(g) for g in inp1]

    inp2 = []
    result2 = stable_digit_group_sum(copy.deepcopy(inp2))
    assert result2 == []

    inp3 = [[42], []]
    result3 = stable_digit_group_sum(copy.deepcopy(inp3))
    assert result3 == [42, 0]

    # Ensure input is not mutated
    org = [[1, 2, 3], [], [0]]
    org_copy = copy.deepcopy(org)
    _ = stable_digit_group_sum(org_copy)
    assert org_copy == org


def test_edge_and_order_preservation():
    """Test edge cases and invariants like empty lists, all negatives, zeros, and order preservation."""
    # All groups are empty
    inp = [[], []]
    result = stable_digit_group_sum(copy.deepcopy(inp))
    assert result == [0, 0]

    # Only negatives and zeros
    inp2 = [[-1, -1], [0], [], [0, -2]]
    result2 = stable_digit_group_sum(copy.deepcopy(inp2))
    assert result2 == [sum(g) for g in inp2]

    # Mixed order
    inp3 = [[5, -2, 8], [0], [], [-1, -1]]
    result3 = stable_digit_group_sum(copy.deepcopy(inp3))
    assert result3 == [11, 0, 0, -2]

    # Ensure input not modified for edge
    org = [[-100, 100], []]
    org_copy = copy.deepcopy(org)
    _ = stable_digit_group_sum(org_copy)
    assert org_copy == org


def test_type_errors_and_input_validation():
    """Test all invalid input variants raise TypeError exactly as required by the contract."""
    # Group contains non-int
    with pytest.raises(TypeError):
        stable_digit_group_sum([[1, 'x'], [0]])
    # Top level element is not a list
    with pytest.raises(TypeError):
        stable_digit_group_sum([None])
    with pytest.raises(TypeError):
        stable_digit_group_sum(['foo'])
    # Group contains float
    with pytest.raises(TypeError):
        stable_digit_group_sum([[1.5], [2]])
    # Nested structure not list
    with pytest.raises(TypeError):
        stable_digit_group_sum([[1, 2], 3])
    # Deeply nested but top level is correct
    with pytest.raises(TypeError):
        stable_digit_group_sum([[1, [2]], [3]])

    # No partial output: output is not produced on error
    inp = [[5, 'bad'], [8]]
    with pytest.raises(TypeError):
        stable_digit_group_sum(inp)

    # Confirm input not mutated on error
    orig = [[4, 'err'], [2]]
    orig_copy = copy.deepcopy(orig)
    with pytest.raises(TypeError):
        stable_digit_group_sum(orig_copy)
    assert orig_copy == orig
