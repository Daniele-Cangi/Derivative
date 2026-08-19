
import pytest
from pyuniqseq import unique_sequences

def contiguous_subsequences(seq, min_length):
    # Returns all contiguous subsequences of length >= min_length as tuples, in order.
    n = len(seq)
    result = []
    for l in range(min_length, n + 1):
        for start in range(0, n - l + 1):
            # Convert subsequence to tuple for comparison
            subseq = tuple(seq[start:start + l])
            result.append(subseq)
    return result

def derive_unique_subsequences(iterable, min_length):
    # Reference implementation for expectation
    seen = []
    for seq in iterable:
        for subseq in contiguous_subsequences(seq, min_length):
            if subseq not in seen:
                seen.append(subseq)
    return seen

def test_empty_iterable_returns_empty_list():
    # Edge: empty input iterable
    result = unique_sequences([], min_length=1)
    assert result == []
    # Confirm with reference
    assert result == derive_unique_subsequences([], 1)

def test_sequences_with_repeated_elements():
    # sequences where elements are repeated
    seqs = [
        [1, 2, 1, 2],
        [2, 1, 2],
    ]
    min_length = 2
    expected = derive_unique_subsequences(seqs, min_length)
    result = unique_sequences(seqs, min_length)
    assert result == expected

def test_various_sequence_types():
    # Using list, tuple, and string as sequence types
    seqs = [
        [1, 2, 3],            # list
        (3, 2, 1),            # tuple
        'abc',                # string
        bytearray(b'def'),    # bytearray is a sequence of ints
    ]
    min_length = 2
    expected = derive_unique_subsequences(seqs, min_length)
    result = unique_sequences(seqs, min_length)
    assert result == expected

def test_non_hashable_members():
    # Input sequences contain lists and dicts as elements (non-hashable)
    seqs = [
        [[1], [2], [1]],                        # list of lists
        [{'a': 1}, {'b': 2}],                   # list of dicts
        (([1], {'a': 2}),),                     # tuple containing tuple of list and dict
        [([3, 4],), ({'x': 1},)],               # list with tuple and dict as elements
    ]
    min_length = 1
    expected = derive_unique_subsequences(seqs, min_length)
    result = unique_sequences(seqs, min_length)
    assert result == expected

def test_min_length_less_than_one_raises():
    seqs = [[1, 2, 3]]
    with pytest.raises(ValueError) as exc:
        unique_sequences(seqs, min_length=0)
    assert str(exc.value) == 'min_length must be at least 1'

def test_empty_sequences_in_iterable():
    # Input contains empty sequences
    seqs = [
        [],               # empty list
        (),               # empty tuple
        [1, 2, 3],
        [],               # another empty list
        (),               # another empty tuple
        'ab',
    ]
    min_length = 1
    expected = derive_unique_subsequences(seqs, min_length)
    result = unique_sequences(seqs, min_length)
    assert result == expected

def test_repeated_sequences_and_order():
    # Ensure order is as first encountered, and duplicates in different sequences are not repeated
    seqs = [
        [1, 2, 3],
        [2, 3, 4],
        [1, 2, 3],  # repetition should not produce new subsequences
    ]
    min_length = 2
    expected = derive_unique_subsequences(seqs, min_length)
    result = unique_sequences(seqs, min_length)
    assert result == expected
