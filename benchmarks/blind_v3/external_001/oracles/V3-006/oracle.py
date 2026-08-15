import pytest
from typing import Iterable, Callable

# The function under test has to be imported by its exact name; assuming it is called 'find_first_index'
from src.find_first_index import find_first_index


def test_found_match_basic():
    # Using a simple list and a callable that matches an element
    data = [1, 3, 5, 7, 9]
    predicate = lambda x: x > 4
    result = find_first_index(predicate, data)
    assert result == 2  # 5 is first element > 4


def test_no_match_returns_minus_one():
    data = [2, 4, 6, 8]
    predicate = lambda x: x > 10
    result = find_first_index(predicate, data)
    assert result == -1


def test_non_callable_raises_type_error():
    data = [1, 2, 3]
    not_callable = 5
    with pytest.raises(TypeError):
        find_first_index(not_callable, data)


def test_non_iterable_raises_type_error():
    predicate = lambda x: x == 1
    not_iterable = 42
    with pytest.raises(TypeError):
        find_first_index(predicate, not_iterable)


def test_short_circuit_behavior():
    class CountableIterable:
        def __init__(self):
            self.data = [0, 1, 2, 3, 4]
            self.consumed = 0
        def __iter__(self):
            for item in self.data:
                self.consumed += 1
                yield item
    
    iterable = CountableIterable()
    predicate = lambda x: x == 2
    index = find_first_index(predicate, iterable)
    # The index should be 2
    assert index == 2
    # The number of consumed elements should be exactly index+1, no extra
    assert iterable.consumed == 3


def test_empty_iterable_returns_minus_one():
    predicate = lambda x: True
    empty = []
    assert find_first_index(predicate, empty) == -1


def test_predicate_returns_true_on_first_element():
    data = [10, 20, 30]
    predicate = lambda x: x == 10
    assert find_first_index(predicate, data) == 0


def test_predicate_returns_true_on_last_element():
    data = [10, 20, 30]
    predicate = lambda x: x == 30
    assert find_first_index(predicate, data) == 2


def test_iterable_is_generator():
    def gen():
        for i in range(5):
            yield i
    predicate = lambda x: x == 3
    assert find_first_index(predicate, gen()) == 3


def test_callable_object():
    class Predicate:
        def __init__(self):
            self.calls = 0
        def __call__(self, x):
            self.calls += 1
            return x == 1
    predicate = Predicate()
    data = [0, 1, 2]
    idx = find_first_index(predicate, data)
    assert idx == 1
    assert predicate.calls == 2  # Called twice until match
