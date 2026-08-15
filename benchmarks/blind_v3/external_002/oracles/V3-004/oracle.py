import pytest
from filter_by_predicate import filter_by_predicate

def always_true(x):
    return True

def always_false(x):
    return False

def predicate_raises_on_negatives(x):
    if x < 0:
        raise ValueError("Negative value")
    return x % 2 == 0

def infinite_ones():
    while True:
        yield 1

@ pytest.fixture
def finite_mixed_data():
    # Include negative and positive, zeros and values that will pass/fail predicate
    return [3, -1, 4, 0, -7, 2, 5]

@ pytest.fixture
def finite_data_with_exceptions():
    # List specifically chosen to trigger exceptions
    return [-2, 3, -5, 8, 0]

def test_yields_only_items_where_predicate_true(finite_mixed_data):
    result = list(filter_by_predicate(finite_mixed_data, lambda x: x % 2 == 0))
    assert result == [4, 0, 2]


def test_skips_items_when_predicate_raises(finite_data_with_exceptions):
    # Should yield only numbers >= 0 and even
    result = list(filter_by_predicate(finite_data_with_exceptions, predicate_raises_on_negatives))
    assert result == [8, 0]


def test_order_preserved_and_predicate_true_always():
    data = [10, 20, 30, 40]
    result = list(filter_by_predicate(data, always_true))
    assert result == data


def test_order_preserved_and_predicate_false_always():
    data = [10, 20, 30, 40]
    result = list(filter_by_predicate(data, always_false))
    assert result == []


def test_tolerates_infinite_iterators():
    # Take only first 5 results from infinite iterator where predicate is True
    gen = filter_by_predicate(infinite_ones(), always_true)
    results = []
    for _ in range(5):
        results.append(next(gen))
    assert results == [1, 1, 1, 1, 1]


def test_predicate_raises_non_valueerror_then_skips_item():
    # Predicate that raises TypeError on some input
    def pred(x):
        if x == 42:
            raise TypeError("No 42 allowed")
        return x < 50
    data = [10, 42, 30]
    result = list(filter_by_predicate(data, pred))
    # 42 skipped silently, 10 and 30 pass predicate
    assert result == [10, 30]


def test_empty_input_iterator():
    empty_iter = iter([])
    result = list(filter_by_predicate(empty_iter, always_true))
    assert result == []
