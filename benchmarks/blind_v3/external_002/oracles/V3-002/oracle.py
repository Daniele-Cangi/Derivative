import pytest
from invert_dictionary import invert_dictionary

@pytest.fixture
def sample_dict():
    return {'apple': 'fruit', 'carrot': 'vegetable', 'banana': 'fruit', 'beet': 'vegetable', 'cherry': 'fruit'}

@pytest.fixture
def empty_dict():
    return {}

@pytest.fixture
def single_item_dict():
    return {'key': 'value'}

@pytest.fixture
def non_string_key_dict():
    return {1: 'one', 'two': '2'}

@pytest.fixture
def non_string_value_dict():
    return {'one': 1, 'two': 'two'}

@pytest.fixture
def mixed_same_value_dict():
    return {'a': 'x', 'b': 'x', 'c': 'x', 'd': 'x'}

@pytest.mark.parametrize("input_dict, expected_output", [
    
    ({'a': 'b'}, {'b': ['a']}),
    ({'a': 'b', 'c': 'b'}, {'b': ['a', 'c']}),
    ({'a': 'b', 'c': 'd'}, {'b': ['a'], 'd': ['c']}),
    ({'k1': 'v1', 'k2': 'v2', 'k3': 'v1'}, {'v1': ['k1', 'k3'], 'v2': ['k2']}),
    ({}, {}),
    ({'a': '1', 'b': '1', 'c': '0'}, {'0': ['c'], '1': ['a', 'b']})
])
def test_invert_dictionary_basic(input_dict, expected_output):
    result = invert_dictionary(input_dict)
    assert isinstance(result, dict)
    assert all(isinstance(k, str) for k in result.keys())
    assert all(isinstance(v, list) for v in result.values())
    # Check keys sorted ascending
    for values in result.values():
        assert values == sorted(values)
    # Check expected output
    assert result == expected_output

def test_invert_dictionary_empty(empty_dict):
    result = invert_dictionary(empty_dict)
    assert result == {}


def test_invert_dictionary_single_item(single_item_dict):
    result = invert_dictionary(single_item_dict)
    assert result == {'value': ['key']}


def test_invert_dictionary_non_string_key(non_string_key_dict):
    with pytest.raises(TypeError):
        invert_dictionary(non_string_key_dict)


def test_invert_dictionary_non_string_value(non_string_value_dict):
    with pytest.raises(TypeError):
        invert_dictionary(non_string_value_dict)


def test_invert_dictionary_multiple_keys_same_value(mixed_same_value_dict):
    result = invert_dictionary(mixed_same_value_dict)
    expected = {'x': ['a', 'b', 'c', 'd']}
    assert result == expected
    # Check list sorted
    assert result['x'] == sorted(result['x'])


def test_invert_dictionary_ordering():
    d = {'c': '1', 'b': '1', 'a': '1'}
    result = invert_dictionary(d)
    assert result == {'1': ['a', 'b', 'c']}  # sorted keys list


def test_invert_dictionary_raises_on_non_string_key():
    d = {42: 'answer'}
    with pytest.raises(TypeError):
        invert_dictionary(d)


def test_invert_dictionary_raises_on_non_string_value():
    d = {'answer': 42}
    with pytest.raises(TypeError):
        invert_dictionary(d)
