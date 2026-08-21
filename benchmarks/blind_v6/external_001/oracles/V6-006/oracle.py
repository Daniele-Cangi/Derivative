import pytest
from groupwise_rotate import groupwise_rotate

# (1) groupwise_rotate(['a','b','c','d','e','f'], 2) returns ['b','a','d','c','f','e']
def test_groupwise_rotate_size_2_groups():
    input_data = ['a','b','c','d','e','f']
    group_size = 2
    expected = ['b','a','d','c','f','e']
    result = groupwise_rotate(input_data, group_size)
    assert result == expected
    assert input_data == ['a','b','c','d','e','f']  # Ensure immutability

# (2) groupwise_rotate(['foo','bar','baz'], 2) returns ['bar','foo','baz']
def test_groupwise_rotate_partial_tail():
    input_data = ['foo', 'bar', 'baz']
    group_size = 2
    expected = ['bar', 'foo', 'baz']
    result = groupwise_rotate(input_data, group_size)
    assert result == expected
    assert input_data == ['foo', 'bar', 'baz']

# (3) groupwise_rotate([], 3) returns []
def test_groupwise_rotate_empty_list():
    input_data = []
    group_size = 3
    expected = []
    result = groupwise_rotate(input_data, group_size)
    assert result == expected

# (4) groupwise_rotate(['x'], 5) returns ['x']
def test_groupwise_rotate_single_element_large_group():
    input_data = ['x']
    group_size = 5
    expected = ['x']
    result = groupwise_rotate(input_data, group_size)
    assert result == expected

# (5) groupwise_rotate(['m','n','o','p'], 3) returns ['o','m','n','p']
def test_groupwise_rotate_non_trivial_partial_tail():
    input_data = ['m','n','o','p']
    group_size = 3
    # First group ['m','n','o'] rotates right, becomes ['o','m','n']; last group ['p'] rotates (no-op)
    expected = ['o','m','n','p']
    result = groupwise_rotate(input_data, group_size)
    assert result == expected
    assert input_data == ['m','n','o','p']

# (6) groupwise_rotate(['a', None], 2) raises TypeError
def test_groupwise_rotate_type_error():
    input_data = ['a', None]
    group_size = 2
    with pytest.raises(TypeError):
        groupwise_rotate(input_data, group_size)

# (7) groupwise_rotate(['a','b','c'], 0) raises ValueError
def test_groupwise_rotate_value_error():
    input_data = ['a', 'b', 'c']
    group_size = 0
    with pytest.raises(ValueError):
        groupwise_rotate(input_data, group_size)

# Additional edge: group_size not int
def test_groupwise_rotate_group_size_not_int():
    with pytest.raises(ValueError):
        groupwise_rotate(['a','b','c'], 1.5)
    with pytest.raises(ValueError):
        groupwise_rotate(['x'], '2')

# Additional edge: input is not a list
def test_groupwise_rotate_strings_not_a_list():
    with pytest.raises(TypeError):
        groupwise_rotate('not a list', 2)
    with pytest.raises(TypeError):
        groupwise_rotate({'a','b'}, 2)

# Additional edge: input list contains non-string types
def test_groupwise_rotate_with_ints():
    with pytest.raises(TypeError):
        groupwise_rotate(['a', 7, 'b'], 2)
    with pytest.raises(TypeError):
        groupwise_rotate([None], 1)
