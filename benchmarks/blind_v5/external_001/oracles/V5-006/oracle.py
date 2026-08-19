
import pytest
from pyrotatefields import rotate_fields

# Utility function to perform reference rotation for expected results
def reference_rotate(row, field_order, shift):
    n = len(field_order)
    if not field_order or n == 0:
        return row.copy()
    shift = shift % n if n else 0
    new_row = row.copy()
    values = [row.get(f, None) for f in field_order]
    rotated = values[-shift:] + values[:-shift] if shift else values
    for i, f in enumerate(field_order):
        new_row[f] = rotated[i]
    return new_row

def test_positive_shift_basic():
    rows = [
        {'A': 1, 'B': 2, 'C': 3, 'X': 99},
        {'A': 4, 'B': 5, 'C': 6, 'X': 88},
    ]
    field_order = ['A', 'B', 'C']
    shift = 1
    expected = [reference_rotate(r, field_order, shift) for r in rows]
    result = rotate_fields(rows, field_order, shift)
    assert result == expected


def test_negative_shift():
    rows = [
        {'K': 9, 'J': 8, 'L': 7},
        {'K': 6, 'J': 5, 'L': 4},
    ]
    field_order = ['K', 'J', 'L']
    shift = -1
    expected = [reference_rotate(r, field_order, shift) for r in rows]
    result = rotate_fields(rows, field_order, shift)
    assert result == expected


def test_missing_fields_in_row():
    rows = [
        {'x': 'a', 'y': 'b'},              # missing 'z'
        {'x': 'c', 'z': 'd'},              # missing 'y'
        {'z': 'e'},                        # missing 'x', 'y'
    ]
    field_order = ['x', 'y', 'z']
    shift = 2
    expected = [reference_rotate(r, field_order, shift) for r in rows]
    result = rotate_fields(rows, field_order, shift)
    assert result == expected


def test_extra_fields_remain_unchanged():
    rows = [
        {'foo': 1, 'bar': 2, 'baz': 3, 'unrelated': 'x'},
    ]
    field_order = ['foo', 'bar', 'baz']
    shift = 1
    out = rotate_fields(rows, field_order, shift)
    # unrelated key must be unmodified, rotation applies only to specified fields
    assert out[0]['unrelated'] == rows[0]['unrelated']
    assert all(k in out[0] for k in rows[0])
    assert len(out[0]) == len(rows[0])
    expected = [reference_rotate(r, field_order, shift) for r in rows]
    assert out == expected


def test_empty_rows_returns_empty():
    rows = []
    field_order = ['p', 'q']
    result = rotate_fields(rows, field_order)
    assert result == []


def test_empty_field_order_no_changes():
    rows = [{'a': 1, 'b': 2}, {'b': 4, 'c': 5}]
    field_order = []
    result = rotate_fields(rows, field_order, 1)
    # Should be identical
    assert result == rows
    # Each output is not a reference to input row
    assert all(r1 is not r2 for r1, r2 in zip(result, rows))


def test_shift_zero_or_cycle_len_means_no_change():
    rows = [{'g': 1, 'h': 2, 'i': 3}, {'g': 4, 'h': 5, 'i': 6}]
    field_order = ['g', 'h', 'i']
    shift0 = 0
    shift3 = 3
    expected = [reference_rotate(r, field_order, shift0) for r in rows]
    out0 = rotate_fields(rows, field_order, shift0)
    out3 = rotate_fields(rows, field_order, shift3)
    assert out0 == expected
    assert out3 == expected


def test_type_errors():
    # field_order not list
    with pytest.raises(TypeError):
        rotate_fields([{'a':1}], 'a', 1)
    with pytest.raises(TypeError):
        rotate_fields([{'a':1}], 5, 1)
    with pytest.raises(TypeError):
        rotate_fields([{'a':1}], [1, 'a'], 1)

    # shift not int
    with pytest.raises(TypeError):
        rotate_fields([{'a':1}], ['a'], 1.5)
    with pytest.raises(TypeError):
        rotate_fields([{'a':1}], ['a'], 'b')

    # rows element not dict
    with pytest.raises(TypeError):
        rotate_fields([42], ['a'], 1)
    with pytest.raises(TypeError):
        rotate_fields([{'a':1}, None], ['a'], 1)


def test_rows_unaffected_for_non_field_order_keys():
    rows = [{'x': 1, 'y': 2, 'z': 3, 'extra': 8}, {'x': 5, 'y': 6, 'z': 7, 'extra': 9}]
    field_order = ['x', 'y', 'z']
    shift = 2
    result = rotate_fields(rows, field_order, shift)
    for i, out_row in enumerate(result):
        in_row = rows[i]
        # extra field should be identical
        assert out_row.get('extra') == in_row.get('extra')
