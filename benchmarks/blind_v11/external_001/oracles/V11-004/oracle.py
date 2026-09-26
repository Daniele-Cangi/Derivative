import io
import sys
import pytest
from running_minimum_cli import main
from contextlib import redirect_stdout, redirect_stderr

MAX_UINT = 2 ** 31 - 1

# Utility to run main and capture stdout and stderr deterministically
# argv: list[str]
def run_main_with_io(argv):
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()

# --- Valid input: canonical example ---
def test_valid_running_minimum_example():
    argv = ['8,5,6,3,4,3,9']
    rc, out, err = run_main_with_io(argv)
    values = [8, 5, 6, 3, 4, 3, 9]
    # Compute running minimum reference
    result = []
    curr = None
    for v in values:
        curr = v if curr is None else min(curr, v)
        result.append(str(curr))
    expected = ','.join(result)
    assert rc == 0
    assert out == expected
    # Check that all characters in output are ASCII (ord < 128)
    assert all(0 <= ord(c) < 128 for c in out)
    # Check that the output string can be encoded as ASCII
    out.encode('ascii')
    assert err == ''

# --- Edge: input is a single zero ---
def test_single_zero():
    argv = ['0']
    rc, out, err = run_main_with_io(argv)
    assert rc == 0
    assert out == '0'
    out.encode('ascii')
    assert err == ''

# --- Edge: input is a maximal allowed number ---
def test_max_uint():
    max_str = str(MAX_UINT)
    argv = [max_str]
    rc, out, err = run_main_with_io(argv)
    assert rc == 0
    assert out == max_str
    out.encode('ascii')
    assert err == ''

# --- Forbidden: leading zeros (except for value zero) ---
@pytest.mark.parametrize("val", ['01', '0004', '0000000003'])
def test_leading_zeros(val):
    argv = [val]
    rc, out, err = run_main_with_io(argv)
    assert rc == 2
    assert out == 'ERROR'
    out.encode('ascii')
    assert err == ''

# --- Forbidden: negative values ---
@pytest.mark.parametrize("val", ['-1', '-42', '-0', '3,-2,2'])
def test_negative_values(val):
    argv = [val]
    rc, out, err = run_main_with_io(argv)
    assert rc == 2
    assert out == 'ERROR'
    out.encode('ascii')
    assert err == ''

# --- Forbidden: empty values (consecutive commas, leading/trailing commas, as input and as parts) ---
@pytest.mark.parametrize("val", [',1,2', '1,,2', '1,2,', '', ',', '1,', ',2', '3,,'])
def test_empty_segments(val):
    argv = [val]
    rc, out, err = run_main_with_io(argv)
    assert rc == 2
    assert out == 'ERROR'
    out.encode('ascii')
    assert err == ''

# --- Forbidden: whitespace input anywhere ---
@pytest.mark.parametrize("val", [' 0', '0 ', '1, 2,3', '4,5\t,6', ' 7,8'])
def test_whitespace_in_input(val):
    argv = [val]
    rc, out, err = run_main_with_io(argv)
    assert rc == 2
    assert out == 'ERROR'
    out.encode('ascii')
    assert err == ''

# --- Forbidden: argument missing ---
def test_missing_argument():
    argv = []
    rc, out, err = run_main_with_io(argv)
    assert rc == 2
    assert out == 'ERROR'
    out.encode('ascii')
    assert err == ''

# --- Forbidden: extra arguments ---
def test_extra_arguments():
    argv = ['3,2,1', 'extra']
    rc, out, err = run_main_with_io(argv)
    assert rc == 2
    assert out == 'ERROR'
    out.encode('ascii')
    assert err == ''

# --- Out-of-bounds: larger than 2^31 - 1 ---
@pytest.mark.parametrize("val", [str(2**31), str(2**31 + 874), '543,41234,999999999999'])
def test_out_of_bounds(val):
    argv = [val]
    rc, out, err = run_main_with_io(argv)
    assert rc == 2
    assert out == 'ERROR'
    out.encode('ascii')
    assert err == ''

# --- Forbidden: non-integer input ---
@pytest.mark.parametrize("val", ['1.2', 'x', '2,y,3', '3.5', '1e9'])
def test_non_integer_input(val):
    argv = [val]
    rc, out, err = run_main_with_io(argv)
    assert rc == 2
    assert out == 'ERROR'
    out.encode('ascii')
    assert err == ''

# --- Valid: mixed non-monotonic sequence ---
def test_valid_non_monotonic():
    values = [50, 40, 60, 45, 45, 44]
    argv = [','.join(str(x) for x in values)]
    result = []
    curr = None
    for v in values:
        curr = v if curr is None else min(curr, v)
        result.append(str(curr))
    expected = ','.join(result)
    rc, out, err = run_main_with_io(argv)
    assert rc == 0
    assert out == expected
    out.encode('ascii')
    assert err == ''

# --- Valid: all identical values ---
def test_valid_all_identical():
    argv = ['7,7,7,7']
    rc, out, err = run_main_with_io(argv)
    assert rc == 0
    assert out == '7,7,7,7'
    out.encode('ascii')
    assert err == ''

# --- Required: No stderr on success or error ---
def test_no_stderr_on_success_and_error():
    # Success case
    argv = ['1,2,3']
    rc_s, out_s, err_s = run_main_with_io(argv)
    assert rc_s == 0
    assert err_s == ''
    # Error case
    argv = ['bad,input']
    rc_e, out_e, err_e = run_main_with_io(argv)
    assert rc_e == 2
    assert out_e == 'ERROR'
    assert err_e == ''
    out_s.encode('ascii')
    out_e.encode('ascii')

# --- Valid: input sequence decreasing ---
def test_valid_strictly_decreasing():
    values = [10, 5, 2, 1, 0]
    argv = [','.join(str(x) for x in values)]
    result = []
    curr = None
    for v in values:
        curr = v if curr is None else min(curr, v)
        result.append(str(curr))
    expected = ','.join(result)
    rc, out, err = run_main_with_io(argv)
    assert rc == 0
    assert out == expected
    out.encode('ascii')
    assert err == ''
