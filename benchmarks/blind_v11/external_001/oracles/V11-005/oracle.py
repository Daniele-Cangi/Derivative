import pytest
import sys
from io import StringIO
from rotate_cli import main

# Utility to capture stdout for assertions, never allow trailing newline or extra whitespace in output

def run_cli(args):
    old_stdout = sys.stdout
    buf = StringIO()
    sys.stdout = buf
    try:
        rc = main(args)
    finally:
        sys.stdout = old_stdout
    return rc, buf.getvalue()

# --- ACCEPTANCE TESTS ---

def test_rotation_typical():
    args = ['2,11,7,8']
    # K = 2, N = 4, rotate left by 2 -> [7,8,2,11]
    seq = [2, 11, 7, 8]
    K, N = seq[0], len(seq)
    K_mod = K % N
    rotated = seq[K_mod:] + seq[:K_mod]
    expected = ','.join(str(x) for x in rotated)
    rc, out = run_cli(args)
    assert out == expected
    assert rc == 0

def test_rotation_k_larger_than_length():
    args = ['5,10,20,30']
    seq = [5, 10, 20, 30]
    K, N = seq[0], len(seq)
    K_mod = K % N
    rotated = seq[K_mod:] + seq[:K_mod]
    expected = ','.join(str(x) for x in rotated)
    rc, out = run_cli(args)
    assert out == expected
    assert rc == 0

def test_rotation_by_zero_first_element():
    args = ['0,100,200,300']
    seq = [0, 100, 200, 300]
    K, N = seq[0], len(seq)
    K_mod = K % N
    rotated = seq[K_mod:] + seq[:K_mod]
    expected = ','.join(str(x) for x in rotated)
    rc, out = run_cli(args)
    assert out == expected
    assert rc == 0

def test_single_element_input():
    args = ['0']
    expected = '0'
    rc, out = run_cli(args)
    assert out == expected
    assert rc == 0

def test_modulo_wrap_two_elements():
    args = ['10,7']
    seq = [10, 7]
    K, N = seq[0], len(seq)
    K_mod = K % N
    rotated = seq[K_mod:] + seq[:K_mod]
    expected = ','.join(str(x) for x in rotated)
    rc, out = run_cli(args)
    assert out == expected
    assert rc == 0

def test_rotation_leading_zero_first():
    args = ['0,33']
    seq = [0, 33]
    K, N = seq[0], len(seq)
    K_mod = K % N
    rotated = seq[K_mod:] + seq[:K_mod]
    expected = ','.join(str(x) for x in rotated)
    rc, out = run_cli(args)
    assert out == expected
    assert rc == 0

# --- ERROR HANDLING ---

def test_invalid_empty_input():
    args = ['']
    rc, out = run_cli(args)
    assert out == 'ERROR'
    assert rc == 2

def test_invalid_leading_zeros():
    args = ['01,2,3']
    rc, out = run_cli(args)
    assert out == 'ERROR'
    assert rc == 2
    args2 = ['0,02,3']
    rc2, out2 = run_cli(args2)
    assert out2 == 'ERROR'
    assert rc2 == 2

def test_invalid_negative_value():
    args = ['1,-2,3']
    rc, out = run_cli(args)
    assert out == 'ERROR'
    assert rc == 2

def test_invalid_out_of_range():
    args = ['0,999,1000']
    rc, out = run_cli(args)
    assert out == 'ERROR'
    assert rc == 2
    args = ['1001']
    rc, out = run_cli(args)
    assert out == 'ERROR'
    assert rc == 2

def test_invalid_empty_segment():
    for bad_in in [',', '1,,2', '3,', ',4', '0,,1']:
        rc, out = run_cli([bad_in])
        assert out == 'ERROR'
        assert rc == 2

def test_invalid_space_and_plus():
    for bad_in in [' 1,2', '1 ,2', '1,2 ', '1, 2', '+1,2', '1,+2']:
        rc, out = run_cli([bad_in])
        assert out == 'ERROR'
        assert rc == 2

def test_non_ascii_input():
    for bad_in in ['1,2,3\u00e9', '1,2\u00a03', '\uff11,3']:
        rc, out = run_cli([bad_in])
        assert out == 'ERROR'
        assert rc == 2

def test_missing_argument():
    rc, out = run_cli([])
    assert out == 'ERROR'
    assert rc == 2

def test_extra_positional_argument():
    rc, out = run_cli(['2,3,4', '5'])
    assert out == 'ERROR'
    assert rc == 2

def test_whitespace_segments():
    for bad_in in ['1, 99', '1,99 ', ' 1,99', '1,\t99', '1,\n99']:
        rc, out = run_cli([bad_in])
        assert out == 'ERROR'
        assert rc == 2

def test_no_stderr_output_on_error(monkeypatch):
    import sys
    from io import StringIO
    fake_err = StringIO()
    monkeypatch.setattr(sys, 'stderr', fake_err)
    rc, out = run_cli(['foo'])
    assert fake_err.getvalue() == ''

def test_no_stderr_output_on_success(monkeypatch):
    import sys
    from io import StringIO
    fake_err = StringIO()
    monkeypatch.setattr(sys, 'stderr', fake_err)
    rc, out = run_cli(['1,20,30'])
    assert fake_err.getvalue() == ''
