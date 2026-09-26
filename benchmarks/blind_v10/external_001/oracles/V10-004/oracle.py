import pytest
import sys
from io import StringIO
from forge_bench_slot4 import main

def run_cli(args):
    '''Invoke main(args), capture sys.stdout, and return (exit code, captured output).'''
    old_stdout = sys.stdout
    buf = StringIO()
    try:
        sys.stdout = buf
        rc = main(args)
    finally:
        sys.stdout = old_stdout
    return rc, buf.getvalue()

def reference_expand_rle(pattern):
    # Pure reference expansion: Raise ValueError on any nonconformant input
    import re
    runs = []
    i = 0
    n = len(pattern)
    if n == 0:
        raise ValueError('Pattern is empty')
    while i < n:
        # Extract 1-3 ASCII digits
        j = i
        while j < n and pattern[j].isdigit() and (j-i) < 3:
            j += 1
        if j == i:
            raise ValueError('No digits for run at pos %d' % i)
        digits = pattern[i:j]
        if not (1 <= len(digits) <= 3):
            raise ValueError('Digits group wrong length')
        if j >= n:
            raise ValueError('No letter after run digits')
        letter = pattern[j]
        if not (letter.isascii() and letter.islower() and letter.isalpha()):
            raise ValueError('Run letter not valid')
        count = int(digits)
        if not (1 <= count <= 255):
            raise ValueError('Count out of range')
        runs.append(letter * count)
        i = j + 1
    return ''.join(runs) + '\n'

def test_valid_examples():
    # Simple valid: 10a3b2z -> 10 a's, 3 b's, 2 z's
    arg = '10a3b2z'
    expected = reference_expand_rle(arg)
    rc, out = run_cli([arg])
    assert rc == 0
    assert out == expected
    assert out.count('a') == 10
    assert out.count('b') == 3
    assert out.count('z') == 2
    assert out.endswith('\n')
    assert out.count('\n') == 1

def test_valid_min_and_max_runs():
    # Edge runs, single minimum/maximum
    for pattern in ['1a', '255z']:
        expected = reference_expand_rle(pattern)
        rc, out = run_cli([pattern])
        assert rc == 0
        assert out == expected
        value = int(pattern[:-1])
        letter = pattern[-1]
        assert out[:-1] == letter * value
        assert out.endswith('\n')
        assert out.count('\n') == 1
    # Multiple maximum-length runs
    pattern = '255b255c1a'
    expected = reference_expand_rle(pattern)
    rc, out = run_cli([pattern])
    assert rc == 0
    assert out == expected
    assert out.count('b') == 255
    assert out.count('c') == 255
    assert out.count('a') == 1

def test_valid_multi_run_edge_length():
    # Test maximal run digit field (3 digits) and edge counts
    pattern = '1x99y255z'
    expected = reference_expand_rle(pattern)
    rc, out = run_cli([pattern])
    assert rc == 0
    assert out == expected
    assert out.count('x') == 1
    assert out.count('y') == 99
    assert out.count('z') == 255
    # Confirm all output is concatenated as specified
    splits = [c for c in 'xyz']
    out_no_nl = out[:-1]
    assert out_no_nl.startswith('x')
    assert 'y' in out_no_nl and 'z' in out_no_nl

def test_invalid_counts():
    # Out-of-range or repeated zeroes
    for pattern in ['0a', '00a', '000a', '002a', '256b', '999z']:
        rc, out = run_cli([pattern])
        assert rc == 1
        assert out == 'error\n'

def test_invalid_nonletters_and_format():
    # Missing or extra digits, uppercase letters, missing letter, stray characters, whitespace, symbols
    invalids = [
        '',      # Empty
        'a2c',   # Letters in digit field
        'A2c',   # Uppercase letter in run
        '3A',    # Uppercase
        '12',    # No final letter
        '123',   # No letter
        '1234a', # Too many digits
        '2',     # No letter
        '1',     # No letter
        '3\n',   # Symbol
        '1 a',   # Whitespace
        '2a3b9', # Extra trailing digits
        '1\t',   # Tab
        '1_a',   # Underscore
        '1\u00e1', # Non-ASCII letter
        '255Z',  # Uppercase
        '1#',    # Symbol
        '1.',    # Symbol
        ' 1a',   # Leading whitespace
        '1a ',   # Trailing whitespace
        '1',     # Only digits
        '10',    # Only digits
    ]
    for pattern in invalids:
        rc, out = run_cli([pattern])
        assert rc == 1
        assert out == 'error\n'

def test_invalid_trailing_and_extra_arg():
    # Trailing data after valid pattern
    rc, out = run_cli(['1afoo'])
    assert rc == 1
    assert out == 'error\n'
    # Extra CLI argument
    rc, out = run_cli(['10a','3b'])
    assert rc == 1
    assert out == 'error\n'
