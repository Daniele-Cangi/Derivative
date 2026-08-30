import io
import sys
import pytest
from forge_blind_v9_requirements_slot4 import main

# Helper context manager to patch sys.stdin and sys.stdout in-process,
# capturing output deterministically.
class IOSubstitute:
    def __init__(self, stdin_text):
        self.stdin_text = stdin_text
        self.stdin = io.StringIO(stdin_text)
        self.stdout = io.StringIO()
        self._orig_stdin = None
        self._orig_stdout = None
    def __enter__(self):
        self._orig_stdin = sys.stdin
        self._orig_stdout = sys.stdout
        sys.stdin = self.stdin
        sys.stdout = self.stdout
        return self
    def __exit__(self, exc_type, exc_value, tb):
        sys.stdin = self._orig_stdin
        sys.stdout = self._orig_stdout


def pad_value(val):
    # Pads a single value as per requirements
    if val == '':
        return 'XXXXX'
    gaps = 5 - len(val)
    return ('X' * gaps) + val if gaps > 0 else val

def reference_line(line):
    # Returns the reference output for a single line
    body = line.rstrip('\n')
    postfix = '\n' if line.endswith('\n') else ''
    vals = body.split(',')
    return ','.join([pad_value(v) for v in vals]) + postfix

def reference_fixture_multiline(input_text):
    return ''.join([reference_line(l) for l in input_text.splitlines(keepends=True)])

# TESTS

def test_standard_mixed_input():
    input_text = (
        'A,B,CD,\n'
        'ZZZZ,\n'
        ',,HELLO\n'
        'APPLE,,ORANGE,\n'
        '\n'             # empty line
        'EGG,,'
    )
    # Expected output
    expected_output = reference_fixture_multiline(input_text)
    with IOSubstitute(input_text) as ioctx:
        exit_code = main()
        output = ioctx.stdout.getvalue()
    assert exit_code == 0
    assert output == expected_output
    # Confirm exact output lines and padding per column
    out_lines = output.splitlines(keepends=True)
    for inp, outp in zip(input_text.splitlines(keepends=True), out_lines):
        inp_vals = inp.rstrip('\n').split(',')
        out_vals = outp.rstrip('\n').split(',')
        assert len(inp_vals) == len(out_vals)
        for iv, ov in zip(inp_vals, out_vals):
            if iv == '':
                assert ov == 'XXXXX'
            else:
                assert ov.endswith(iv)
                assert len(ov) >= 5
                assert all(c == 'X' for c in ov[:max(0,5-len(iv))])


def test_edge_cases_empty_and_short_and_long():
    # Empty fields, minimum and maximum width
    input_text = ',,,\nZZZZZ,XXXXX,,\nPRIDE,JOY,\nLONGFIELD,SUPERLONGFIELD123,\n'
    # For valid input, all values must be only uppercase letters or empty
    # Here 'SUPERLONGFIELD123' is NOT valid and should trigger code 2 without output.
    with IOSubstitute(input_text) as ioctx:
        exit_code = main()
        output = ioctx.stdout.getvalue()
    # Should have exited with code 2 and produced no output
    assert exit_code == 2
    assert output == ''
    # Now, test with all-valid fields covering empty, short and at-min width
    good_input = ',ZZZZZ,,,YYY,\nXXXXX,AA,\n'
    expected_output = reference_fixture_multiline(good_input)
    with IOSubstitute(good_input) as ioctx:
        exit_code = main()
        output = ioctx.stdout.getvalue()
    assert exit_code == 0
    assert output == expected_output
    # Confirm padding and field widths
    for val in output.replace('\n', '').split(','):
        assert len(val) >= 5
        if val != 'XXXXX':
            assert set(val).issubset({'X'} | set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))


def test_invalid_characters_and_utf8():
    # Various input errors
    bad_inputs = [
        'ABCdEF,XYZ\n',       # lowercase
        '12ABC,DEF\n',        # digit in field
        'APPLE,BANA@NA\n',    # symbol in field
        ',,apples,oranges\n', # lowercase everywhere
        'FOO ,BAR\n',         # space in value
        '\N{LATIN SMALL LETTER ETH},FOO\n', # non-ascii unicode (ð)
        'CAFE,DOSÉ\n',        # non-ascii (É)
        'CA,FE,!,\n',         # punctuation
    ]
    # Each should yield exit 2 and no output
    for bad in bad_inputs:
        with IOSubstitute(bad) as ioctx:
            exit_code = main()
            out = ioctx.stdout.getvalue()
        assert exit_code == 2
        assert out == ''
    # Now, simulate undecodable UTF-8 input by directly patching sys.stdin to a non-UTF-8-bytes stream
    # The requirement mandates exit(2) and no output. This is only possible for the CLI logic if it directly reads undecodable characters and fails.
    # Since Python 3 sys.stdin expects text IO (already decoded), simulate this by setting its encoding to non-UTF8 and reading "broken" bytes.
    # Instead, patch sys.stdin with a mock that raises UnicodeDecodeError
    class FailingStdin:
        def readline(self, *a, **kw):
            raise UnicodeDecodeError('utf-8', b'\xff\xfe', 0, 2, 'invalid start byte')
    orig_stdin = sys.stdin
    orig_stdout = sys.stdout
    try:
        sys.stdin = FailingStdin()
        sys.stdout = io.StringIO()
        exit_code = main()
        assert exit_code == 2
        assert sys.stdout.getvalue() == ''
    finally:
        sys.stdin = orig_stdin
        sys.stdout = orig_stdout
