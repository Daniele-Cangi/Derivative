import io
import sys
import pytest
from forge_blind_v9_requirements_slot6 import main

class STDIO:
    def __init__(self, lines):
        self.stdin = io.StringIO(lines)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
    def __enter__(self):
        self._old_stdin = sys.stdin
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdin = self.stdin
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdin = self._old_stdin
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

def hex_sum_pipe_line(line):
    return str(sum(int(part, 16) for part in line.split('|'))) + '\n'

def test_mix_valid_invalid_and_single_substring():
    # Contains: valid lines (multiple substrings), valid line (single substring), various invalid edge cases
    lines = (
        '1A2b|FFFF|0001\n'      # valid: 6699+65535+1=72235
        'DEAD\n'               # valid: single substring, DEAD=57005
        'abcd|12fF\n'          # valid, [43981,4863]=48844
        'abcd|12f\n'           # invalid: '12f' is only 3 digits
        'ABCD|GHIJ\n'          # invalid: 'GHIJ' is not hex
        '0000\n'               # valid: single zero, 0
        '|1234|5678\n'         # invalid: leading pipe->empty substr
        '12 34|5678\n'         # invalid space in first substring
        '1234|5678|\n'         # invalid: trailing pipe->empty substr
        '1234 |abcd|5678\n'    # invalid: space after '1234'
        '\t1234|abcd|5678\n'  # invalid: leading tab
    )
    expected_valid = [
        '1A2b|FFFF|0001',
        'DEAD',
        'abcd|12fF',
        '0000',
    ]
    expected_output = ''.join([hex_sum_pipe_line(line) for line in expected_valid])
    with STDIO(lines) as stdio:
        exit_code = main([])
        output = stdio.stdout.getvalue()
    assert exit_code == 0
    assert output == expected_output
    assert output.count('\n') == len(expected_valid)

def test_all_whitespace_and_empty_rejections():
    # All lines invalid for pure whitespace or whitespace around substrings or delimiters.
    lines = (
        '\n'                      # empty
        '   \n'                   # whitespace only
        '1234 |abcd\n'            # space after 1234
        '1234| abcd\n'            # space before abcd
        '1234|abcd \n'            # space before newline
        ' 1234|abcd\n'            # leading space before line
        '1234|abcd|    \n'        # trailing space as extra substring
        '1234| abcd|1122\n'       # space before abcd
        'abcd|1122 |\n'           # space after 1122
        '1234|abcd|112 2\n'       # space inside substring
    )
    with STDIO(lines) as stdio:
        exit_code = main([])
        output = stdio.stdout.getvalue()
    assert exit_code == 0
    assert output == ''


def test_case_insensitivity_and_hex_validation():
    # Tests that case insensitivity and full hex subset is accepted, and non-hex appropriately rejected.
    lines = (
        'abcd|Ef12\n'           # valid: mixed case, [43981, 61202]=105183
        'FFFF\n'                # valid: all-f uppercase, 65535
        'ffff\n'                # valid: all-f lowercase, 65535
        '12AB|105f|aBcD|ef01\n' # valid: [4779,4191,43981,61185]=111136
        '0123|4567|89ab|cdef\n' # valid, all hex, [291,17767,35243,52719]=106020
        'GHIJ\n'                # invalid: non-hex
        'abcd|12345\n'          # invalid: '12345' is 5 digits
        '00g1|abcd\n'           # invalid: '00g1' contains non-hex 'g'
        'deadbeef\n'            # invalid: not exactly 4 digits
    )
    expected_valid = [
        'abcd|Ef12',
        'FFFF',
        'ffff',
        '12AB|105f|aBcD|ef01',
        '0123|4567|89ab|cdef',
    ]
    expected_output = ''.join([hex_sum_pipe_line(l) for l in expected_valid])
    with STDIO(lines) as stdio:
        exit_code = main([])
        output = stdio.stdout.getvalue()
    assert exit_code == 0
    assert output == expected_output
    assert output.count('\n') == len(expected_valid)

def test_utf8_decode_error(monkeypatch):
    # Simulate undecodable UTF-8 sequence on stdin/readline
    class BadStdin:
        def readline(self, *a, **k):
            raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')
        def __next__(self):
            raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')
        def __iter__(self):
            return self
    orig_stdin = sys.stdin
    orig_stdout = sys.stdout
    sys.stdin = BadStdin()
    sys.stdout = io.StringIO()
    try:
        exit_code = main([])
        output = sys.stdout.getvalue()
    finally:
        sys.stdin = orig_stdin
        sys.stdout = orig_stdout
    assert exit_code == 2
    assert output == ''
