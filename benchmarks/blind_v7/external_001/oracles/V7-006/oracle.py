import io
import sys
import os
import tempfile
import pytest
from forge_blind_v7.cli_filter_alternating_case import main

def ref_alternating_case_lines(lines):
    # Returns list of lines that satisfy the alternation rule, preserving line endings and non-alphabetic characters
    def is_alternating(line):
        chars = [c for c in line if c.isalpha()]
        if len(chars) < 2:
            return True
        for a, b in zip(chars, chars[1:]):
            if (a.islower() and b.isupper()) or (a.isupper() and b.islower()):
                continue
            return False
        return True
    return [line for line in lines if is_alternating(line)]

def run_main_with_input(input_text, file_arg=None):
    orig_stdin = sys.stdin
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        if file_arg:
            with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf8', newline='') as f:
                f.write(input_text)
                f.flush()
                filename = f.name
            try:
                exit_code = main([filename])
            finally:
                os.unlink(filename)
        else:
            sys.stdin = io.StringIO(input_text)
            exit_code = main([])
        sys.stdout.seek(0)
        sys.stderr.seek(0)
        return sys.stdout.read(), sys.stderr.read(), exit_code
    finally:
        sys.stdin = orig_stdin
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

def test_canonical_behaviors():
    lines = ['AbCdEf', 'abC', 'A-b', 'ab', '', 'C', '--', 'aB!cD', 'A1b2C3']
    input_text = '\n'.join(lines) + '\n'
    expected_lines = ref_alternating_case_lines([l + '\n' for l in lines])
    expected_output = ''.join(expected_lines)
    # CLI direct invocation: main
    sys.stdin = io.StringIO(input_text)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    rc_stdin = main([])
    sys.stdout.seek(0)
    sys.stderr.seek(0)
    out_stdin = sys.stdout.read()
    err_stdin = sys.stderr.read()
    assert rc_stdin == 0
    assert out_stdin == expected_output
    assert err_stdin == ''
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf8', newline='') as f:
        f.write(input_text)
        f.flush()
        fname = f.name
    try:
        rc_file = main([fname])
        sys.stdout.seek(0)
        sys.stderr.seek(0)
        out_file = sys.stdout.read()
        err_file = sys.stderr.read()
    finally:
        os.unlink(fname)
    assert rc_file == 0
    assert out_file == expected_output
    assert err_file == ''

def test_unicode_and_multilanguage():
    lines = [
        '\u0391a\u0392b\u0393g\u0394d\n',     # Greek and Latin with alternation
        '\u03b1\u0392\u03b3\u0394\n',         # Greek: alternate lower/UPPER
        '\u0431\u0416\u0433\u0417\n',         # Cyrillic, upper/lower upper/lower (should altern)
        '\u042b\u044b\u0429\u0449\n',         # Cyrillic, upper/lower upper/lower (should altern)
        '\u0414\u0415\u0416\n',          # All uppercase (no alternation)
        '\u0414\u0435\u0416\n',          # Upper/lower/upper (should altern)
        '\u03a3\n',            # Single Greek letter, should always be output
        '!',              # Non-alphabetic only, should output
        '\n'              # Empty line
    ]
    input_text = ''.join(lines)
    expected_lines = ref_alternating_case_lines(lines)
    expected_output = ''.join(expected_lines)
    sys.stdin = io.StringIO(input_text)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    rc = main([])
    sys.stdout.seek(0)
    sys.stderr.seek(0)
    out = sys.stdout.read()
    err = sys.stderr.read()
    assert rc == 0
    assert out == expected_output
    assert err == ''
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf8', newline='') as f:
        f.write(input_text)
        f.flush()
        fname = f.name
    try:
        rc_file = main([fname])
        sys.stdout.seek(0)
        sys.stderr.seek(0)
        out_file = sys.stdout.read()
        err_file = sys.stderr.read()
    finally:
        os.unlink(fname)
    assert rc_file == 0
    assert out_file == expected_output
    assert err_file == ''

def test_file_not_found_error():
    not_a_file = 'this_file_does_not_exist_12345.txt'
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    rc = main([not_a_file])
    sys.stdout.seek(0)
    sys.stderr.seek(0)
    out = sys.stdout.read()
    err = sys.stderr.read()
    assert rc == 2
    assert out == ''
    assert err == f'Error: file not found: {not_a_file}\n'

def test_treated_two_alpha():
    for line_end in ('\n', '\r\n', ''):
        input_text = f'N/A{line_end}'
        expected = f'N/A{line_end}' if input_text else ''
        sys.stdin = io.StringIO(input_text)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        rc = main([])
        sys.stdout.seek(0)
        sys.stderr.seek(0)
        out = sys.stdout.read()
        err = sys.stderr.read()
        assert rc == 0
        assert out == expected
        assert err == ''
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf8', newline='') as f:
            f.write(input_text)
            f.flush()
            fname = f.name
        try:
            rc_file = main([fname])
            sys.stdout.seek(0)
            sys.stderr.seek(0)
            out_file = sys.stdout.read()
            err_file = sys.stderr.read()
        finally:
            os.unlink(fname)
        assert rc_file == 0
        assert out_file == expected
        assert err_file == ''

def test_large_and_edge_cases():
    lines = [
        'A' * 1000 + 'b' * 1000 + '\n',    # 2000 letters, alternation from upper to lower (not alternating)
        'A' * 1000 + '\n',                 # single alpha, should output
        '1' * 1000 + '\n',                 # no alpha, should output
        'aB' + 'x' * 1000 + 'C' + 'y' * 1000 + 'd\n',  # Only a,B,C,d are alpha, alternation checked across them
        'ab\n',                            # 2 alpha, allowed
        '\n'
    ]
    input_text = ''.join(lines)
    expected_lines = ref_alternating_case_lines(lines)
    expected_output = ''.join(expected_lines)
    sys.stdin = io.StringIO(input_text)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    rc = main([])
    sys.stdout.seek(0)
    sys.stderr.seek(0)
    out = sys.stdout.read()
    err = sys.stderr.read()
    assert rc == 0
    assert out == expected_output
    assert err == ''
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf8', newline='') as f:
        f.write(input_text)
        f.flush()
        fname = f.name
    try:
        rc_file = main([fname])
        sys.stdout.seek(0)
        sys.stderr.seek(0)
        out_file = sys.stdout.read()
        err_file = sys.stderr.read()
    finally:
        os.unlink(fname)
    assert rc_file == 0
    assert out_file == expected_output
    assert err_file == ''
