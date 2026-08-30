import io
import sys
import pytest
from forge_blind_v9_requirements_slot5 import main

# Reference sort: key function implements strict rules.
def reference_sort(line):
    def key(substr):
        if substr == 'INF':
            return (float('inf'),)
        else:
            # Strip leading zeros for integer value for sorting;
            # '000' --> 0, '007' -> 7, '' not possible (see validation)
            return (int(substr.lstrip('0') or '0'),)
    # lines already validated to have no leading/trailing/inner ws, and all substrings valid
    substrings = line.split(';')
    return ';'.join(sorted(substrings, key=key, reverse=True))

@pytest.fixture
def valid_lines():
    # Valid lines, all substrings are integers (with possible leading zeros) or exactly 'INF'
    # No extra whitespace anywhere
    return [
        '007;2;0002;INF;07',           # 7,2,2,INF,7
        '00000;00001;123;INF;12;00012',# 0,1,123,INF,12,12
        '5',                           # single substring (int)
        'INF',                         # single substring (INF)
        '0003;3;03;3;003',             # all are same integer (3)
        'INF;INF;002;001',             # duplicate INF
    ]

@pytest.fixture
def valid_expected(valid_lines):
    return [reference_sort(line) for line in valid_lines]

@pytest.fixture
def duplicate_preservation_lines():
    # Duplicates of INF, zeros, and arbitrary integers
    return [
        'INF;3;INF;3;3;INF',       # alternating INF/3
        '0000;0;0;0000;0',         # representations of 0
        '5;5;INF;5;INF;5;5',       # 5 and INF duplicates
    ]

@pytest.fixture
def duplicate_preservation_expected(duplicate_preservation_lines):
    return [reference_sort(line) for line in duplicate_preservation_lines]

@pytest.fixture
def whitespace_error_lines():
    # Any line with whitespace anywhere (leading, trailing, inner, or between substrings) is forbidden
    return [
        ' 8;9',           # leading whitespace
        '1;9 ',           # trailing whitespace
        '6 ;7',           # whitespace after substrings
        '1; 6',           # whitespace before substring
        '2;5; 5',         # whitespace before a duplicate
        '3;4;5\t;6',     # tab in field
        ' 0 ',            # whitespace both ends
        '\t4;5',         # tab leading
        '2;3;4;5 ',       # whitespace trailing
        '2;3 ;4'          # whitespace trailing field
    ]

@pytest.fixture
def invalid_encoding_bytes():
    # Input containing bytes not valid as UTF-8 for fail-fast test
    return [
        b'3;4;\xff\n',
        b'6;7\x80\n',
        b'1;INF\xfe\n',
    ]

@pytest.fixture
def non_numeric_or_inf_substring_lines():
    # Other invalid substrings: non-digits, not exactly 'INF'
    return [
        '12;foo;1',        # alpha
        '12;-7;8',         # negative not allowed
        '8;42.5;7',        # float
        '8;4e5;7',         # sci notation
        '01;INf',          # case mismatch
        '0;Inf;INF',       # case mismatch
        'INF;NAN',         # non-integer, non-INF
        'INF;0x7',         # not decimal
        '42;;43',          # empty substring (split emits '', which is forbidden per contract, since each substring is non-empty)
        ';7;8',            # leading semicolon yields empty substring
        '8;9;',            # trailing semicolon yields empty substring
    ]

# Helper to invoke CLI with custom sys.stdin and sys.stdout

def run_cli_with_stdin_bytes(stdin_bytes):
    import builtins
    orig_stdin = sys.stdin
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    try:
        # Use TextIOWrapper over BytesIO to simulate true UTF-8/bytes input
        sys.stdin = io.TextIOWrapper(io.BytesIO(stdin_bytes), encoding='utf-8', newline='')
        out = io.StringIO()
        sys.stdout = out
        sys.stderr = io.StringIO()  # Silence any error prints
        result = main([])
        sys.stdout.flush()
        output = out.getvalue()
        return result, output
    finally:
        sys.stdin = orig_stdin
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

# 1. Happy path: All valid input lines, expect correct reordering and code 0

def test_strictly_descending_sort_valid_cases(valid_lines, valid_expected):
    data = ('\n'.join(valid_lines) + '\n').encode('utf-8')
    exitcode, output = run_cli_with_stdin_bytes(data)
    expected = '\n'.join(valid_expected) + '\n'
    assert exitcode == 0
    assert output == expected

# 2. Duplicates (equal integer values or INF) must be preserved and sorted

def test_duplicate_value_preservation(duplicate_preservation_lines, duplicate_preservation_expected):
    data = ('\n'.join(duplicate_preservation_lines) + '\n').encode('utf-8')
    exitcode, output = run_cli_with_stdin_bytes(data)
    expected = '\n'.join(duplicate_preservation_expected) + '\n'
    assert exitcode == 0
    assert output == expected

# 3. Any whitespace in line triggers code 2 and no output

def test_any_whitespace_is_fatal(whitespace_error_lines):
    for line in whitespace_error_lines:
        data = (line + '\n').encode('utf-8')
        exitcode, output = run_cli_with_stdin_bytes(data)
        assert exitcode == 2
        assert output == ''

# 4. Any substring that is not a non-empty digit sequence or 'INF', or an empty substring, is fatal

def test_invalid_substring_errors(non_numeric_or_inf_substring_lines):
    for line in non_numeric_or_inf_substring_lines:
        data = (line + '\n').encode('utf-8')
        exitcode, output = run_cli_with_stdin_bytes(data)
        assert exitcode == 2
        assert output == ''

# 5. Invalid UTF-8 on any line is immediately fatal

def test_invalid_utf8_is_fatal(invalid_encoding_bytes):
    for bytes_line in invalid_encoding_bytes:
        exitcode, output = run_cli_with_stdin_bytes(bytes_line)
        assert exitcode == 2
        assert output == ''

# 6. On multiple lines, ANY invalid one is globally fatal and silences output

def test_global_fatal_on_any_invalid_line(valid_lines):
    # Prepare several lines, one invalid (with whitespace)
    good = valid_lines[0]
    bad = ' 99;100'  # leading whitespace is error
    lines = [good, bad, good]
    data = ('\n'.join(lines) + '\n').encode('utf-8')
    exitcode, output = run_cli_with_stdin_bytes(data)
    assert exitcode == 2
    assert output == ''
