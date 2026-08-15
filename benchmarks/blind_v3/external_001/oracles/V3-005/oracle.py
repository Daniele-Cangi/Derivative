import pytest
import io
import sys
import builtins

from yourmodule import main  # Replace 'yourmodule' and 'main' with actual module and function names

@ pytest.fixture
def capsys_unicode():
    # This fixture provides capturing of stdout/stderr with unicode support
    return pytest.CaptureFixture[str]

@ pytest.fixture
def input_lines():
    return [
        "1 + 2 * 3",             # simple valid expression
        "10 // 3 - 1",          # includes floor division and subtraction
        "4 + 5 * 6 - 7 // 2",   # multiple operators
        "42",                   # single number
        "10 // 0",              # division by zero edge case
        "1 + foo",              # invalid token
        "2 ** 3",               # invalid operator
        "",                     # empty line (should handle gracefully or produce error)
        "   7 - 5  "            # expression with spaces
    ]


def run_main_with_input_lines(lines):
    input_stream = io.StringIO("\n".join(lines) + "\n")
    # Patch sys.stdin to our input_stream
    original_stdin = sys.stdin
    sys.stdin = input_stream
    try:
        main()
    finally:
        sys.stdin = original_stdin


def test_valid_expressions_output(capsys, input_lines):
    # Only pass valid lines: 0,1,2,3,8
    valid_lines = [input_lines[i] for i in [0,1,2,3,8]]

    def filter_valid(line):
        try:
            # Try to eval to compare results
            tokens = line.split()
            # Only accept valid tokens
            valid_ops = {'+', '-', '*', '//'}
            for token in tokens:
                if not (token.isdigit() or token in valid_ops):
                    raise ValueError("Invalid token")
            # Evaluate using python's eval with safe replacement of floor division
            temp_expr = line.replace('//', '//')
            expected = eval(temp_expr, {}, {})
            return expected
        except Exception:
            return None

    lines_to_test = []
    expected_outputs = []
    for l in valid_lines:
        expected = filter_valid(l)
        if expected is not None:
            lines_to_test.append(l)
            expected_outputs.append(str(expected))

    input_stream = io.StringIO("\n".join(lines_to_test) + "\n")
    original_stdin = sys.stdin
    sys.stdin = input_stream
    try:
        main()
    finally:
        sys.stdin = original_stdin

    captured = capsys.readouterr()
    # Results should be output line-by-line matching expected
    output_lines = captured.out.strip().split('\n')
    assert output_lines == expected_outputs


def test_invalid_tokens_report_line_numbers(capsys):
    lines = [
        "1 + 2",
        "3 + badtoken",   # line 2 invalid token
        "4 - 1",
        "5 ** 2",        # line 4 invalid operator
    ]
    input_stream = io.StringIO("\n".join(lines) + "\n")
    original_stdin = sys.stdin
    sys.stdin = input_stream
    try:
        main()
    finally:
        sys.stdin = original_stdin

    captured = capsys.readouterr()
    # Error messages should be printed mentioning the line numbers
    # Usually we expect errors for line 2 and 4 only
    err_msg = captured.err or captured.out
    assert "2" in err_msg and "invalid" in err_msg.lower()
    assert "4" in err_msg and "invalid" in err_msg.lower()


def test_division_by_zero_handling(capsys):
    lines = ["10 // 0"]
    input_stream = io.StringIO("\n".join(lines) + "\n")
    original_stdin = sys.stdin
    sys.stdin = input_stream
    try:
        main()
    finally:
        sys.stdin = original_stdin

    captured = capsys.readouterr()
    # Should report an error mentioning line 1 and division by zero
    err_msg = captured.err or captured.out
    assert "1" in err_msg and ("zero" in err_msg.lower() or "division" in err_msg.lower())
    # No output lines or maybe an error line only
    assert captured.out.strip() == "" or "error" in captured.out.lower()


def test_empty_and_whitespace_handling(capsys):
    lines = ["", "    ", "7 - 5"]
    input_stream = io.StringIO("\n".join(lines) + "\n")
    original_stdin = sys.stdin
    sys.stdin = input_stream
    try:
        main()
    finally:
        sys.stdin = original_stdin

    captured = capsys.readouterr()
    # Should ignore or error on empty lines, output only one result for last line
    output_lines = [line for line in captured.out.strip().split('\n') if line.strip()]
    # Last line 3 expression "7 - 5" = 2
    assert any(line.strip() == '2' for line in output_lines) or "error" in captured.out.lower() or "invalid" in captured.out.lower()


# End of tests
