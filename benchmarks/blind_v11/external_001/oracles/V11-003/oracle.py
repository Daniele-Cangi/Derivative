import io
import sys
import builtins
import pytest
from ascii_circle_cli import main

# Helper: Computes the expected ASCII circle outline as a string
def ascii_circle_outline(R):
    size = 2 * R + 1
    lines = []
    for y in range(size):
        line = ''
        for x in range(size):
            if abs((x - R) ** 2 + (y - R) ** 2 - R ** 2) <= R:
                line += '*'
            else:
                line += ' '
        lines.append(line)
    return '\n'.join(lines) + '\n'

@pytest.mark.parametrize("args, expected_out, expected_code", [
    (['0'], '*\n', 0),
    (['2'], ascii_circle_outline(2), 0),
    (['3'], ascii_circle_outline(3), 0),
])
def test_circle_cli_success(args, expected_out, expected_code, monkeypatch):
    captured = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured)
    retcode = main(args)
    sys.stdout = sys.__stdout__
    assert captured.getvalue() == expected_out
    assert retcode == expected_code

def test_circle_cli_error_missing_argument(monkeypatch):
    captured = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured)
    retcode = main([])
    sys.stdout = sys.__stdout__
    assert captured.getvalue() == 'ERROR\n'
    assert retcode == 2

def test_circle_cli_error_invalid_argument(monkeypatch):
    for bad in ['-1', '00', '+1', '1 ', ' 1', 'abc', '', '1.5', '2e1', '--help', '\t', '01']:
        captured = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', captured)
        retcode = main([bad])
        sys.stdout = sys.__stdout__
        assert captured.getvalue() == 'ERROR\n'
        assert retcode == 2

def test_circle_cli_error_extra_arguments(monkeypatch):
    captured = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured)
    retcode = main(['1', '2'])
    sys.stdout = sys.__stdout__
    assert captured.getvalue() == 'ERROR\n'
    assert retcode == 2

def test_circle_output_line_lengths_and_no_trailing_whitespace(monkeypatch):
    R = 5
    captured = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured)
    retcode = main([str(R)])
    sys.stdout = sys.__stdout__
    output = captured.getvalue()
    lines = output.split('\n')
    expected_length = 2 * R + 1
    # Last element after split could be '', so exclude it if so
    if lines and lines[-1] == '':
        lines = lines[:-1]
    assert all(len(line) == expected_length for line in lines)
    assert all(not line.endswith(' ') for line in lines)
    assert len(lines) == expected_length
    assert retcode == 0

def test_circle_cli_no_stderr(monkeypatch):
    R = 4
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_out)
    monkeypatch.setattr(sys, 'stderr', captured_err)
    retcode = main([str(R)])
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    assert captured_err.getvalue() == ''
    assert retcode == 0
