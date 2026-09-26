import pytest
import io
import sys
import builtins
from forge_bench_slot3 import main


def staircase_ways(n):
    # Dynamic programming: ways[n] = ways[n-1] + ways[n-2] + ways[n-5], base case for n < 0
    if n < 0:
        return 0
    ways = [0] * (n + 6)
    ways[0] = 1
    for i in range(1, n + 1):
        ways[i] = ways[i - 1] + ways[i - 2] + ways[i - 5]
    return ways[n]


def run_cli(argv):
    # Captures stdout and exit code of main(argv)
    out = io.StringIO()
    old_stdout = sys.stdout
    exitcode = None
    try:
        sys.stdout = out
        exitcode = main(argv)
        return (out.getvalue(), exitcode)
    finally:
        sys.stdout = old_stdout


def test_valid_input_typical():
    """
    For N=5: Expected output is '8\n'.
    """
    argv = ['5']
    expected = str(staircase_ways(5)) + '\n'
    output, exitcode = run_cli(argv)
    assert output == expected
    assert exitcode == 0


def test_valid_input_zero():
    """
    For N=0: Expected output is '1\n' (one way to take zero steps).
    """
    argv = ['0']
    expected = str(staircase_ways(0)) + '\n'
    output, exitcode = run_cli(argv)
    assert output == expected
    assert exitcode == 0


def test_valid_input_max():
    """
    For N=1000: Works for large n, output is deterministic and correct.
    """
    argv = ['1000']
    expected = str(staircase_ways(1000)) + '\n'
    output, exitcode = run_cli(argv)
    assert output == expected
    assert exitcode == 0


def test_invalid_negative():
    """
    Negative integers are invalid. Triggers error path.
    """
    argv = ['-1']
    output, exitcode = run_cli(argv)
    assert output == 'error\n'
    assert exitcode == 1


def test_invalid_nonint():
    """
    Non-integer input is invalid. Triggers error path.
    """
    argv = ['abc']
    output, exitcode = run_cli(argv)
    assert output == 'error\n'
    assert exitcode == 1


def test_invalid_leading_zero():
    """
    Leading zeros except '0' itself are invalid (e.g., '007', '01').
    """
    argv = ['01']
    output, exitcode = run_cli(argv)
    assert output == 'error\n'
    assert exitcode == 1
    argv = ['007']
    output, exitcode = run_cli(argv)
    assert output == 'error\n'
    assert exitcode == 1


def test_invalid_too_large():
    """
    N above 1000 is invalid.
    """
    argv = ['1001']
    output, exitcode = run_cli(argv)
    assert output == 'error\n'
    assert exitcode == 1


def test_invalid_empty_arg():
    """
    Empty string is not a valid argument.
    """
    argv = ['']
    output, exitcode = run_cli(argv)
    assert output == 'error\n'
    assert exitcode == 1


def test_invalid_multiple_args():
    """
    Only one positional argument allowed; extra args should trigger error.
    """
    argv = ['5', '6']
    output, exitcode = run_cli(argv)
    assert output == 'error\n'
    assert exitcode == 1
