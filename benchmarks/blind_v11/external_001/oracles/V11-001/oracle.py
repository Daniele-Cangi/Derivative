import sys
import io
import pytest
from divisors_cli import main

# Utility context to capture stdout
class CaptureStdout:
    def __enter__(self):
        self._original = sys.stdout
        self._stringio = io.StringIO()
        sys.stdout = self._stringio
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original
    def getvalue(self):
        return self._stringio.getvalue()

# Reference implementation for divisors, matches contract (abs(N), except 0)
def reference_divisors_str(n):
    if n == 0:
        return '0'
    n = abs(n)
    return ','.join(str(i) for i in sorted([d for d in range(1, n+1) if n % d == 0]))

def test_positive_integer():
    argv = ['prog', '12']
    with CaptureStdout() as cap:
        exit_code = main(argv)
    expected = reference_divisors_str(12) + '\n'
    assert cap.getvalue() == expected
    assert exit_code == 0

def test_zero_input():
    argv = ['prog', '0']
    with CaptureStdout() as cap:
        exit_code = main(argv)
    expected = '0\n'
    assert cap.getvalue() == expected
    assert exit_code == 0

def test_negative_integer():
    argv = ['prog', '-8']
    with CaptureStdout() as cap:
        exit_code = main(argv)
    expected = reference_divisors_str(8) + '\n'
    assert cap.getvalue() == expected
    assert exit_code == 0

def test_prime_input():
    argv = ['prog', '17']
    with CaptureStdout() as cap:
        exit_code = main(argv)
    expected = reference_divisors_str(17) + '\n'
    assert cap.getvalue() == expected
    assert exit_code == 0

def test_one_input():
    argv = ['prog', '1']
    with CaptureStdout() as cap:
        exit_code = main(argv)
    expected = '1\n'
    assert cap.getvalue() == expected
    assert exit_code == 0

def test_too_many_arguments():
    argv = ['prog', '1', '2']
    with CaptureStdout() as cap:
        exit_code = main(argv)
    assert cap.getvalue() == 'ERROR\n'
    assert exit_code == 2

def test_no_argument():
    argv = ['prog']
    with CaptureStdout() as cap:
        exit_code = main(argv)
    assert cap.getvalue() == 'ERROR\n'
    assert exit_code == 2

def test_non_integer_argument():
    argv = ['prog', 'abc']
    with CaptureStdout() as cap:
        exit_code = main(argv)
    assert cap.getvalue() == 'ERROR\n'
    assert exit_code == 2

def test_float_argument():
    argv = ['prog', '3.5']
    with CaptureStdout() as cap:
        exit_code = main(argv)
    assert cap.getvalue() == 'ERROR\n'
    assert exit_code == 2

def test_empty_string_argument():
    argv = ['prog', '']
    with CaptureStdout() as cap:
        exit_code = main(argv)
    assert cap.getvalue() == 'ERROR\n'
    assert exit_code == 2
