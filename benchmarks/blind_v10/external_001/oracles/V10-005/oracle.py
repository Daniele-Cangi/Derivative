import pytest
import sys
import io
from forge_bench_slot5 import main

# Helper to test CLI entrypoint
class CaptureStd:
    def __enter__(self):
        self._stdout = sys.stdout
        self.buf = io.StringIO()
        sys.stdout = self.buf
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
    def get(self):
        return self.buf.getvalue()

def cyclic_repeat_truncate(s, n):
    repeat_count = (n + len(s) - 1) // len(s)
    result = (s * repeat_count)[:n]
    return result

# Test with valid input, checking cyclic repetition and truncation
@pytest.mark.parametrize("n, letters", [
    (7, 'xyz'),
    (5, 'a'),
    (10, 'abcde'),
])
def test_valid_inputs(n, letters):
    argv = [str(n), letters]
    expected = cyclic_repeat_truncate(letters, n) + '\n'
    with CaptureStd() as cap:
        status = main(argv)
    assert cap.get() == expected
    assert status == 0

# Edge: N equals string length, output is just the string
@pytest.mark.parametrize("letters", ['w', 'abc', 'zxyw'])
def test_n_equals_string_len(letters):
    N = len(letters)
    argv = [str(N), letters]
    expected = letters + '\n'
    with CaptureStd() as cap:
        status = main(argv)
    assert cap.get() == expected
    assert status == 0

# Invalid: N has leading zero(s), N not integer, N too large, N zero, etc.
@pytest.mark.parametrize("argv", [
    ["01", "abc"],   # leading zero
    ["0002", "ab"],  # more leading zeros
    ["0", "a"],      # zero not allowed
    ["10000", "a"],  # N > 9999
    ["-5", "abc"],   # negative number
    ["3.5", "ab"],   # not integer
    ["abc", "xyz"],  # not a number at all
    ["7.0", "abc"],  # looks like float
])
def test_invalid_n(argv):
    with CaptureStd() as cap:
        status = main(argv)
    assert cap.get() == 'error\n'
    assert status == 1

# Invalid: string out of bounds, or N less than string len
@pytest.mark.parametrize("n, letters", [
    (2, 'xyz'),    # string longer than N
    (1, ''),       # string too short (empty)
    (5, ''),       # string empty even with N > 1
])
def test_invalid_length_and_empty(n, letters):
    argv = [str(n), letters]
    with CaptureStd() as cap:
        status = main(argv)
    assert cap.get() == 'error\n'
    assert status == 1

# Invalid: letters contains invalid characters
@pytest.mark.parametrize("letters", [
    'abcD',        # uppercase letter
    'a_b',         # underscore
    'a1b',         # digit
    'hello!',      # punctuation
    'a b',         # space
])
def test_invalid_letters(letters):
    n = max(3, len(letters))
    argv = [str(n), letters]
    with CaptureStd() as cap:
        status = main(argv)
    assert cap.get() == 'error\n'
    assert status == 1

# Invalid: missing or extra arguments
@pytest.mark.parametrize("argv", [
    ["3"],                 # missing second arg
    [],                     # missing both
    ["10", "abc", "x"],   # too many args
])
def test_argument_count(argv):
    with CaptureStd() as cap:
        status = main(argv)
    assert cap.get() == 'error\n'
    assert status == 1
