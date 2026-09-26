import io
import sys
import pytest
from cyclic_bigram_set_cli import main

# Helper context manager to capture exactly sys.stdout and sys.stderr
class CapOutErr:
    def __enter__(self):
        self._oldout = sys.stdout
        self._olderr = sys.stderr
        self.out = io.StringIO()
        self.err = io.StringIO()
        sys.stdout = self.out
        sys.stderr = self.err
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._oldout
        sys.stderr = self._olderr
    def outval(self):
        return self.out.getvalue()
    def errval(self):
        return self.err.getvalue()

# Reference to compute the cyclic bigram set, deduplicate, and lexicographically sort
# (literal computation for each concrete fixture only)
def compute_cyclic_bigrams(words):
    bigrams = set()
    for w in words:
        N = len(w)
        for i in range(N):
            b = w[i] + w[(i+1)%N]
            bigrams.add(b)
    return ','.join(sorted(bigrams))

# --- VALID CASES ---
def test_basic_success_two_simple_words():
    argv = ["cat,dog"]
    # 'cat' → ca, at, tc ; 'dog' → do, og, gd ; union, sorted
    w = ["cat", "dog"]
    expected = compute_cyclic_bigrams(w)
    with CapOutErr() as cap:
        code = main(argv)
    assert cap.outval() == expected + '\n'
    assert cap.errval() == ''
    assert code == 0

def test_success_with_overlap_and_three_letter_words():
    argv = ["abc,cab"]
    # 'abc': ab, bc, ca ; 'cab': ca, ab, bc ; so ab, bc, ca
    w = ["abc", "cab"]
    expected = compute_cyclic_bigrams(w)
    with CapOutErr() as cap:
        code = main(argv)
    assert cap.outval() == expected + '\n'
    assert cap.errval() == ''
    assert code == 0

def test_success_varied_lengths():
    argv = ["bat,flap"]
    # bat: ba, at, tb; flap: fl, la, ap, pf
    w = ["bat", "flap"]
    expected = compute_cyclic_bigrams(w)
    with CapOutErr() as cap:
        code = main(argv)
    assert cap.outval() == expected + '\n'
    assert cap.errval() == ''
    assert code == 0

# --- INVALID CASES ---
def test_invalid_short_word():
    # 'a' is too short
    argv = ["hi,a"]
    with CapOutErr() as cap:
        code = main(argv)
    assert cap.outval() == 'ERROR\n'
    assert cap.errval() == ''
    assert code == 2

def test_invalid_nonalpha_characters():
    # Contains digit
    argv1 = ["hey,ab1"]
    with CapOutErr() as cap1:
        code1 = main(argv1)
    assert cap1.outval() == 'ERROR\n'
    assert cap1.errval() == ''
    assert code1 == 2
    # Contains uppercase
    argv2 = ["az,BA"]
    with CapOutErr() as cap2:
        code2 = main(argv2)
    assert cap2.outval() == 'ERROR\n'
    assert cap2.errval() == ''
    assert code2 == 2
    # Contains whitespace
    argv3 = ["az,bc d"]
    with CapOutErr() as cap3:
        code3 = main(argv3)
    assert cap3.outval() == 'ERROR\n'
    assert cap3.errval() == ''
    assert code3 == 2

def test_invalid_empty_element_extra_comma():
    # empty word
    argv = ["foo,,bar"]
    with CapOutErr() as cap:
        code = main(argv)
    assert cap.outval() == 'ERROR\n'
    assert cap.errval() == ''
    assert code == 2

# --- EDGE/BOUNDARY CONDITIONS ---
def test_invalid_one_word():
    # Only one word is invalid
    argv = ["foobar"]
    with CapOutErr() as cap:
        code = main(argv)
    assert cap.outval() == 'ERROR\n'
    assert cap.errval() == ''
    assert code == 2

def test_missing_argument_and_too_many_args():
    # No argument
    argv1 = []
    with CapOutErr() as cap1:
        code1 = main(argv1)
    assert cap1.outval() == 'ERROR\n'
    assert cap1.errval() == ''
    assert code1 == 2
    # Too many arguments
    argv2 = ["ab,cd", "ef"]
    with CapOutErr() as cap2:
        code2 = main(argv2)
    assert cap2.outval() == 'ERROR\n'
    assert cap2.errval() == ''
    assert code2 == 2

def test_trailing_leading_commas_only():
    # Trailing comma (empty word at end)
    argv1 = ["ab,cd,"]
    with CapOutErr() as cap1:
        code1 = main(argv1)
    assert cap1.outval() == 'ERROR\n'
    assert cap1.errval() == ''
    assert code1 == 2
    # Leading comma (empty word at start)
    argv2 = [",ab,cd"]
    with CapOutErr() as cap2:
        code2 = main(argv2)
    assert cap2.outval() == 'ERROR\n'
    assert cap2.errval() == ''
    assert code2 == 2

def test_no_output_to_stderr_in_success_and_error():
    # Success case
    argv = ["loaf,bread"]
    with CapOutErr() as cap:
        code = main(argv)
    assert cap.errval() == ''
    assert code == 0
    # Error case
    argv2 = ["ok,123"]
    with CapOutErr() as cap2:
        code2 = main(argv2)
    assert cap2.errval() == ''
    assert code2 == 2
