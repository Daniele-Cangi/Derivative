import io
import sys
import pytest
from forge_bench_slot6 import main
from contextlib import redirect_stdout

# --- Utility for in-process CLI capture ---
def run_cli(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(list(args))
    return rc, buf.getvalue()

# --- Independent reference minimal palindrome solution (only for fixed testcases) ---
def reference_shortest_palindrome(s):
    # For contract, only use fixed, known testcases not dependent on candidate
    # Used in test with expected literals only
    L = len(s)
    for i in range(L):
        suffix = s[i:]
        if suffix == suffix[::-1]:
            return s + s[:i][::-1]
    # fallback (shouldn't be needed)
    return s + s[::-1][1:]

def is_palindrome(s):
    return s == s[::-1]

# --- Test Cases: Palindromic Inputs ---
@pytest.mark.parametrize("value", [
    "a",
    "aa",
    "aba",
    "racecar",
    "level",
    "m",
    "noon",
    "zzzzz",
    "abcdefghhgfedcba"
])
def test_already_palindrome(value):
    rc, out = run_cli([value])
    assert rc == 0
    assert out == value + "\n"
    assert is_palindrome(value)

# --- Test Cases: Construct Shortest Palindrome (explicit expectation) ---
def test_shortest_palindrome_abcdef():
    inp = "abcdef"
    # Suffix-palindrome check: suffixes are [abcdef, bcdef, cdef, def, ef, f]
    # Only f is palindrome at i=5.
    expected = 'abcdefedcba' # full reverse except first character
    rc, out = run_cli([inp])
    assert rc == 0
    assert out == expected + '\n'
    # Manually checked, this is minimal
    assert is_palindrome(out.strip())
    assert out.startswith(inp)

def test_shortest_palindrome_abac():
    inp = "abac"
    # suffixes: abac, bac, ac, c
    # c is palindrome at i=3, need abac + 'ba' = abacba
    expected = 'abacaba'
    rc, out = run_cli([inp])
    assert rc == 0
    assert out == expected + '\n'
    assert is_palindrome(out.strip())
    assert out.startswith(inp)

def test_shortest_palindrome_abcd():
    inp = "abcd"
    # suffixes: abcd, bcd, cd, d. Only d is palindrome
    # Result: "abcd" + "cba" = "abcdcba"
    expected = "abcdcba"
    rc, out = run_cli([inp])
    assert rc == 0
    assert out == expected + '\n'
    assert is_palindrome(out.strip())
    assert out.startswith(inp)

# --- Edge case: single-letter all valid, must print as-is ---
@pytest.mark.parametrize("letter", list('abcdefghijklmnopqrstuvwxyz'))
def test_single_letter(letter):
    rc, out = run_cli([letter])
    assert rc == 0
    assert out == letter + "\n"
    assert is_palindrome(out.strip())

# --- Edge case: maximal valid input (50 lowercase letters) ---
def test_max_length_fifty_a():
    inp = 'a'*50
    rc, out = run_cli([inp])
    assert rc == 0
    assert out == inp + "\n"
    assert is_palindrome(out.strip())
    assert out.startswith(inp)
    assert len(inp) == 50

def test_max_length_abc_pattern():
    inp = 'abc' * 16 + 'a' # len = 49
    inp += 'b' # len = 50
    assert len(inp) == 50
    # Not palindromic; must compute expected output
    # Suffixes: try for minimal; by analysis, needs full reverse except first char
    expected = inp + inp[:-1][::-1]
    rc, out = run_cli([inp])
    assert rc == 0
    assert out == expected + '\n'
    assert is_palindrome(out.strip())
    assert out.startswith(inp)
    assert len(out.strip()) == len(expected)

# --- Invalid argument cases (output and rc must be exact) ---
@pytest.mark.parametrize("bad_input", [
    "",
    "A",
    "abcD",
    "abc d",
    " abc",
    "abc ",
    "abc1",
    "1abc",
    "ab@c",
    '!@#',
    'Abc',
    'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabc', # 53 chars
    # >50 characters
    'a'*51,
    'abc\n',
    '-',
    '_',
])
def test_invalid_input_characters(bad_input):
    rc, out = run_cli([bad_input])
    assert rc == 1
    assert out == 'error\n'

# --- Invalid argument count (none or too many) ---
@pytest.mark.parametrize("argv", [
    [],
    ['a', 'b'],
    ['abc', 'def'],
    [''],
])
def test_invalid_argv_count(argv):
    rc, out = run_cli(argv)
    assert rc == 1
    assert out == 'error\n'
