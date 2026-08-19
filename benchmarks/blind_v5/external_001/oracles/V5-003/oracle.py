# oracle for V5-003 (pyutfinvert.invert_case_preserve_nonletters)
import pytest
from pyutfinvert import invert_case_preserve_nonletters
import unicodedata

def reference_invert_case_preserve_nonletters(text):
    out = []
    for c in text:
        if c.isalpha():
            if c.isupper():
                out.append(c.lower())
            elif c.islower():
                out.append(c.upper())
            else:
                # Letter but neither isupper nor islower (uncommon, but possible)
                out.append(c)
        else:
            out.append(c)
    return ''.join(out)

def test_ascii_inversion_and_nonletters():
    inp = 'Hello, World! 1234.'
    expect = reference_invert_case_preserve_nonletters(inp)
    result = invert_case_preserve_nonletters(inp)
    assert result == expect

def test_nonascii_unicode_inversion():
    # Letters from various scripts with mixed case and combining marks
    inp = 'ΑλΦα ΒήΤΑ кирИЛлица ΔοκιΜΉ טקסט עִברִית üÖâğŞİ'
    expect = reference_invert_case_preserve_nonletters(inp)
    result = invert_case_preserve_nonletters(inp)
    assert result == expect


def test_preserves_non_letters_and_symbols():
    inp = '\t\n!@#[]()_+-=1234;:.,<>?/\\|'
    expect = reference_invert_case_preserve_nonletters(inp)
    result = invert_case_preserve_nonletters(inp)
    assert result == expect  # should be unchanged


def test_empty_string_returns_empty():
    inp = ''
    expect = ''
    result = invert_case_preserve_nonletters(inp)
    assert result == expect


def test_only_uppercase_unicode_letters():
    inp = 'ABCÇĞİÖŞÜЖЮЯÞЛŠŽÄÉÍÓÚ'
    expect = reference_invert_case_preserve_nonletters(inp)
    result = invert_case_preserve_nonletters(inp)
    # Must all become lowercase
    assert result == expect
    # Additional check: all alphabetic input -> all chars are lower if initially upper
    for i, c in enumerate(inp):
        if c.isupper():
            assert result[i] == c.lower()


def test_only_lowercase_unicode_letters():
    inp = 'abcçğıöşüжюяþлšžäéíóú'
    expect = reference_invert_case_preserve_nonletters(inp)
    result = invert_case_preserve_nonletters(inp)
    # Must all become uppercase
    assert result == expect
    for i, c in enumerate(inp):
        if c.islower():
            assert result[i] == c.upper()


def test_letter_neither_upper_nor_lower():
    # Some scripts' letters may return True for isalpha but False for both isupper and islower
    # Using '' (Thai char ก)
    inp = '\u0E01\u0E2D'  # กอ, two Thai letters
    assert all(c.isalpha() and not c.isupper() and not c.islower() for c in inp)
    # Should remain unchanged
    expect = inp
    result = invert_case_preserve_nonletters(inp)
    assert result == expect


def test_no_letters():
    inp = '1234567890!@#$%^&*()'
    expect = inp  # original string, must be unchanged
    result = invert_case_preserve_nonletters(inp)
    assert result == expect


def test_mixed_kind_letters():
    # String mixing upper/lower/noncase unicode letter, digits, and symbols
    inp = 'AbCĞğİıЖжЛлעִברִית-1234-ΣσςΣȺȾȿſΩωṃṁŊŋ'
    expect = reference_invert_case_preserve_nonletters(inp)
    result = invert_case_preserve_nonletters(inp)
    assert result == expect
    # Check digits and dashes are unchanged
    for i, c in enumerate(inp):
        if c.isdigit() or c == '-':
            assert result[i] == c
    # Letters must invert case if possible
    for i, c in enumerate(inp):
        if c.isupper():
            assert result[i] == c.lower()
        elif c.islower():
            assert result[i] == c.upper()


def test_typeerror_on_non_str_input():
    bad_inputs = [None, 100, 3.14, ['A', 'b'], {'foo': 'bar'}, (b'A',), True, b'ABC']
    for bad in bad_inputs:
        with pytest.raises(TypeError) as e:
            invert_case_preserve_nonletters(bad)
        # Exception message should mention 'str'
        assert 'str' in str(e.value)
