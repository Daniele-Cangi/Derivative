import pytest
from collapse_whitespace_runs import collapse_whitespace_runs

def test_basic_unicode_and_ascii_whitespace_behavior():
    # Requirement example (1): ['foo\t\tbar', ' a\nb\r\tc '] returns ['foo bar', 'a b c']
    input_data = ['foo\t\tbar', ' a\nb\r\tc ']
    expected = ['foo bar', 'a b c']
    assert collapse_whitespace_runs(input_data) == expected

def test_empty_and_trivial_strings_behavior():
    # Requirement example (2): ['   ', '','foo','bar\tbaz'] returns ['', '', 'foo', 'bar baz']
    input_data = ['   ', '', 'foo', 'bar\tbaz']
    expected = ['', '', 'foo', 'bar baz']
    assert collapse_whitespace_runs(input_data) == expected

def test_unicode_whitespace_collapse():
    # Requirement example (3): ['\u2003A\u2009B'] returns ['A B']
    input_data = ['\u2003A\u2009B']  # em space, thin space between A/B
    expected = ['A B']
    assert collapse_whitespace_runs(input_data) == expected

def test_typeerror_for_non_str_elements():
    # Requirement (4): Input ['foo', 123, 'bar'] raises TypeError
    with pytest.raises(TypeError):
        collapse_whitespace_runs(['foo', 123, 'bar'])

def test_typeerror_for_non_list_input():
    for bad in (None, 'foobar', 42, {'a': 1}):
        with pytest.raises(TypeError):
            collapse_whitespace_runs(bad)

def test_only_whitespace_variants():
    # Strings that are only whitespace must become empty
    input_data = ['   ', '\n\t\r', '\u2005\u2028', '\u2003']
    expected = ['', '', '', '']
    assert collapse_whitespace_runs(input_data) == expected

def test_mixed_whitespace_long_runs():
    # Maximal runs and internal non-whitespace preserved
    input_data = [
        'foo  \t\u2003 bar',            # multiple different whitespaces
        'baz\n   qux',                  # newline and spaces
        '\t\nfoo\u2028   bar\u2005',  # mix with unicode line sep, space
        'alpha beta',                    # already single space
        '   lead and trail   ',          # leading/trailing and internal
    ]
    # Each string processed independently, all whitespace runs become a single ASCII space, leading/trailing removed
    expected = [
        'foo bar',
        'baz qux',
        'foo bar',
        'alpha beta',
        'lead and trail'
    ]
    assert collapse_whitespace_runs(input_data) == expected

def test_leading_and_trailing_whitespace_removed():
    input_data = [
        '   foo',
        'bar   ',
        '\tfoobar\n',
        '\u2005foo\u2003',
        '\n\t',
    ]
    expected = [
        'foo',
        'bar',
        'foobar',
        'foo',
        ''
    ]
    assert collapse_whitespace_runs(input_data) == expected

def test_all_whitespace_string_becomes_empty():
    input_data = ['    ', '\n', '\t', '\u2009', '\u2028']
    expected = ['', '', '', '', '']
    assert collapse_whitespace_runs(input_data) == expected

def test_preserves_non_whitespace_runs():
    input_data = ['abc', 'x y', 'foo\u2005bar', '\tquux']
    expected = ['abc', 'x y', 'foo bar', 'quux']
    assert collapse_whitespace_runs(input_data) == expected

# End of oracle
