import io
import os
import sys
import tempfile
import pytest
from forge_blind_v7.cli_rotate_words import main

# Utility to capture stdout/stderr by swapping sys.{stdout,stderr}
from contextlib import contextmanager

@contextmanager
def swap_stdio(stdin=None):
    old_in, old_out, old_err = sys.stdin, sys.stdout, sys.stderr
    in_stream = stdin if stdin is not None else sys.stdin
    out_stream = io.StringIO()
    err_stream = io.StringIO()
    sys.stdin, sys.stdout, sys.stderr = in_stream, out_stream, err_stream
    try:
        yield out_stream, err_stream
    finally:
        sys.stdin, sys.stdout, sys.stderr = old_in, old_out, old_err

def expect_rotated(line):
    # Source-independent expectation generator for the contract
    s = line.strip()
    if s == '':
        return ''
    words = s.split()
    if not words:
        return ''
    if len(words) == 1:
        return words[0]
    # right rotate by 1
    rotated = [words[-1]] + words[:-1]
    return ' '.join(rotated)

@pytest.mark.parametrize("input_line", [
    "the quick brown",           # Multiple words
    "   sp ace  ",              # Multiple words with leading/trailing/multiple spaces
    " måne øl sol ",            # Unicode words
    "word",                     # Single word
    "   ",                      # Only whitespace
    "",                         # Empty line
    "\u2002foo\u2003bar\u2004baz\u2005",    # Unicode whitespace
])
def test_rotate_words_single_lines(input_line):
    # Prepare input/output for one line
    inp = input_line + "\n"
    with swap_stdio(io.StringIO(inp)) as (out, err):
        status = main([])
    assert status == 0
    # Output one line, maintain newline
    lines = out.getvalue().splitlines()
    assert len(lines) == 1
    assert lines[0] == expect_rotated(input_line)
    assert err.getvalue() == ''


def test_rotate_words_multiple_lines():
    input_lines = [
        '',          # Empty line
        '   ',       # Whitespace only
        'abc',       # Single word
        'the quick brown',   # Multi word
        'måne øl sol',       # Unicode
        '   sp ace  '       # Whitespace + multi word
    ]
    inp = '\n'.join(input_lines) + '\n'
    with swap_stdio(io.StringIO(inp)) as (out, err):
        status = main([])
    assert status == 0
    output_lines = out.getvalue().splitlines()
    expected_lines = [expect_rotated(line) for line in input_lines]
    assert output_lines == expected_lines
    assert err.getvalue() == ''


def test_rotate_words_empty_input():
    with swap_stdio(io.StringIO('')) as (out, err):
        status = main([])
    assert status == 0
    assert out.getvalue() == ''
    assert err.getvalue() == ''


def test_file_input_and_output(tmp_path):
    lines = [
        'måne øl sol',
        '   ',
        'the quick brown',
        'foo'
    ]
    file_contents = '\n'.join(lines) + '\n'
    test_file = tmp_path / "test_unicode.txt"
    test_file.write_text(file_contents, encoding="utf-8")
    with swap_stdio() as (out, err):
        status = main([str(test_file)])
    assert status == 0
    output_lines = out.getvalue().splitlines()
    expected_lines = [expect_rotated(line) for line in lines]
    assert output_lines == expected_lines
    assert err.getvalue() == ''


def test_unicode_whitespace_and_characters():
    # Mix of tabs, em space, non-break space, etc.
    ws = '\t\u2003\u2002'  # tab, em space, en space
    inp = f"{ws}a{ws}b  c{ws}  \n"
    # This is equivalent to "a b c" split
    with swap_stdio(io.StringIO(inp)) as (out, err):
        status = main([])
    assert status == 0
    exp = expect_rotated("a b c")
    lines = out.getvalue().splitlines()
    assert lines[0] == exp
    assert err.getvalue() == ''


def test_file_not_found(tmp_path):
    missing = tmp_path / "no_such_file.txt"
    with swap_stdio() as (out, err):
        status = main([str(missing)])
    # Per contract: exit 2, write message to stderr
    assert status == 2
    # Output must be empty
    assert out.getvalue() == ''
    # There should be a clear message
    msg = err.getvalue()
    assert msg.strip() != ''
    assert 'no_such_file' in msg or 'No such file' in msg or 'not found' in msg or 'No such' in msg


def test_extra_argument_is_ignored(tmp_path):
    # Should only use first arg as filename, ignore extra.
    f = tmp_path / "file.txt"
    f.write_text('foo bar\n', encoding='utf-8')
    with swap_stdio() as (out, err):
        status = main([str(f), "extra.txt"])  # Only first arg used per contract
    assert status == 0
    lines = out.getvalue().splitlines()
    assert lines == [expect_rotated('foo bar')]
    assert err.getvalue() == ''


def test_large_unicode_line(tmp_path):
    words = [chr(i) for i in range(0x391, 0x3a9)] * 100  # Greek caps repeated
    line = ' \t\u2002 '.join(words)
    inp = line + '\n'
    tmpfile = tmp_path / "greek.txt"
    tmpfile.write_text(inp, encoding="utf-8")
    with swap_stdio() as (out, err):
        status = main([str(tmpfile)])
    assert status == 0
    # Split and rotate, reconstruct expected line
    norm_words = words
    exp_line = ' '.join([norm_words[-1]] + norm_words[:-1])
    lines = out.getvalue().splitlines()
    assert lines[0] == exp_line
    assert err.getvalue() == ''
