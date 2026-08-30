import io
import sys
import pytest

from forge_blind_v9_requirements_slot2 import main

def ref_reduce_line(line: str) -> str:
    # If line is empty, return as is
    if line == '':
        return ''
    parts = line.split('-')
    if not parts:
        return ''
    result = []
    prev = None
    for p in parts:
        if p != prev:
            result.append(p)
        prev = p
    return '-'.join(result)

class DummyStream:
    # dummy to simulate input stream that raises UnicodeDecodeError
    def __init__(self, exc: Exception):
        self.exc = exc
    def readline(self, *_, **__):
        raise self.exc
    def __iter__(self):
        return self
    def __next__(self):
        raise self.exc
    def read(self, *_, **__):
        raise self.exc


def test_reduce_contiguous_duplicates_basic(monkeypatch, capsys):
    # Lines with contiguous, noncontiguous, and empty lines, and trailing newline preserved.
    user_input = 'a-a-b-b-b-c\nfoo-bar-baz\na-b-a-b\n\n\nabc-def-def-abc\n'  # includes two empty lines
    exp_lines = []
    for line in user_input.splitlines():
        exp_lines.append(ref_reduce_line(line))
    expected_output = '\n'.join(exp_lines) + '\n'  # preserve trailing newlines
    # Simulate stdin text
    monkeypatch.setattr('sys.stdin', io.StringIO(user_input))
    rc = main([])
    out, err = capsys.readouterr()
    assert rc == 0
    assert out == expected_output
    assert err == ''


def test_invalid_input_characters(monkeypatch, capsys):
    # Non-ASCII lowercase and bad characters
    user_input = 'valid-line\ninvalid-line-123\n'  # second line contains digits → error
    monkeypatch.setattr('sys.stdin', io.StringIO(user_input))
    rc = main([])
    out, err = capsys.readouterr()
    # Per contract: exit code 2 and output nothing
    assert rc == 2
    assert out == ''
    assert err == ''


def test_unicode_decode_error(monkeypatch, capsys):
    # Simulate decoding error on input
    # Replace stdin with a DummyStream that raises UnicodeDecodeError
    err = UnicodeDecodeError('utf-8', b'abc', 0, 1, 'test')
    monkeypatch.setattr('sys.stdin', DummyStream(err))
    rc = main([])
    out, err = capsys.readouterr()
    assert rc == 2
    assert out == ''
    assert err == ''


def test_empty_and_single_element_lines(monkeypatch, capsys):
    # Input with multiple edge cases: single token, only hyphens, all empty lines
    user_input = '--\nword\n\n-foo-\n'  # '--' is empty strings between hyphens; '-foo-' leads/trails hyphens
    ref_lines = []
    for line in user_input.splitlines():
        ref_lines.append(ref_reduce_line(line))
    expected_output = '\n'.join(ref_lines) + '\n'
    monkeypatch.setattr('sys.stdin', io.StringIO(user_input))
    rc = main([])
    out, err = capsys.readouterr()
    assert rc == 0
    assert out == expected_output


def test_no_contiguous_duplicates(monkeypatch, capsys):
    # Line with no contiguous duplicates; must be unchanged
    user_input = 'abc-def-ghi-jkl\nabc\nfoo-bar-baz\n'
    exp_lines = user_input.splitlines()
    expected_output = '\n'.join(exp_lines) + '\n'
    monkeypatch.setattr('sys.stdin', io.StringIO(user_input))
    rc = main([])
    out, err = capsys.readouterr()
    assert rc == 0
    assert out == expected_output


def test_trailing_newline_handling(monkeypatch, capsys):
    # Test that output line count and trailing newlines match exactly
    user_input = 'a-a-b-b\n\n\n'  # input with two trailing empty lines
    ref_lines = []
    for line in user_input.splitlines():
        ref_lines.append(ref_reduce_line(line))
    expected_output = '\n'.join(ref_lines) + '\n\n'  # match input trailing lines
    monkeypatch.setattr('sys.stdin', io.StringIO(user_input))
    rc = main([])
    out, err = capsys.readouterr()
    assert rc == 0
    assert out == expected_output
