import io
import sys
import pytest
from forge_blind_v9_requirements_slot3 import main

# UTILITIES FOR TESTING
ASCII_PRINTABLE = bytes(range(0x21, 0x7F)).decode('ascii')
NON_PRINTABLE_BYTE = b'\x1b'  # 0x1B is not 0x21-0x7E


def make_ascii_line(fields):
    """Join fields with tabs, add a newline, ensure all fields are ascii-printable."""
    for f in fields:
        for c in f:
            assert 0x21 <= ord(c) <= 0x7E
    return '\t'.join(fields) + '\n'


def encode_invalid_utf8():
    """Return bytes that are not valid UTF-8."""
    return b'\xfffoo\n'


@pytest.mark.parametrize("stdin_content,expected_lines", [
    # Test: all lines have unique fields and ascii printable chars.
    (make_ascii_line(['foo', 'bar']) + make_ascii_line(['baz', 'qux']),
     [make_ascii_line(['foo', 'bar']), make_ascii_line(['baz', 'qux'])]),
    # Test: some lines have duplicate fields
    (make_ascii_line(['foo', 'foo']) + make_ascii_line(['bar', 'baz']) + make_ascii_line(['a', 'a', 'a']),
     [make_ascii_line(['bar', 'baz'])]),
    # Test: mix of legit and empty/tab-only/dup lines with edge tabs
    ('\t\t\n' + make_ascii_line(['AA', 'BB', 'CC']) + make_ascii_line(['xy', 'xy']) + '\n' + make_ascii_line(['A', 'B']) + '\t\n',
     [make_ascii_line(['AA', 'BB', 'CC']), make_ascii_line(['A', 'B'])])
])
def test_accepts_and_filters_valid_lines(monkeypatch, stdin_content, expected_lines, capsys):
    # Patch sys.stdin as TextIO, feed valid ASCII-printable input
    monkeypatch.setattr(sys, 'stdin', io.StringIO(stdin_content))
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    expect = ''.join(expected_lines)
    assert captured.out == expect
    assert captured.err == ''


def test_rejects_invalid_ascii_fields(monkeypatch, capsys):
    # Line contains a non-printable ASCII char (0x1B)
    invalid_line = 'foo' + chr(0x1B) + '\tbar\n'
    monkeypatch.setattr(sys, 'stdin', io.StringIO(invalid_line))
    exit_code = main([])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


def test_rejects_non_utf8_input(monkeypatch, capsys):
    # Simulate sys.stdin.buffer delivering invalid UTF-8 bytes
    class BadBuffer:
        def read(self, *a, **k):
            return encode_invalid_utf8()
    class BadStdin:
        buffer = BadBuffer()
    monkeypatch.setattr(sys, 'stdin', BadStdin())
    exit_code = main([])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


def test_ignores_empty_and_tab_only_lines(monkeypatch, capsys):
    input_lines = '\n\t\n\t\t\nfoo\tbar\n\n\t\t\t\nbar\tbaz\tqux\n'
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input_lines))
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    expect_lines = ['foo\tbar\n', 'bar\tbaz\tqux\n']
    assert captured.out == ''.join(expect_lines)
    assert captured.err == ''


def test_case_sensitivity_of_uniqueness(monkeypatch, capsys):
    # Fields 'abc' and 'ABC' are not duplicates (case-sensitive)
    lines = make_ascii_line(['abc', 'ABC', 'Abc']) + make_ascii_line(['X', 'X'])
    monkeypatch.setattr(sys, 'stdin', io.StringIO(lines))
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    # Only first line should be printed
    assert captured.out == make_ascii_line(['abc', 'ABC', 'Abc'])
    assert captured.err == ''


def test_no_output_on_all_invalid_lines(monkeypatch, capsys):
    # All lines are invalid due to duplicates or being empty
    data = make_ascii_line(['foo', 'foo']) + '\t\t\n\n'
    monkeypatch.setattr(sys, 'stdin', io.StringIO(data))
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


def test_mixed_valid_and_invalid(monkeypatch, capsys):
    # One good line sandwiched by invalids (dup and bad ascii)
    valid = make_ascii_line(['X', 'YZ'])
    invalid1 = make_ascii_line(['foo', 'foo'])
    nonascii = 'A' + chr(0x80) + '\tB\n'  # 0x80 is not allowed
    mixed = invalid1 + valid + nonascii
    monkeypatch.setattr(sys, 'stdin', io.StringIO(mixed))
    exit_code = main([])
    # Program must reject with code 2 and no output for invalid char
    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


def test_trailing_newlines_preserved(monkeypatch, capsys):
    # Only valid line with explicit extra newline at end
    line = make_ascii_line(['a', 'b'])
    monkeypatch.setattr(sys, 'stdin', io.StringIO(line + '\n'))
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    # Output preserves original trailing newlines
    assert captured.out == line
    assert captured.err == ''
