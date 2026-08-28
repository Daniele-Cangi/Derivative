import io
import os
import sys
import tempfile
import pytest
from line_tail_sort import main

def _write_temp_file(content_bytes):
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(content_bytes)
        return path
    except Exception:
        os.unlink(path)
        raise

def test_basic_sorting_and_preserved_endings(monkeypatch):
    # Prepare test file with diverse line endings, whitespace, and duplicates
    lines = [
        'apple \n',         # last n-ws 'e'
        'banana\t\r\n',    # last n-ws 'a'
        '    \t \n',       # all whitespace
        '\n',               # empty
        'kiwi \t\n',       # last n-ws 'i'
        'pear\r\n',        # last n-ws 'r'
        'apple  \n',        # last n-ws 'e' (trailing spaces)
        'banana\n',         # last n-ws 'a' (plain line)
    ]
    # Compose file as bytes
    content = ''.join(lines).encode('utf-8')
    path = _write_temp_file(content)
    try:
        out = io.StringIO()
        err = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        # Directly invoke main (requirement: argv[1] is filename)
        exitcode = main(["line_tail_sort", path])
        # Compute key used for sorting
        def sort_key(line):
            # Remove line ending (preserve for output)
            content = line.rstrip('\r\n')
            s = content.rstrip()
            if not s:
                return chr(0)
            tail = next((ch for ch in reversed(s) if not ch.isspace()), '\x00')
            return tail
        # Python's sort is stable, so lines with same key retain relative order
        expected_lines = sorted(lines, key=sort_key)
        expected_output = ''.join(expected_lines)
        assert exitcode == 0
        assert out.getvalue() == expected_output
        assert err.getvalue() == ''
    finally:
        os.unlink(path)

def test_handles_whitespace_and_empty_lines(monkeypatch):
    lines = [
        '    \t   \n',   # all whitespace
        '\n',            # empty
        'abc\n',         # 'c'
        'zzz     \n',    # 'z'
        '  \t\n',        # all ws
        'another\n',     # 'r'
    ]
    content = ''.join(lines).encode('utf-8')
    path = _write_temp_file(content)
    try:
        out = io.StringIO()
        err = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        # Direct call
        rc = main(["line_tail_sort", path])
        def sort_key(line):
            s = line.rstrip('\r\n').rstrip()
            if not s:
                return chr(0)
            return next((ch for ch in reversed(s) if not ch.isspace()), chr(0))
        expected = sorted(lines, key=sort_key)
        expect_out = ''.join(expected)
        assert rc == 0
        assert out.getvalue() == expect_out
        assert err.getvalue() == ''
    finally:
        os.unlink(path)

def test_utf8_decode_error(monkeypatch):
    # Bytes that are not valid UTF-8 (e.g. lone 0xFF)
    bad_bytes = b'good line\nthis will fail: \xff\n'
    path = _write_temp_file(bad_bytes)
    try:
        out = io.StringIO()
        err = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        rc = main(["line_tail_sort", path])
        assert rc == 1
        assert out.getvalue() == ''
        assert err.getvalue() == 'error: cannot read file'
    finally:
        os.unlink(path)

def test_file_does_not_exist(monkeypatch):
    # Pass a filename that does not exist
    nonexist = os.path.join(tempfile.gettempdir(), 'no_such_file_v8')
    if os.path.exists(nonexist):
        os.unlink(nonexist)
    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', out)
    monkeypatch.setattr(sys, 'stderr', err)
    rc = main(["line_tail_sort", nonexist])
    assert rc == 1
    assert out.getvalue() == ''
    assert err.getvalue() == 'error: cannot read file'

def test_empty_file(monkeypatch):
    # Valid empty file: expect nothing on either channel and exit 0
    path = _write_temp_file(b'')
    try:
        out = io.StringIO()
        err = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', out)
        monkeypatch.setattr(sys, 'stderr', err)
        rc = main(["line_tail_sort", path])
        assert rc == 0
        assert out.getvalue() == ''
        assert err.getvalue() == ''
    finally:
        os.unlink(path)
