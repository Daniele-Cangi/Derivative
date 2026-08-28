import io
import os
import sys
import tempfile
import pytest
from doubled_lines import main

@pytest.fixture
def sample_file(request):
    """
    Fixture to create a temp file with given content. Returns the filename and ensures cleanup.
    """
    files = []
    def _makefile(content, encoding='utf-8', mode='w'):
        fd, fname = tempfile.mkstemp()
        files.append(fname)
        with open(fname, mode, encoding=encoding) as f:
            f.write(content)
        return fname
    yield _makefile
    for f in files:
        try:
            os.remove(f)
        except Exception:
            pass

@pytest.mark.parametrize("lines,expected", [
    (['a\n', 'b\n', 'a\n', 'c\n', 'b\n'], ['a\n', 'b\n']),
    ([chr(10)]*4, [chr(10)]),  # Four empty lines -> only one emitted, as repeated blank
    (['apple\r\n', 'banana\n', 'apple\r', 'apple\r\n', 'banana\n', 'apple\r'], ['apple\r\n', 'banana\n', 'apple\r']),
])
def test_find_doubled_lines_stdout_stderr_exit(monkeypatch, tmp_path, lines, expected, sample_file):
    # Create file with the given lines
    content = ''.join(lines)
    fname = sample_file(content)

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)
    exit_code = main([None, fname])
    result_output = stdout.getvalue()
    result_err = stderr.getvalue()
    # Output must match the first instance of each duplicate line (original order, with line endings)
    assert exit_code == 0
    assert result_output == ''.join(expected)
    assert result_err == ''

def test_unique_lines_no_output(monkeypatch, sample_file):
    # File with only unique lines
    content = 'x\nA\r\nB\rC\n'  # every line unique even if similar appearing
    fname = sample_file(content)
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)
    exit_code = main([None, fname])
    assert exit_code == 0
    # No output at all
    assert stdout.getvalue() == ''
    assert stderr.getvalue() == ''

def test_empty_file_is_no_output(monkeypatch, sample_file):
    # Completely empty file
    fname = sample_file('')
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)
    exit_code = main([None, fname])
    # stdout/stderr empty, proper exit
    assert exit_code == 0
    assert stdout.getvalue() == ''
    assert stderr.getvalue() == ''

def test_unicode_decode_error(monkeypatch, tmp_path):
    # Write invalid UTF-8 bytes
    fname = os.path.join(tmp_path, "badutf8.txt")
    with open(fname, 'wb') as f:
        f.write(b'abc\n\xff\xfe\xfa')  # invalid bytes
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)
    exit_code = main([None, fname])
    assert exit_code == 1
    assert stdout.getvalue() == ''
    assert stderr.getvalue() == 'error: cannot read file'

def test_file_io_error(monkeypatch, tmp_path):
    # Nonexistent file
    fname = os.path.join(tmp_path, 'nofile.txt')
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)
    exit_code = main([None, fname])
    assert exit_code == 1
    assert stdout.getvalue() == ''
    assert stderr.getvalue() == 'error: cannot read file'

def test_lines_that_differ_by_ending_are_not_duplicates(monkeypatch, sample_file):
    # Lines with only line-ending difference must NOT be considered duplicates
    # Example: 'foo\n' and 'foo\r\n' are different
    lines = ['foo\n', 'foo\r\n', 'foo\n', 'foo\r\n']
    # Only 'foo\n' and 'foo\r\n' will each be output their first time if and only if they are duplicated
    expected = ['foo\n', 'foo\r\n']
    fname = sample_file(''.join(lines))
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)
    exit_code = main([None, fname])
    assert exit_code == 0
    # Should only get the first 'foo\n' and the first 'foo\r\n'
    assert stdout.getvalue() == ''.join(expected)
    assert stderr.getvalue() == ''

def test_only_repeated_empty_lines(monkeypatch, sample_file):
    # Only repeated empty lines (could be "\n", "\r", or "\r\n")
    # Here, use three blank lines with different endings and repeat each once:
    lines = ['\n', '\r', '\r\n', '\n', '\r', '\r\n']
    expected = ['\n', '\r', '\r\n']
    fname = sample_file(''.join(lines))
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)
    exit_code = main([None, fname])
    assert exit_code == 0
    assert stdout.getvalue() == ''.join(expected)
    assert stderr.getvalue() == ''
