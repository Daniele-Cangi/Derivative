import io
import os
import sys
import tempfile
import pytest

from colpad import main

# Helper functions for file-based tests
def write_bytes(path, data):
    with open(path, 'wb') as f:
        f.write(data)

def read_bytes(path):
    with open(path, 'rb') as f:
        return f.read()

class DummyStdIO:
    def __init__(self, data=b"", mode="r"):
        # mode: "r" for reading (simulate stdin), "w" for capturing output (simulate stdout/stderr)
        self._input = io.BytesIO(data) if 'r' in mode else None
        self._output = io.BytesIO() if 'w' in mode else None
        self.mode = mode
        self.buffer = self._input or self._output
        self.encoding = 'utf-8'
    def read(self, *a, **kw):
        return self._input.read(*a, **kw).decode('utf-8') if self._input else ''
    def write(self, text):
        return self._output.write(text.encode('utf-8')) if self._output else 0
    def flush(self):
        pass
    def getvalue(self):
        return self._output.getvalue().decode('utf-8') if self._output else ''
    def readline(self, *a, **kw):
        return self._input.readline(*a, **kw).decode('utf-8') if self._input else ''
    def __getattr__(self, name):
        # fallback to BytesIO attributes
        return getattr(self.buffer, name)

@pytest.mark.parametrize(
    "in_lines, line_ending, n, expected_lines",
    [
        # a) fewer, exact, and more than N columns
        (["a\tb", "c\td\te", "f\tg\th\ti"], '\n', 3, ["a\tb\t", "c\td\te", "f\tg\th"]),
        (["x", "y\tz", "1\t2\t3\t4"], '\r\n', 2, ["x\t", "y\tz", "1\t2"]),
        (["aa\tbb\tcc", "xx"], '\r', 4, ["aa\tbb\tcc\t", "xx\t\t\t"]),
    ]
)
def test_colpad_basic_file(tmp_path, in_lines, line_ending, n, expected_lines):
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    # Compose input file
    input_bytes = line_ending.join(in_lines).encode('utf-8')
    write_bytes(input_path, input_bytes)
    # Main invocation
    ret = main([str(input_path), str(output_path), str(n)])
    assert ret == 0
    out_bytes = read_bytes(output_path)
    # Expected output
    expected = line_ending.join(expected_lines).encode('utf-8')
    assert out_bytes == expected

@pytest.mark.parametrize("in_bytes, line_ending", [
    (b"", '\n'), # c) empty input file with LF
    (b"", '\r\n'), # c) empty input file with CRLF
    (b"", '\r'), # c) empty input file with CR
])
def test_colpad_empty_file(tmp_path, in_bytes, line_ending):
    input_path = tmp_path / "empty.txt"
    output_path = tmp_path / "out.txt"
    write_bytes(input_path, in_bytes)
    ret = main([str(input_path), str(output_path), '3'])
    assert ret == 0
    out_bytes = read_bytes(output_path)
    assert out_bytes == b""

@pytest.mark.parametrize("n_arg, errmsg", [
    ("0", "Error: N must be a positive integer"),
    ("-4", "Error: N must be a positive integer"),
    ("abc", "Error: N must be a positive integer"),
])
def test_colpad_invalid_n(tmp_path, n_arg, errmsg):
    input_path = tmp_path / "in.txt"
    output_path = tmp_path / "out.txt"
    write_bytes(input_path, b"a\tb\tc\n")
    stderr = DummyStdIO(mode="w")
    orig_stderr = sys.stderr
    try:
        sys.stderr = stderr
        ret = main([str(input_path), str(output_path), n_arg])
    finally:
        sys.stderr = orig_stderr
    assert ret == 1
    assert errmsg in stderr.getvalue()

@pytest.mark.parametrize("args,err,code", [
    ([], 'usage: colpad INPUT OUTPUT N', 2),
    (["foo.txt"], 'usage: colpad INPUT OUTPUT N', 2),
    (["a.txt","b.txt","9","extra"], 'usage: colpad INPUT OUTPUT N', 2),
    (["-","-","4"], 'usage: colpad INPUT OUTPUT N', 2),
])
def test_colpad_usage_errors(args, err, code):
    stderr = DummyStdIO(mode="w")
    orig_stderr = sys.stderr
    try:
        sys.stderr = stderr
        ret = main(list(args))
    finally:
        sys.stderr = orig_stderr
    assert ret == code
    assert err in stderr.getvalue()

def test_colpad_input_not_found(tmp_path):
    input_path = tmp_path / "noexist.txt"
    output_path = tmp_path / "out.txt"
    # Intentionally missing input file
    stderr = DummyStdIO(mode="w")
    orig_stderr = sys.stderr
    try:
        sys.stderr = stderr
        ret = main([str(input_path), str(output_path), "3"])
    finally:
        sys.stderr = orig_stderr
    assert ret == 1
    assert f"Input file not found or unreadable: {input_path}" in stderr.getvalue()


def test_colpad_output_not_writable(tmp_path):
    input_path = tmp_path / "input.txt"
    write_bytes(input_path, b"1\t2\n")
    # Use a directory as output path (writing to directory is invalid)
    unwritable_path = tmp_path
    stderr = DummyStdIO(mode="w")
    orig_stderr = sys.stderr
    try:
        sys.stderr = stderr
        ret = main([str(input_path), str(unwritable_path), '2'])
    finally:
        sys.stderr = orig_stderr
    assert ret == 1
    assert f"Output file not writable: {unwritable_path}" in stderr.getvalue()


def test_colpad_stdin_stdout(monkeypatch):
    # e) input from stdin, output to stdout
    lines = ["foo", "bar\tbaz", "one\ttwo\tthree"]
    input_bytes = "\n".join(lines).encode("utf-8")
    stdin = DummyStdIO(input_bytes, mode="r")
    stdout = DummyStdIO(mode="w")
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    ret = main(['-', '-', '2'])
    assert ret == 2
    # This should error due to both input and output as '-'
    # Handled in usage errors test; so here, confirm this is 2 and no extra output on stdout
    assert stdout.getvalue() == ''

def test_colpad_stdin_to_file(monkeypatch, tmp_path):
    # Read from stdin, write to file
    lines = ["x", "y\tz", "a\tb\tc\td"]
    input_bytes = "\r\n".join(lines).encode("utf-8")
    stdin = DummyStdIO(input_bytes, mode="r")
    monkeypatch.setattr(sys, "stdin", stdin)
    output_path = tmp_path / "out.txt"
    ret = main(['-', str(output_path), '3'])
    assert ret == 0
    result = read_bytes(output_path)
    assert result == b"x\t\r\ny\tz\r\na\tb\tc\r\n"


def test_colpad_file_to_stdout(monkeypatch, tmp_path):
    # Read from file, write to stdout
    input_path = tmp_path / "input.txt"
    write_bytes(input_path, b"foo\nbar\tbaz\nelem1\telem2\telem3\telem4\n")
    stdout = DummyStdIO(mode="w")
    monkeypatch.setattr(sys, "stdout", stdout)
    ret = main([str(input_path), '-', '2'])
    assert ret == 0
    out = stdout.getvalue()
    # Should see only 2 columns in each line
    assert out == "foo\t\nbar\tbaz\nelem1\telem2\n"


def test_colpad_other_exception(monkeypatch, tmp_path):
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "out.txt"
    write_bytes(input_path, b"will\tfail\n")
    def bad_open(*a, **k): raise RuntimeError("BOOM")
    # Monkeypatch open to raise exception on output
    monkeypatch.setattr("builtins.open", bad_open)
    stderr = DummyStdIO(mode="w")
    orig_stderr = sys.stderr
    try:
        sys.stderr = stderr
        ret = main([str(input_path), str(output_path), '2'])
    finally:
        sys.stderr = orig_stderr
    assert ret == 1
    assert "Error: BOOM" in stderr.getvalue()
