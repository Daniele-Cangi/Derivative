import io
import sys
import pytest

import dupfilter

class DummyStdin:
    def __init__(self, data: bytes):
        self.buffer = io.BytesIO(data)

    def read(self, size=-1):
        return self.buffer.read(size)

    def readline(self):
        return self.buffer.readline()

    def __iter__(self):
        return self.buffer.__iter__()


@pytest.fixture
def capsys_dupfilter():
    # Helper to capture DUPFILTER output and exit code
    class Result:
        def __init__(self):
            self.exit_code = None
            self.out = None
            self.err = None

    res = Result()
    orig_stdin = sys.stdin

    def run_with_input(input_bytes):
        # Patch sys.stdin with UTF-8 decoded content or raise
        try:
            sys.stdin = io.TextIOWrapper(io.BytesIO(input_bytes), encoding='utf-8')
            dupfilter.main()
            res.exit_code = 0
        except SystemExit as e:
            res.exit_code = e.code
        except UnicodeDecodeError:
            res.exit_code = 1
        finally:
            sys.stdin = orig_stdin

    res.run_with_input = run_with_input
    return res


def test_dupfilter_unique_lines_preserve_order(capsys_dupfilter):
    data = """apple\nbanana\napple\ncherry\nbanana\ndate\n""".encode('utf-8')
    capsys_dupfilter.run_with_input(data)
    assert capsys_dupfilter.exit_code == 0
    # Capture stdout using sys.stdout by rerunning manually
    out = []
    seen = set()
    for line in data.decode('utf-8').splitlines():
        if line not in seen:
            seen.add(line)
            out.append(line)

    captured = []
    # To test output, simulate run and capture print calls by redirecting sys.stdout
    sys.stdin = io.TextIOWrapper(io.BytesIO(data), encoding='utf-8')
    sys.stdout = io.StringIO()
    try:
        dupfilter.main()
        output = sys.stdout.getvalue().splitlines()
    finally:
        sys.stdout = sys.__stdout__
        sys.stdin = sys.__stdin__
    assert output == out


def test_dupfilter_empty_input(capsys_dupfilter):
    capsys_dupfilter.run_with_input(b'')
    assert capsys_dupfilter.exit_code == 0
    sys.stdin = io.TextIOWrapper(io.BytesIO(b''), encoding='utf-8')
    sys.stdout = io.StringIO()
    try:
        dupfilter.main()
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = sys.__stdout__
        sys.stdin = sys.__stdin__
    assert output == ''


def test_dupfilter_non_utf8_input(capsys_dupfilter):
    # Input contains bytes invalid in UTF-8
    invalid_utf8 = b'apple\nbanana\xff\n'
    capsys_dupfilter.run_with_input(invalid_utf8)
    assert capsys_dupfilter.exit_code != 0
    # The utility must not succeed


def test_dupfilter_large_unique_input(monkeypatch):
    # Test memory usage not required, but ensure correctness on large input
    lines = [f'line{i}\n' for i in range(1000)]
    data = ''.join(lines).encode('utf-8')
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(data), encoding='utf-8'))
    sys.stdout = io.StringIO()
    try:
        dupfilter.main()
        output_lines = sys.stdout.getvalue().splitlines()
    finally:
        sys.stdout = sys.__stdout__
        sys.stdin = sys.__stdin__
    expected_lines = [line.strip() for line in lines]
    assert output_lines == expected_lines


def test_dupfilter_duplicate_consecutive_lines(monkeypatch):
    data = b'a\na\na\nb\nb\nc\nc\nc\n'
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(data), encoding='utf-8'))
    sys.stdout = io.StringIO()
    try:
        dupfilter.main()
        output_lines = sys.stdout.getvalue().splitlines()
    finally:
        sys.stdout = sys.__stdout__
        sys.stdin = sys.__stdin__
    assert output_lines == ['a', 'b', 'c']
