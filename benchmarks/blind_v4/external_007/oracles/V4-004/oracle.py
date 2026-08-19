import os
import io
import sys
import stat
import tempfile
import shutil
import pytest

from squeeze import main

USAGE_MSG = 'usage: squeeze INPUT OUTPUT\n'

# Helper to read file as bytes for exact line ending checks
def _read_bytes(path):
    with open(path, 'rb') as f:
        return f.read()

def _set_unreadable(path):
    os.chmod(path, 0)

def _set_unwritable(path):
    os.chmod(path, stat.S_IREAD)

def _restore_permissions(path):
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)

@pytest.fixture
def temp_text_file():
    files = []
    def _create(contents: bytes):
        fd, p = tempfile.mkstemp(suffix='.txt')
        with os.fdopen(fd, 'wb') as f:
            f.write(contents)
        files.append(p)
        return p
    yield _create
    for p in files:
        try:
            _restore_permissions(p)
            os.remove(p)
        except Exception:
            pass

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)

# (a) Whitespace collapse: spaces/tabs/etc not at boundaries
# (b) Mixed line endings, (c) leading/trailing whitespace
def test_squeeze_basic_functionality(tmp_path, temp_text_file):
    # Compose input: 4 lines
    content = b"  hey\t\tthere\f   friend\r\nfoo\tbar\nqux\r\n   a   b\t\t\f c   \n\n"
    # 1. Preserve leading/trailing
    # 2. collapse runs in the middle
    input_file = temp_text_file(content)
    output_file = tmp_path / "out.txt"
    ret = main([input_file, str(output_file)])
    assert ret == 0
    expected = (b"  hey there friend\r\n"  # spaces preserved, \t\t and \f collapsed to one space
                b"foo bar\n"               # \t in the middle
                b"qux\r\n"               # nothing to collapse
                b"   a b c   \n"          # leading/trailing preserved, runs collapsed
                b"\n")                   # empty line
    actual = _read_bytes(str(output_file))
    assert actual == expected

def test_preserves_line_endings_and_empty_lines(tmp_path, temp_text_file):
    # (b), (c), (i) Empty file and input mixing line endings
    content = b"foo  \t bar\r\n\tlead\f\ttrail  \n\n"
    inf = temp_text_file(content)
    outf = tmp_path / "oe.txt"
    ret = main([inf, str(outf)])
    assert ret == 0
    expected = (b"foo bar\r\n"      #\r\n
                b"\tlead trail  \n" # preserves leading \t, trailing space space
                b"\n")             # empty line
    assert _read_bytes(str(outf)) == expected
    # test empty file
    inf2 = temp_text_file(b"")
    outf2 = tmp_path / "b.txt"
    assert main([inf2, str(outf2)]) == 0
    assert _read_bytes(str(outf2)) == b""

def test_stdin_stdout(monkeypatch, capsys):
    # (d) stdin/stdout, newline at end
    data = "\t foo\t  bar\f baz  \n"
    input_bytes = data.encode('utf-8')
    stdin = io.StringIO(data)
    stdout = io.StringIO()
    monkeypatch.setattr(sys, 'stdin', stdin)
    monkeypatch.setattr(sys, 'stdout', stdout)
    ret = main(['-', '-'])
    assert ret == 2
    err = capsys.readouterr()
    assert USAGE_MSG in err.err
    # now: valid use, stdin -> file
    stdin = io.StringIO(data)
    monkeypatch.setattr(sys, 'stdin', stdin)
    with tempfile.NamedTemporaryFile('w+b', delete=False) as outf:
        outp = outf.name
    try:
        ret = main(['-', outp])
        assert ret == 0
        with open(outp, 'rb') as f:
            assert f.read() == b"\t foo bar baz  \n"
    finally:
        os.remove(outp)
    # file -> stdout
    with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False) as inf:
        inf.write(data)
        inf.flush()
        inp = inf.name
    try:
        stdout = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', stdout)
        ret = main([inp, '-'])
        assert ret == 0
        assert stdout.getvalue() == "\t foo bar baz  \n"
    finally:
        os.remove(inp)

def test_input_file_missing_or_unreadable(tmp_path):
    infile = tmp_path / "notfound.txt"
    outfile = tmp_path / "g.txt"
    # (e) input path missing
    import sys
    from io import StringIO
    orig_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        ret = main([str(infile), str(outfile)])
        assert ret == 1
        err = sys.stderr.getvalue()
    finally:
        sys.stderr = orig_stderr
    assert f"Input file not found or unreadable: {infile}" in err
    # Now, unreadable file
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write("test\n"); inp = f.name
    try:
        _set_unreadable(inp)
        sys.stderr = StringIO()
        ret = main([inp, str(outfile)])
        err = sys.stderr.getvalue()
        assert ret == 1
        assert f"Input file not found or unreadable: {inp}" in err
    finally:
        _restore_permissions(inp)
        os.remove(inp)

def test_output_file_unwritable_and_directory_missing(tmp_path, temp_text_file):
    # (f) Unwritable output path
    content = b"abc\n"
    infile = temp_text_file(content)
    # unwritable file
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write("abc\n"); outf = f.name
    try:
        _set_unwritable(outf)
        from io import StringIO
        orig_stderr = sys.stderr
        sys.stderr = StringIO()
        ret = main([infile, outf])
        err = sys.stderr.getvalue()
        assert ret == 1
        assert f"Output file not writable: {outf}" in err
        sys.stderr = orig_stderr
    finally:
        _restore_permissions(outf)
        os.remove(outf)
    # output directory not existing
    nonex = tmp_path / "nope" / "some.txt"
    from io import StringIO
    orig_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        ret = main([infile, str(nonex)])
        err = sys.stderr.getvalue()
        assert ret == 1
        assert f"Output file not writable: {nonex}" in err
    finally:
        sys.stderr = orig_stderr

def test_both_input_and_output_dash(monkeypatch, capsys):
    # (g): both input and output are '-'
    monkeypatch.setattr(sys, 'stdin', io.StringIO('abc\n'))
    monkeypatch.setattr(sys, 'stdout', io.StringIO())
    ret = main(['-', '-'])
    assert ret == 2
    err = capsys.readouterr().err
    assert USAGE_MSG in err

def test_argument_count_cases(capsys):
    # (h) improper arg counts: zero, one, three, four
    for args in ([], ['foo'], ['a','b','c'], ['a','b','c','d']):
        ret = main(args)
        assert ret == 2
        err = capsys.readouterr().err
        assert USAGE_MSG in err

# Invariant: Unhandled error prints deterministic message, exit 1
def test_internal_error(monkeypatch, temp_text_file):
    # Simulate a crash during writing
    content = b"foo\n"
    infile = temp_text_file(content)
    class ExplodingWriter:
        def write(self, data):
            raise OSError("disk full")
        def close(self):
            return None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            return False
    monkeypatch.setattr('builtins.open', lambda *a, **kw: ExplodingWriter() if a[0].endswith('fail.txt') else open(*a, **kw))
    from io import StringIO
    orig_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        ret = main([infile, 'fail.txt'])
        msg = sys.stderr.getvalue()
    finally:
        sys.stderr = orig_stderr
    assert ret == 1
    assert msg.startswith('Error: disk full')
