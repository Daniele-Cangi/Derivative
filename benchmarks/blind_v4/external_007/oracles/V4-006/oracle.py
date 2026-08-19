import io
import os
import sys
import tempfile
import pytest
from rstrim import main

USAGE_MSG = 'usage: rstrim INPUT OUTPUT'  # required verbatim

@pytest.fixture
def temp_text_file():
    fd, path = tempfile.mkstemp(suffix='.txt')
    os.close(fd)
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)

# a) lines with only trailing whitespace
# b) lines with only whitespace before line endings
# c) preservation of leading whitespace
# d) mixed/unusual line endings
# e) input or output as '-'
# f) improper argument counts
# g) missing input
# h) unwritable output
# i) empty input file


def _read_file(path, mode='r'):
    with open(path, mode, encoding='utf-8') as f:
        return f.read()


def test_trailing_and_only_whitespace_and_leading_preserved(temp_text_file, tmp_path, capsys):
    contents = (
        'abc   \n'      # trailing spaces
        '\tabc\t \r\n' # tab, trailing whitespace, CRLF ending
        ' \t \r'        # line with just whitespace and CR
        '  abc\x0c\x0c\n' # leading spaces, internal formfeed, trailing FF, LF
        ' \t\r'         # just whitespace, CR
        '\n'            # just line ending
    )
    with open(temp_text_file, 'w', encoding='utf-8', newline='') as f:
        f.write(contents)
    outfile = tmp_path / 'out.txt'
    ret = main([temp_text_file, str(outfile)])
    assert ret == 0
    out = _read_file(outfile, mode='r')
    # 1: 'abc\n'
    # 2: '\tabc\r\n'
    # 3: '\r' (all whitespace removed before CR)
    # 4: '  abc\x0c\x0c\n'
    # 5: '\r' (all whitespace removed before CR)
    # 6: '\n' (line with only LF preserved)
    assert out == (
        'abc\n'
        '\tabc\r\n'
        '\r'
        '  abc\x0c\x0c\n'
        '\r'
        '\n'
    )
    captured = capsys.readouterr()
    assert captured.err == ''


def test_mixed_line_endings_and_preserves(tmp_path, capsys):
    infile = tmp_path / 'in.txt'
    # Mix: line1 (LF), line2 (CRLF), line3 (CR)
    data = ' a \n\t\t  \r\n\t\rabc\r'
    with open(infile, 'w', encoding='utf-8', newline='') as f:
        f.write(data)
    outfile = tmp_path / 'out.txt'
    ret = main([str(infile), str(outfile)])
    assert ret == 0
    out = _read_file(outfile)
    # ' a\n'  (one space trimmed, LF)
    # '\t\t\r\n'  (no trailing whitespace, CRLF)
    # '\tabc\r' (no trailing whitespace, CR)
    assert out == ' a\n\t\t\r\n\tabc\r'
    captured = capsys.readouterr()
    assert captured.err == ''


def test_input_dash_reads_stdin(tmp_path, capsys):
    # Input from stdin, output file
    lines = ['abc  \n', '\t  \r\n', ' \t\n']
    text = ''.join(lines)
    outfile = tmp_path / 'out.txt'
    # Patch sys.stdin
    orig_stdin = sys.stdin
    sys.stdin = io.StringIO(text)
    try:
        ret = main(['-', str(outfile)])
    finally:
        sys.stdin = orig_stdin
    assert ret == 0
    out = _read_file(outfile)
    assert out == 'abc\n\t\r\n\n'
    captured = capsys.readouterr()
    assert captured.err == ''


def test_output_dash_writes_stdout(temp_text_file, capsys):
    # Input file, output to stdout
    with open(temp_text_file, 'w', encoding='utf-8', newline='') as f:
        f.write('abc  \n   \nfoo\t\n')
    ret = main([temp_text_file, '-'])
    assert ret == 0
    captured = capsys.readouterr()
    # First line: 'abc\n', second: only whitespace->'\n', third: 'foo\n'
    assert captured.out == 'abc\n\nfoo\n'
    assert captured.err == ''


def test_input_and_output_dash_usage(capsys):
    ret = main(['-', '-'])
    captured = capsys.readouterr()
    assert USAGE_MSG in captured.err
    assert ret == 2


def test_too_few_arguments(capsys):
    ret = main(['foo.txt'])
    captured = capsys.readouterr()
    assert USAGE_MSG in captured.err
    assert ret == 2


def test_too_many_arguments(capsys):
    ret = main(['a.txt', 'b.txt', 'c.txt'])
    captured = capsys.readouterr()
    assert USAGE_MSG in captured.err
    assert ret == 2


def test_nonexistent_input_file(tmp_path, capsys):
    missing = tmp_path / 'doesnotexist.txt'
    outfile = tmp_path / 'out.txt'
    ret = main([str(missing), str(outfile)])
    captured = capsys.readouterr()
    assert ret == 1
    assert f'Input file not found or unreadable: {missing}' in captured.err
    assert not outfile.exists()


def test_unwritable_output_file(temp_text_file, tmp_path, capsys):
    # Make directory where file is unwritable
    unwritable_dir = tmp_path / 'unwritable'
    unwritable_dir.mkdir()
    os.chmod(unwritable_dir, 0o400)  # Read-only
    outfile = unwritable_dir / 'output.txt'
    try:
        ret = main([temp_text_file, str(outfile)])
        captured = capsys.readouterr()
        assert ret == 1
        assert f'Output file not writable: {outfile}' in captured.err
    finally:
        # Reset permissions to allow cleanup
        os.chmod(unwritable_dir, 0o700)


def test_empty_input_file(tmp_path, capsys):
    infile = tmp_path / 'empty.txt'
    outfile = tmp_path / 'out.txt'
    open(infile, 'w', encoding='utf-8').close()
    ret = main([str(infile), str(outfile)])
    assert ret == 0
    out = _read_file(outfile)
    assert out == ''
    captured = capsys.readouterr()
    assert captured.err == ''


def test_general_error_prints_message(tmp_path, monkeypatch, capsys):
    # Simulate unexpected error by patching open to raise
    infile = tmp_path / 'f.txt'
    outfile = tmp_path / 'g.txt'
    with open(infile, 'w', encoding='utf-8') as f:
        f.write('test')
    def bad_open(*a, **kw):
        raise ValueError('BOOM')
    monkeypatch.setattr('builtins.open', bad_open)
    ret = main([str(infile), str(outfile)])
    captured = capsys.readouterr()
    assert ret == 1
    assert 'Error: BOOM' in captured.err
