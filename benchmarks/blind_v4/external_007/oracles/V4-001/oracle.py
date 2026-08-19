import io
import os
import sys
import tempfile
import pytest

from revwords import main

def write_utf8_file(path, content):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)

def read_utf8_file(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return f.read()

# --- Fixtures ---
@pytest.fixture
def mixed_line_input(tmp_path):
    # Each line ends with: \r\n, \n, \r
    text = 'Unicode тест αβγ!\r\nTwo\twords\nThird\r\n\r\nΕλλάδα\r'
    p = tmp_path / 'input.txt'
    write_utf8_file(p, text)
    return str(p), text

@pytest.fixture
def unicode_input(tmp_path):
    text = 'ſoͤme wørds Δοκιμή 🚀\nこんにちは 世界\n'
    p = tmp_path / 'unicode.txt'
    write_utf8_file(p, text)
    return str(p), text

@pytest.fixture
def empty_input_file(tmp_path):
    p = tmp_path / 'empty.txt'
    write_utf8_file(p, '')
    return str(p)

# --- Test Cases ---
def test_basic_mixed_line_endings(tmp_path, mixed_line_input):
    inp_path, text = mixed_line_input
    out_path = tmp_path / 'out.txt'
    # Expected: every word reversed, line preservation
    expected = (
        'edocinU тсет !γβα\r\n'   # Line endings preserved
        'owT\tsdrow\n'
        'drihT\r\n'
        '\r\n'  # Empty lines unchanged
        'αδάλΕ\r'
    )
    # Call main directly
    result = main(['revwords', inp_path, str(out_path)])
    assert result == 0
    with open(out_path, 'r', encoding='utf-8', newline='') as fout:
        out = fout.read()
    assert out == expected


def test_unicode_and_empty_lines(tmp_path, unicode_input):
    inp_path, text = unicode_input
    out_path = tmp_path / 'out2.txt'
    # Each word reversed, Unicode preserved
    expected = (
        'emͤos sdrøw ήμηκιΔ 🚀\n'
        'はちにんこ 界世\n'
    )
    result = main(['revwords', inp_path, str(out_path)])
    assert result == 0
    out = read_utf8_file(out_path)
    assert out == expected


def test_empty_input_file(tmp_path, empty_input_file):
    out_path = tmp_path / 'empty_out.txt'
    result = main(['revwords', empty_input_file, str(out_path)])
    assert result == 0
    content = read_utf8_file(out_path)
    assert content == ''


def test_empty_lines_preserved(tmp_path):
    inp = tmp_path / 'uel.txt'
    lines = '\n\r\n\n'
    write_utf8_file(inp, lines)
    outp = tmp_path / 'uel_out.txt'
    result = main(['revwords', str(inp), str(outp)])
    assert result == 0
    assert read_utf8_file(outp) == lines


def test_stdin_to_stdout(monkeypatch, capsys):
    # input: '-', output: '-'
    input_content = 'abc def\n\nμνλ 123\n'
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(input_content.encode('utf-8')), encoding='utf-8'))
    monkeypatch.setattr(sys, 'stdout', io.TextIOWrapper(io.BytesIO(), encoding='utf-8'))
    # Should raise usage error if both are '-'
    with pytest.raises(SystemExit) as se:
        main(['revwords', '-', '-'])
    assert se.value.code == 2
    # usage error
    sys.stdout.seek(0)
    # should not print to stdout
    assert sys.stdout.read() == ''


def test_stdin_to_file(monkeypatch, tmp_path):
    # input: '-', output: file
    input_content = 'x yz\n\rΑΒ γδ\n\n'
    input_bytes = input_content.encode('utf-8')
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(input_bytes), encoding='utf-8'))
    out_path = tmp_path / 'sout.txt'
    result = main(['revwords', '-', str(out_path)])
    assert result == 0
    expected = 'x z y\n\rΒΑ δγ\n\n'
    assert read_utf8_file(out_path) == expected


def test_file_to_stdout(tmp_path, monkeypatch, capsys):
    # input: file, output: '-'
    input_content = 'foo\tбар\n\r\nz\n'
    inp = tmp_path / 'stdoutin.txt'
    write_utf8_file(inp, input_content)
    out_buffer = io.BytesIO()
    monkeypatch.setattr(sys, 'stdout', io.TextIOWrapper(out_buffer, encoding='utf-8'))
    result = main(['revwords', str(inp), '-'])
    assert result == 0
    sys.stdout.flush()
    out_buffer.seek(0)
    output = out_buffer.read().decode('utf-8')
    expected = 'oof\tраб\n\r\nz\n'
    assert output == expected


def test_input_file_not_found(tmp_path, capsys):
    # non-existent input file
    inp = tmp_path / 'nofile.txt'
    outp = tmp_path / 'out.txt'
    result = main(['revwords', str(inp), str(outp)])
    assert result == 1
    err = capsys.readouterr().err
    assert err == f'Input file not found or unreadable: {inp}\n'
    # should not write output
    assert not outp.exists()


def test_argument_count_errors(capsys):
    # < 2 args
    r1 = main(['revwords'])
    assert r1 == 2
    err1 = capsys.readouterr().err
    assert err1 == 'usage: revwords INPUT OUTPUT\n'
    # 1 arg
    r2 = main(['revwords', 'onearg'])
    assert r2 == 2
    err2 = capsys.readouterr().err
    assert err2 == 'usage: revwords INPUT OUTPUT\n'
    # >2 args
    r3 = main(['revwords', 'a', 'b', 'c'])
    assert r3 == 2
    err3 = capsys.readouterr().err
    assert err3 == 'usage: revwords INPUT OUTPUT\n'


def test_other_error(monkeypatch, tmp_path, capsys):
    # Simulate error when opening output file
    inp = tmp_path / 'xx.txt'
    outp = tmp_path / 'xxout.txt'
    write_utf8_file(inp, 'foo bar\n')
    # Patch open to raise for output file
    orig_open = open
    def fail_open(*a, **kw):
        if a[0] == str(outp):
            raise OSError('whoops')
        return orig_open(*a, **kw)
    monkeypatch.setattr('builtins.open', fail_open)
    result = main(['revwords', str(inp), str(outp)])
    assert result == 1
    err = capsys.readouterr().err
    # May print nothing or an implementation error message (requirement: exit 1; message not specified)
    # Just check exit code
