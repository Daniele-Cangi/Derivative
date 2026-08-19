# V4-002 Acceptance Oracle for dedupcase: tests for main(argv)
import os
import sys
import io
import tempfile
import shutil
import pytest

# Import the required public target exactly as named.
from dedupcase import main

def read_binary_file(path):
    # Return bytes (preserving line endings)
    with open(path, 'rb') as f:
        return f.read()

def write_text_file(path, lines, eol='\n', encoding='utf-8'):
    # lines: list[str] (no eol at end of each line)
    # eol: the line ending to append to each line, can vary per-line
    with open(path, 'wb') as f:
        for line, ending in lines:
            f.write(line.encode(encoding) + ending.encode(encoding))

# -- Fixture: mixed case and whitespace duplicate lines, preserve first appearance and original endings.
def make_mixed_case_duplicates(tmp_path):
    # Set up a text file with deliberately mixed-case and whitespace/line-ending variations
    # The list elements: (str(line), str(line_ending))
    lines = [
        ("  Apple", "\r\n"),
        ("apple ", "\n"),
        ("Banana", "\r\n"),
        ("BANANA", "\n"),
        ("Cherry", "\r"),
        ("cherry ", "\r"),
        ("  mango", "\n"),
        ("MANGO  ", "\r\n"),
        ("grape", "\n")
    ]
    inpath = tmp_path / "in.txt"
    outpath = tmp_path / "out.txt"
    write_text_file(inpath, lines)
    # Expected output: first unique (case-space-stripped) with original eol
    expected = b"  Apple\r\nBanana\r\nCherry\r  mango\ngrape\n"
    return str(inpath), str(outpath), expected

# -- Fixture: file with Unicode and mixed endings/whitespace
def make_unicode_file(tmp_path):
    lines = [
        ("\u2603 snowman", "\n"),
        ("  \u2603 Snowman ", "\r\n"),
        ("\u00df stra\u00dfe ", "\r"),
        ("\u00DF STRASSE", "\n"),
        ("café", "\n"),
        ("CAFÉ ", "\r\n")
    ]
    inpath = tmp_path / "in_unicode.txt"
    outpath = tmp_path / "out_unicode.txt"
    write_text_file(inpath, lines)
    # Expect only distinct (case-insensitive, whitespace-stripped, Unicode-wise) lines, keep first for each
    expected = b"\xe2\x98\x83 snowman\n\xdf stra\xc3\x9fe \rcaf\xc3\xa9\n"
    return str(inpath), str(outpath), expected

# -- Fixture: empty file
def make_empty_file(tmp_path):
    inpath = tmp_path / "emptyin.txt"
    outpath = tmp_path / "emptyout.txt"
    open(inpath, 'w', encoding='utf-8').close()
    return str(inpath), str(outpath)

# ---------------- Tests ------------------

def test_dedupcase_mixed_case_and_whitespace(tmp_path):
    inpath, outpath, expected = make_mixed_case_duplicates(tmp_path)
    ret = main([inpath, outpath])
    assert ret == 0
    actual = read_binary_file(outpath)
    assert actual == expected


def test_dedupcase_unicode_and_mixed_eol(tmp_path):
    inpath, outpath, expected = make_unicode_file(tmp_path)
    ret = main([inpath, outpath])
    assert ret == 0
    actual = read_binary_file(outpath)
    assert actual == expected


def test_dedupcase_empty_input(tmp_path):
    inpath, outpath = make_empty_file(tmp_path)
    ret = main([inpath, outpath])
    assert ret == 0
    # Output file must exist and be empty
    assert os.path.exists(outpath)
    assert read_binary_file(outpath) == b""


def test_dedupcase_nonexistent_input(tmp_path, capsys):
    fake_in = tmp_path / "doesnotexist.txt"
    outpath = tmp_path / "willnotmatter.txt"
    ret = main([str(fake_in), str(outpath)])
    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"Input file not found or unreadable: {fake_in}\n"
    # Output file must not exist
    assert not os.path.exists(outpath)


def test_dedupcase_insufficient_args(capsys):
    ret = main([])
    assert ret == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "usage: dedupcase INPUT OUTPUT\n"

    ret = main(["foo.txt"])
    assert ret == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "usage: dedupcase INPUT OUTPUT\n"


def test_dedupcase_too_many_args(capsys):
    ret = main(["a", "b", "c"])
    assert ret == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "usage: dedupcase INPUT OUTPUT\n"


def test_dedupcase_stdin_to_stdout(monkeypatch, capsys):
    # read from stdin, write to stdout
    # lines have to have various spaces -- only first occurrence per (case, strip) allowed
    data = "One \r\n  two \nONE\n  two\r\nThree \r\n three\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(data))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    ret = main(["-", "-"])
    assert ret == 2   # not allowed
    captured = capsys.readouterr()
    assert "usage: dedupcase INPUT OUTPUT\n" == captured.err


def test_dedupcase_stdin_to_file(tmp_path, monkeypatch):
    # stdin, regular output file
    input_data = "KeepMe\n     keepme    \nSecond\n  SECOND  \nThird\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(input_data))
    outpath = tmp_path / "out2.txt"
    ret = main(["-", str(outpath)])
    assert ret == 0
    with open(outpath, "r", encoding="utf-8", newline="") as f:
        contents = f.read()
    # Only first appearance (trim+case-insensitive) of each line retained, keep original line endings
    assert contents == "KeepMe\nSecond\nThird\n"


def test_dedupcase_file_to_stdout(tmp_path, monkeypatch):
    # Test file input, - for stdout
    lines = [("Alpha", "\n"), ("alpha ", "\n"), ("BETA", "\r\n"), ("beta", "\r\n"), ("Gamma", "\n")]
    inpath = tmp_path / "in3.txt"
    write_text_file(inpath, lines)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    ret = main([str(inpath), "-"])
    assert ret == 0
    output = sys.stdout.getvalue()
    assert output == "Alpha\nBETA\r\nGamma\n"


def test_dedupcase_identical_lines_whitespace_differs(tmp_path):
    # identical text lines, differing only by whitespace
    lines = [("dupe", "\n"), (" dupe ", "\n"), ("dupe", "\n"), ("DUPE", "\n"), ("Dupe", "\n"), ("unique", "\r\n")]
    inpath = tmp_path / "in4.txt"
    outpath = tmp_path / "out4.txt"
    write_text_file(inpath, lines)
    ret = main([str(inpath), str(outpath)])
    assert ret == 0
    with open(outpath, "rb") as f:
        actual = f.read()
    # Only first appearance of 'dupe' (with exact original padding and eol), then unique
    assert actual == b"dupe\nunique\r\n"
