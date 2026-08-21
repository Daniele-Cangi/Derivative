import io
import sys
import tempfile
import os
import pytest
from forge_blind_v7.cli_filter_palindromic_fields import main

def pal_field_line_generator(lines):
    # Generator for building input lines with all palindromic fields
    for rec in lines:
        fields = rec.split(',')
        if all(field == field[::-1] for field in fields):
            yield rec

def test_all_palindromic_fields_file(tmp_path, capsys):
    # Prepare palindromic record lines and non-palindromic lines
    pal_lines = [
        "abcba,civic,",
        ",,",
        "a,bb,a",
        "MOM,121,A"
    ]
    nonpal_lines = [
        "abcba,dog",
        "pal,evil,lap"
    ]
    all_lines = pal_lines + nonpal_lines
    input_content = '\n'.join(all_lines) + '\n'
    # Write to temp file
    file_path = tmp_path / "palindromes.csv"
    file_path.write_text(input_content, encoding="utf-8")

    expected_output = '\n'.join(pal_lines) + '\n'
    exit_code = main([str(file_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out == expected_output
    assert captured.err == ''

def test_stdin_empty_input(monkeypatch, capsys):
    # Empty stdin should produce empty stdout
    monkeypatch.setattr(sys, "stdin", io.StringIO(''))
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''

def test_mixed_palindromes_stdin(monkeypatch, capsys):
    # Only some records have all-palindrome fields
    input_lines = [
        "racecar,madam,",
        "nonpal,12321,",
        "bob,otto,civic",
        "abc,def,ghi"
    ]
    input_data = '\n'.join(input_lines) + '\n'
    # Only first and third lines are all-palindrome fields
    expected_lines = [input_lines[0], input_lines[2]]
    monkeypatch.setattr(sys, "stdin", io.StringIO(input_data))
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    expected = '\n'.join(expected_lines) + '\n'
    assert captured.out == expected
    assert captured.err == ''

def test_file_not_found(tmp_path, capsys):
    # Refer to a filename that does not exist
    nonexistent = tmp_path / "missing_file.csv"
    exit_code = main([str(nonexistent)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ''
    assert ("No such file" in captured.err or "not found" in captured.err or "No such file or directory" in captured.err)


def test_unicode_and_empty_fields(tmp_path, capsys):
    # Unicode palindromes and empty fields handling
    pal1 = "あいいあ,,"
    pal2 = "radar,шалаш,"
    nonpal = "あい,eve,radarz"
    content = f"{pal1}\n{nonpal}\n{pal2}\n"
    fpath = tmp_path / "unicode_pals.csv"
    fpath.write_text(content, encoding="utf-8")
    # Only pal1 and pal2 should be output
    expected = f"{pal1}\n{pal2}\n"
    exit_code = main([str(fpath)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == expected
    assert captured.err == ''


def test_completely_non_palindromic_file(tmp_path, capsys):
    # All records contain at least one non-palindrome field, so no output
    lines = [
        "abc,def,ghi",
        "qwerty,asdfg,zxcvb",
        "radar,otto,hello"
    ]
    fpath = tmp_path / "nonpals.csv"
    fpath.write_text('\n'.join(lines) + '\n', encoding="utf-8")
    exit_code = main([str(fpath)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ''
    assert captured.err == ''


def test_single_empty_line_stdin(monkeypatch, capsys):
    # Empty line should be considered as a single palindromic-empty field
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    exit_code = main([])
    captured = capsys.readouterr()
    # Output should contain the empty line
    assert exit_code == 0
    assert captured.out == "\n"
    assert captured.err == ''
