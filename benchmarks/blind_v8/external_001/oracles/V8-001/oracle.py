import io
import os
import sys
import tempfile
import pytest
from word_freq_stats import main

# Helper to write bytes to a temp file and return its path
def _write_temp_bytes(data: bytes):
    fd, path = tempfile.mkstemp(suffix='.txt')
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    return path

# Helper to write text to a temp file as utf-8
def _write_temp_text(text: str):
    return _write_temp_bytes(text.encode('utf-8'))

def test_simple_word_count_and_frequency(monkeypatch, capsys):
    text = "Hello world hello"
    path = _write_temp_text(text)
    # Expect: 3 words (Hello, world, hello)\n
    # Casefolded words: ['hello', 'world', 'hello']
    # Frequencies: {'hello': 2, 'world': 1}
    # Highest: 'hello' 2

    exit_code = main([path])
    captured = capsys.readouterr()
    os.remove(path)
    assert exit_code == 0
    out_lines = captured.out.strip().split('\n')
    assert out_lines == ['3', 'hello 2']
    assert captured.err == ''

def test_ties_and_unicode_casefold(monkeypatch, capsys):
    # 'Straße', 'strasse', 'Straße' with casefold all become 'strasse'
    text = 'Straße strasse Straße café CaFÉ'
    path = _write_temp_text(text)
    # Words: ['Straße', 'strasse', 'Straße', 'café', 'CaFÉ']
    # Casefolded: ['strasse', 'strasse', 'strasse', 'café', 'café']
    # Freqs: {'strasse': 3, 'café': 2}
    # Highest: 'strasse 3'

    exit_code = main([path])
    captured = capsys.readouterr()
    os.remove(path)
    assert exit_code == 0
    out_lines = captured.out.strip().split('\n')
    assert out_lines == ['5', 'strasse 3']
    assert captured.err == ''

def test_no_words_outputs_zero(monkeypatch, capsys):
    # Only punctuation and digits; no alphabetic words
    text = "123 456 789! @#$%"
    path = _write_temp_text(text)
    exit_code = main([path])
    captured = capsys.readouterr()
    os.remove(path)
    assert exit_code == 0
    lines = captured.out.strip().split('\n')
    assert lines == ['0', '0']
    assert captured.err == ''

def test_tie_lexicographically_smallest(monkeypatch, capsys):
    text = "Beta beta Alpha alpha"
    path = _write_temp_text(text)
    # Words: ['Beta', 'beta', 'Alpha', 'alpha']
    # Casefolded: ['beta', 'beta', 'alpha', 'alpha']
    # Both 'alpha' and 'beta' occur 2 times; 'alpha' < 'beta'
    exit_code = main([path])
    captured = capsys.readouterr()
    os.remove(path)
    assert exit_code == 0
    lines = captured.out.strip().split('\n')
    assert lines == ['4', 'alpha 2']
    assert captured.err == ''

def test_empty_file(monkeypatch, capsys):
    path = _write_temp_text("")
    exit_code = main([path])
    captured = capsys.readouterr()
    os.remove(path)
    assert exit_code == 0
    lines = captured.out.strip().split('\n')
    assert lines == ['0', '0']
    assert captured.err == ''

def test_decode_error(monkeypatch, capsys):
    # Write invalid UTF-8
    path = _write_temp_bytes(b'\xff\xfe\xfd')
    exit_code = main([path])
    captured = capsys.readouterr()
    os.remove(path)
    assert exit_code == 1
    assert captured.out == ''
    assert captured.err == 'error: cannot read file\n'

def test_file_not_found(monkeypatch, capsys):
    # File does not exist
    path = '/nonexistent/path/to/file.txt'
    exit_code = main([path])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ''
    assert captured.err == 'error: cannot read file\n'

def test_word_boundary_in_nonalpha(monkeypatch, capsys):
    text = '!foo-bar@baz.'
    # Only 'foo', 'bar', 'baz' should be detected as words
    path = _write_temp_text(text)
    exit_code = main([path])
    captured = capsys.readouterr()
    os.remove(path)
    assert exit_code == 0
    lines = captured.out.strip().split('\n')
    # Casefolded words: ['foo', 'bar', 'baz'], all frequency 1; smallest is 'bar'
    assert lines == ['3', 'bar 1']
    assert captured.err == ''

def test_large_input_multiple_words(monkeypatch, capsys):
    words = ['Apple', 'Banana', 'apple', 'banana', 'banana', 'Cherry']
    # Apple: 2, Banana: 3, Cherry:1 (after casefold)
    path = _write_temp_text(' '.join(words))
    exit_code = main([path])
    captured = capsys.readouterr()
    os.remove(path)
    assert exit_code == 0
    lines = captured.out.strip().split('\n')
    # Expect 6, and banana 3
    assert lines == ['6', 'banana 3']
    assert captured.err == ''
