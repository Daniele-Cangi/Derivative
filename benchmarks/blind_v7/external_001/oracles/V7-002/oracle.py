import io
import sys
import pytest
from forge_blind_v7.cli_reverse_nonempty import main

# -- In-memory capture helpers --
class InMemoryIO:
    def __enter__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self
    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

# --- Fixtures ---
@pytest.fixture
def input_file_with_mixed_lines(tmp_path):
    # Non-empty and empty lines, including empty at start/end and multiple in sequence
    content = '\n'.join([
        '',           # 0: empty
        'Line 1',     # 1: nonempty
        '',           # 2: empty
        'Line 2',     # 3: nonempty
        'Line 3',     # 4: nonempty
        '',           # 5: empty
        '',           # 6: empty
        'Line 4',     # 7: nonempty
        ''            # 8: empty, yields a trailing newline
    ])
    file_path = tmp_path / "mixed_lines.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path, content

@pytest.fixture
def input_file_all_empty(tmp_path):
    content = '\n\n\n\n'   # Four empty lines (i.e., three newlines)
    file_path = tmp_path / "all_empty.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path, content

@pytest.fixture
def input_file_unicode(tmp_path):
    # Mixed Unicode, blank lines
    content = '\n'.join([
        '\u3053\u3093\u306b\u3061\u306f',     # Japanese 'Konnichiwa'
        '',
        '\ud83d\ude00 Smile',                  # Emoji + text
        '\u041b\u0438\u043d\u0438\u044f',    # Russian 'Liniya'
        '',
        ''
    ])
    file_path = tmp_path / "unicode_lines.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path, content

# --- Tests ---
def test_reverse_nonempty_preserves_empty_lines(input_file_with_mixed_lines):
    file_path, content = input_file_with_mixed_lines
    lines = content.split('\n')
    nonempty = [l for l in lines if l]
    reversed_nonempty = list(reversed(nonempty))
    ne_iter = iter(reversed_nonempty)
    expected_lines = [l if l == '' else next(ne_iter) for l in lines]
    expected_output = '\n'.join(expected_lines)
    with InMemoryIO() as cap:
        exit_code = main([str(file_path)])
    assert exit_code == 0
    output = cap.stdout.getvalue()
    assert output == expected_output


def test_file_not_found_returns_status_2(tmp_path):
    # Test: invoking main() on non-existent file exits 2 and prints clear error
    missing = tmp_path / "no_such_input.txt"
    with InMemoryIO() as cap:
        exit_code = main([str(missing)])
    assert exit_code == 2
    # The error must be nonempty and mention the file name, but specifics are not required
    err = cap.stderr.getvalue()
    assert str(missing.name) in err
    assert err.strip() != ""


def test_stdin_all_empty_lines_unchanged(monkeypatch):
    # All lines empty: output unchanged
    input_content = '\n\n\n'   # Three empty lines
    monkeypatch.setattr(sys, 'stdin', io.StringIO(input_content))
    with InMemoryIO() as cap:
        exit_code = main([])
    assert exit_code == 0
    assert cap.stdout.getvalue() == input_content


def test_unicode_support_from_file(input_file_unicode):
    file_path, content = input_file_unicode
    lines = content.split('\n')
    nonempty = [l for l in lines if l]
    reversed_nonempty = list(reversed(nonempty))
    ne_iter = iter(reversed_nonempty)
    expected_lines = [l if l == '' else next(ne_iter) for l in lines]
    expected_output = '\n'.join(expected_lines)
    with InMemoryIO() as cap:
        exit_code = main([str(file_path)])
    assert exit_code == 0
    output = cap.stdout.getvalue()
    assert output == expected_output


def test_zero_lines_input_gives_zero_lines(monkeypatch):
    # Zero lines of input (empty file/stdin): zero lines output
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))
    with InMemoryIO() as cap:
        exit_code = main([])
    assert exit_code == 0
    assert cap.stdout.getvalue() == ''


def test_cli_with_no_args_reads_stdin(monkeypatch):
    # One non-empty plus two empties (should remain invariant except possible non-empty reversal)
    test_input = 'a\n\n\n'  # single non-empty followed by two empties
    # Only one non-empty, so output equals input
    expected_output = test_input
    monkeypatch.setattr(sys, 'stdin', io.StringIO(test_input))
    with InMemoryIO() as cap:
        exit_code = main([])
    assert exit_code == 0
    assert cap.stdout.getvalue() == expected_output
