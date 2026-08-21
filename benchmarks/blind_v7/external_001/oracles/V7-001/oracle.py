import io
import os
import sys
import tempfile
import pytest
from forge_blind_v7.cli_dedupe_adjacent import main

# Fixture: Distinct content
@pytest.fixture
def input_adjacent_duplicates():
    # Unicode, empty, and duplicate lines mixed
    return (
        "foo\n"
        "foo\n"
        "bar\n"
        "baz\n"
        "baz\n"
        "baz\n"
        "qux\n"
        "\n"
        "\n"
        "λambda\n"
        "lambda\n"
        "λambda\n"
        "λambda\n"
        "\n"
        "the end\n"
    )

@pytest.fixture
def expected_no_adjacent_duplicates():
    # Compute deduplication reference for above input
    orig = [
        "foo\n", "foo\n", "bar\n", "baz\n", "baz\n", "baz\n", "qux\n", "\n", "\n", "λambda\n", "lambda\n", "λambda\n", "λambda\n", "\n", "the end\n"
    ]
    out = []
    last = None
    for line in orig:
        if line != last:
            out.append(line)
        last = line
    return ''.join(out)

# Helper to run main() with redirected std streams
class redirect_std:
    def __init__(self, in_content=None):
        self.in_content = in_content
        self.stdin = None
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self._oldout = None
        self._olderr = None
        self._oldin = None
    def __enter__(self):
        self._oldout = sys.stdout
        self._olderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if self.in_content is not None:
            self.stdin = io.StringIO(self.in_content)
            self._oldin = sys.stdin
            sys.stdin = self.stdin
    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self._oldout
        sys.stderr = self._olderr
        if self.in_content is not None and self._oldin is not None:
            sys.stdin = self._oldin

# Test CLI and function: file input, dedupes adjacents
def test_main_file_argument_dedupes(input_adjacent_duplicates, expected_no_adjacent_duplicates):
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as tf:
        tf.write(input_adjacent_duplicates)
        tf.flush()
        fname = tf.name
    try:
        with redirect_std() as redir:
            rv = main([fname])
        assert rv == 0
        assert redir.stdout.getvalue() == expected_no_adjacent_duplicates
        assert redir.stderr.getvalue() == ''
    finally:
        os.remove(fname)

# Test CLI: stdin input, including Unicode and empty lines
def test_main_stdin_argument_dedupes(input_adjacent_duplicates, expected_no_adjacent_duplicates):
    with redirect_std(in_content=input_adjacent_duplicates) as redir:
        rv = main([])
    assert rv == 0
    assert redir.stdout.getvalue() == expected_no_adjacent_duplicates
    assert redir.stderr.getvalue() == ''

# Test: Non-existent file emits error, exit code 2, no output to stdout
def test_main_file_not_found():
    fakefile = "nonexistent_test_file_xyz.txt"
    assert not os.path.exists(fakefile)
    with redirect_std() as redir:
        rv = main([fakefile])
    # Exit code is 2, error to stderr, nothing to stdout
    assert rv == 2
    err = redir.stderr.getvalue()
    assert fakefile in err and "No such file" in err or "not found" in err or "does not exist" in err
    assert redir.stdout.getvalue() == ''

# Test: Empty input yields empty output for both CLI and function
@pytest.mark.parametrize("argv,stdin_mode", [([], True), (None, True)])
def test_main_empty_input_outputs_empty(argv, stdin_mode):
    with redirect_std(in_content="" if stdin_mode else None) as redir:
        rv = main(argv)
    assert rv == 0
    assert redir.stdout.getvalue() == ''
    assert redir.stderr.getvalue() == ''

# Test: Idempotence of main for equivalent argv
def test_main_idempotence(input_adjacent_duplicates):
    # Run twice with same args/inputs, outputs must match, so must exit code
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as tf:
        tf.write(input_adjacent_duplicates)
        tf.flush()
        fname = tf.name
    try:
        with redirect_std() as r1:
            rv1 = main([fname])
        with redirect_std() as r2:
            rv2 = main([fname])
        assert (rv1, r1.stdout.getvalue(), r1.stderr.getvalue()) == (rv2, r2.stdout.getvalue(), r2.stderr.getvalue())
    finally:
        os.remove(fname)

# Test: Output encodes valid Unicode (including astral plane and combining)
def test_main_unicode_variety():
    unicode_input = ''.join([
        'é\n',            # e + combining acute
        'é\n',            # repeat
        '𝔘𝔫𝔦𝔠𝔬𝔡𝔢\n',    # Fancy Unicode letters
        '𝔘𝔫𝔦𝔠𝔬𝔡𝔢\n',    # Repeat
        'unicode\n',
        'Αθήνα\n',         # Greek Athens
        'Αθήνα\n',         # Repeat
        'Москва\n',        # Moscow in Cyrillic
        'Москва\n',        # Repeat
    ])
    # Reference deduplication (exact as for the main requirement)
    orig = unicode_input.splitlines(keepends=True)
    out = []
    last = None
    for line in orig:
        if line != last:
            out.append(line)
        last = line
    expected = ''.join(out)
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as tf:
        tf.write(unicode_input)
        tf.flush()
        fname = tf.name
    try:
        with redirect_std() as redir:
            rv = main([fname])
        assert rv == 0
        assert redir.stdout.getvalue() == expected
        assert redir.stderr.getvalue() == ''
    finally:
        os.remove(fname)

# Test: File input with only adjacent duplicates, all lines same
def test_main_file_all_duplicates():
    input_lines = "repeat\nrepeat\nrepeat\nrepeat\n"
    expected = "repeat\n"
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as tf:
        tf.write(input_lines)
        tf.flush()
        fname = tf.name
    try:
        with redirect_std() as redir:
            rv = main([fname])
        assert rv == 0
        assert redir.stdout.getvalue() == expected
        assert redir.stderr.getvalue() == ''
    finally:
        os.remove(fname)

# Test: CLI with one non-adjacent duplicate (should preserve both)
def test_main_non_adjacent_duplicates():
    lines = ["x\n", "y\n", "x\n", "z\n", "x\n"]
    input_data = ''.join(lines)
    expected = input_data
    with redirect_std(in_content=input_data) as redir:
        rv = main([])
    assert rv == 0
    assert redir.stdout.getvalue() == expected
    assert redir.stderr.getvalue() == ''

# Edge: Arbitrary single line input
@pytest.mark.parametrize("s", ["abc\n", "\ufeffBOMtest\n", "λ\n", "\n"])
def test_main_single_line(s):
    with redirect_std(in_content=s) as redir:
        rv = main([])
    assert rv == 0
    assert redir.stdout.getvalue() == s
    assert redir.stderr.getvalue() == ''

# Edge: File input is empty
def test_main_file_empty():
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as tf:
        tf.flush()
        fname = tf.name
    try:
        with redirect_std() as redir:
            rv = main([fname])
        assert rv == 0
        assert redir.stdout.getvalue() == ''
        assert redir.stderr.getvalue() == ''
    finally:
        os.remove(fname)

# CLI: Too many arguments should result in error (should exit not 0)
def test_main_too_many_arguments():
    with redirect_std() as redir:
        rv = main(['file1', 'file2'])
    assert rv != 0
    assert redir.stdout.getvalue() == ''
    err = redir.stderr.getvalue()
    assert "usage" in err.lower() or "argument" in err.lower() or "too many" in err.lower()
