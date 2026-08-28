import os
import sys
import io
import tempfile
import pytest
from col_run_length_encode import main

# Helper: reference run-length encoding transformation on a Unicode str (excluding the line ending)
def ref_rle(line):
    if not line:
        return ''
    result = []
    prev = line[0]
    count = 1
    for c in line[1:]:
        if c == prev:
            count += 1
        else:
            result.append(f"{count}{prev}")
            prev = c
            count = 1
    result.append(f"{count}{prev}")
    return ''.join(result)

# Helper: returns list of (content, line_ending) pairs from file content
import re
LINE_RE = re.compile(r'(.*?)(\r\n|\n|\r|$)')
def split_lines_raw(s):
    # Returns non-empty pairs (content, line_ending)
    lines = []
    idx = 0
    while idx < len(s):
        m = LINE_RE.match(s, idx)
        if m:
            content, ending = m.group(1), m.group(2)
            if ending == '' and content == '':
                break
            lines.append((content, ending))
            idx = m.end()
        else:
            break
    return lines

# pytest fixture for a temporary UTF-8 file with text content
@pytest.fixture
def utf8_file(tmp_path):
    def create_file(content_bytes):
        fpath = tmp_path / "input.txt"
        fpath.write_bytes(content_bytes)
        return str(fpath)
    return create_file

# pytest fixture to patch sys.argv, sys.stdout, and sys.stderr
class ArgvOutErr:
    def __init__(self, argv):
        self.argv = argv
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.old_argv = sys.argv
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
    def __enter__(self):
        sys.argv = self.argv[:]
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.argv = self.old_argv
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        return False

# Test: Correct encoding for mixed input, all edge cases and Unicode
@pytest.mark.parametrize("lines_and_endings", [
    # multiple lines, empty, single-char, multi-char, Unicode
    [
        ("aabcc", "\n"),
        ("", "\r\n"),
        ("x", "\n"),
        ("\U0001F600\U0001F600\U0001F47B\U0001F600", "\r"),  # emoji
        ("", ""),
    ],
    # single line, only spaces & punctuation
    [
        ("   !!!", "\n"),
    ],
    # only one empty line
    [
        ("", "\r"),
    ]
])
def test_run_length_encoding_lines(utf8_file, lines_and_endings):
    raw_content = ''.join(line + ending for line, ending in lines_and_endings)
    content_bytes = raw_content.encode('utf-8')
    filename = utf8_file(content_bytes)
    with ArgvOutErr(["col_run_length_encode", filename]) as env:
        exit_code = main(sys.argv)
    # Prepare expected stdout
    expected = []
    for line, ending in lines_and_endings:
        if ending == '':
            continue  # no output for trailing empty
        if line == '':
            expected.append(ending)
        else:
            expected.append(ref_rle(line) + ending)
    expect_stdout = ''.join(expected)
    assert env.stdout.getvalue() == expect_stdout
    assert env.stderr.getvalue() == ''
    assert exit_code == 0

# Test: Reads file with all supported line endings and handles an empty file
@pytest.mark.parametrize("content_bytes, expect_stdout, expect_code", [
    (b"\n\r\r\n", "\n\r\r\n", 0),   # All line endings, all empty lines
    (b"", "", 0),                           # Completely empty file
])
def test_empty_and_blank_lines(utf8_file, content_bytes, expect_stdout, expect_code):
    filename = utf8_file(content_bytes)
    with ArgvOutErr(["col_run_length_encode", filename]) as env:
        exit_code = main(sys.argv)
    assert env.stdout.getvalue() == expect_stdout
    assert env.stderr.getvalue() == ''
    assert exit_code == expect_code

# Test: File cannot be read or is not valid UTF-8
@pytest.mark.parametrize("bad_bytes", [
    b"abc\x80def",    # Invalid UTF-8
])
def test_file_cannot_be_decoded_or_read(utf8_file, bad_bytes):
    filename = utf8_file(bad_bytes)
    with ArgvOutErr(["col_run_length_encode", filename]) as env:
        exit_code = main(sys.argv)
    assert env.stdout.getvalue() == ''
    assert env.stderr.getvalue().strip() == 'error: cannot read file'
    assert exit_code == 1

# Test: File does not exist
def test_file_does_not_exist():
    filename = '/nonexistent/xxxxyz123.txt'
    with ArgvOutErr(["col_run_length_encode", filename]) as env:
        exit_code = main(sys.argv)
    assert env.stdout.getvalue() == ''
    assert env.stderr.getvalue().strip() == 'error: cannot read file'
    assert exit_code == 1

# Test: Wrong number of arguments
@pytest.mark.parametrize("argv", [
    ["col_run_length_encode"],                  # none
    ["col_run_length_encode", "a", "b"],       # too many
])
def test_invalid_argv_triggers_error(argv):
    with ArgvOutErr(argv) as env:
        exit_code = main(sys.argv)
    assert env.stdout.getvalue() == ''
    assert env.stderr.getvalue().strip() == 'error: invalid input'
    assert exit_code == 1

# Test: Input with multi-codepoint Unicode (disjoint, e.g. combining)
def test_unicode_combining_characters(utf8_file):
    text = 'e\u0301e\u0301e\u0300'  # ééè (combining marks)
    lines_and_endings = [(text, '\n')]
    content_bytes = (text + '\n').encode('utf-8')
    filename = utf8_file(content_bytes)
    with ArgvOutErr(["col_run_length_encode", filename]) as env:
        exit_code = main(sys.argv)
    # The run-length encode is by codepoint, not grapheme, so all 'e' and combining marks are encoded separately.
    just_line = lines_and_endings[0][0]
    expected = ref_rle(just_line) + '\n'
    assert env.stdout.getvalue() == expected
    assert env.stderr.getvalue() == ''
    assert exit_code == 0
