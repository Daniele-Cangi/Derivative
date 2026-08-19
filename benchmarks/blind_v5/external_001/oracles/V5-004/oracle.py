# test_pyenvlines.py
import io
import sys
import os
import tempfile
import pytest
from pyenvlines import main

# Util for capturing stdout and stderr in-process
class OutputCapture:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self
    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# Helper: Create a temporary file with given UTF-8 text lines
@pytest.fixture
def temp_file_with_lines():
    files = []
    def _create(lines):
        fd, path = tempfile.mkstemp(suffix='.env')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line)
        files.append(path)
        return path
    yield _create
    for fname in files:
        try: os.remove(fname)
        except FileNotFoundError: pass

# 1. Test: Mixed valid and invalid lines; preserve order, strict pattern, no trailing newlines changed
DEF_MIXED_LINES = [
    'FOO=bar\n',                    # valid
    '123=invalid\n',                # invalid (does not start with letter/_)
    'A1_B2_C3=value123\n',         # valid
    'FOO BAR=fail\n',              # invalid (space in name)
    '_SECRET=topsecret\n',         # valid
    'a_var=notupper\n',            # invalid (lowercase in name)
    'X=1\n',                       # valid (single char name)
    'SPACE =bad\n',                # invalid (space)
    'A=\n',                        # valid (empty value)
    'BADVAR=no\nmore',             # invalid (missing newline)
]

EXPECTED_MIXED = ''.join([
    DEF_MIXED_LINES[0], # 'FOO=bar\n'
    DEF_MIXED_LINES[2], # 'A1_B2_C3=value123\n'
    DEF_MIXED_LINES[4], # '_SECRET=topsecret\n'
    DEF_MIXED_LINES[6], # 'X=1\n'
    DEF_MIXED_LINES[8], # 'A=\n'
])


def test_mixed_valid_and_invalid_lines(temp_file_with_lines):
    file_path = temp_file_with_lines(DEF_MIXED_LINES)
    argv = [file_path]
    with OutputCapture() as cap:
        code = main(argv)
    assert code == 0
    result = cap.stdout.getvalue()
    # Reference: only the lines matching the regex; order preserved, trailing newlines as in file
    assert result == EXPECTED_MIXED
    assert cap.stderr.getvalue() == ''

# 2. Test: File with only valid lines covering allowed patterns
VALID_LINES = [
    'A=1\n',
    '_A_=x\n',
    'FOO123=value\n',
    'BAR_=baz\n',
    'ENV_VAR=abc=def\n',
    'Z=\n',
    'A1B2C3=123\n',
]
EXPECTED_ALL_VALID = ''.join(VALID_LINES)

def test_all_valid_lines(temp_file_with_lines):
    file_path = temp_file_with_lines(VALID_LINES)
    argv = [file_path]
    with OutputCapture() as cap:
        code = main(argv)
    assert code == 0
    # Reference output is just all lines as written, joined
    assert cap.stdout.getvalue() == EXPECTED_ALL_VALID
    assert cap.stderr.getvalue() == ''

# 3. Test: File with only invalid lines (should output nothing, exit 0)
INVALID_LINES = [
    'a=1\n',                    # lowercase start
    '=novar\n',                 # no name
    'FOO-BAR=oops\n',           # dash
    ' FOO=badspace\n',          # leading space
    'FOO_BAR =badspace\n',      # space before =
    'FOObar=badcase\n',         # lowercase in name
    'FOO=bar extra\n',          # space in value okay, but only if no extra after =
    'X\n',                      # missing =
]

def test_invalid_lines_only(temp_file_with_lines):
    file_path = temp_file_with_lines(INVALID_LINES)
    argv = [file_path]
    with OutputCapture() as cap:
        code = main(argv)
    assert code == 0
    assert cap.stdout.getvalue() == ''
    assert cap.stderr.getvalue() == ''

# 4. Test: Empty file (output nothing, exit 0)
def test_empty_file(temp_file_with_lines):
    file_path = temp_file_with_lines([])
    argv = [file_path]
    with OutputCapture() as cap:
        code = main(argv)
    assert code == 0
    assert cap.stdout.getvalue() == ''
    assert cap.stderr.getvalue() == ''

# 5. Test: File does not exist (should exit 1, print error to stderr)
def test_missing_file():
    fake_path = '/tmp/pyenvlines_notfound_' + next(tempfile._get_candidate_names())
    argv = [fake_path]
    with OutputCapture() as cap:
        code = main(argv)
    assert code == 1
    # stderr must not be empty and must mention filename
    err = cap.stderr.getvalue()
    assert fake_path in err
    assert err.strip() != ''
    assert cap.stdout.getvalue() == ''

# 6. Test: File cannot be opened (permissions error) -- create, chmod 0, try to open, cleanup
import stat
def test_unreadable_file(temp_file_with_lines):
    file_path = temp_file_with_lines(['FOO=1\n'])
    old_mode = os.stat(file_path).st_mode
    os.chmod(file_path, 0)
    try:
        argv = [file_path]
        with OutputCapture() as cap:
            code = main(argv)
        assert code == 1
        err = cap.stderr.getvalue()
        assert file_path in err
        assert err.strip() != ''
        assert cap.stdout.getvalue() == ''
    finally:
        os.chmod(file_path, old_mode)
