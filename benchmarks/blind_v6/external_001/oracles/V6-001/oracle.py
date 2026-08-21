import io
import sys
import os
import tempfile
import pytest
from expand_ranges import main

def write_tmp_unicode_file(content):
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception:
        os.unlink(path)
        raise
    return path

@pytest.mark.parametrize("file_content,expected_out,expected_err,expected_code", [
    # Test 1: File contains '1-3\n#foo\n5-5\n', outputs 1\n2\n3\n5, code 0.
    ('1-3\n#foo\n5-5\n', '1\n2\n3\n5\n', '', 0),
    # Test 2: File contains '0-0\n2-1\n', outputs 0, stderr reports line 2 error, code 0.
    ('0-0\n2-1\n', '0\n', 'Line 2: Malformed range\n', 0),
    # Test 3: File contains 'foo\n', outputs nothing, stderr reports line 1 error, code 1.
    ('foo\n', '', 'Line 1: Malformed range\n', 1),
    # Test 4: Blank file: outputs nothing, code 1.
    ('', '', '', 1),
])
def test_expand_ranges_basic_cases(file_content, expected_out, expected_err, expected_code):
    file_path = write_tmp_unicode_file(file_content)
    try:
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            code = main([file_path])
            out = sys.stdout.getvalue()
            err = sys.stderr.getvalue()
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err
    finally:
        os.unlink(file_path)
    assert out == expected_out
    assert err == expected_err
    assert code == expected_code

def test_expand_ranges_duplicate_and_overlap():
    # '1-2\n2-3\n' => 1\n2\n2\n3\n, code 0, empty stderr
    file_path = write_tmp_unicode_file('1-2\n2-3\n')
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        code = main([file_path])
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        os.unlink(file_path)
    expect = ''.join(f"{i}\n" for i in [1,2,2,3])
    assert out == expect
    assert err == ''
    assert code == 0

def test_expand_ranges_non_utf8_file():
    # Write file with invalid utf-8 bytes, expect code 1, no stdout output, some stderr
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(b'1-2\xff\xff')
        # forcibly close file before reading
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        code = main([path])
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        os.unlink(path)
    assert out == ''
    assert code == 1
    assert err.strip() != ''

def test_expand_ranges_comments_and_blanks_only():
    # Only comments and blank lines: outputs nothing, code 1.
    file_path = write_tmp_unicode_file('   \n#blah\n\t \n')
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        code = main([file_path])
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        os.unlink(file_path)
    assert out == ''
    assert err == ''
    assert code == 1

def test_expand_ranges_missing_file():
    path = os.path.join(tempfile.gettempdir(), 'nosuch_ex_ranges_12345.txt')
    if os.path.exists(path):
        os.unlink(path)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    code = main([path])
    out = sys.stdout.getvalue()
    err = sys.stderr.getvalue()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    assert out == ''
    assert code == 1
    assert err.strip() != ''
