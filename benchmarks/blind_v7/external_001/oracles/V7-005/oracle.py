import io
import os
import sys
import tempfile
import pytest
from collections import Counter
from forge_blind_v7.cli_first_unique_nonempty_line import main

def first_unique_nonempty_line(text):
    lines = text.splitlines(keepends=True)
    nonempty = [ln for ln in lines if ln.strip()]
    cnt = Counter(nonempty)
    for ln in lines:
        if ln.strip() and cnt[ln] == 1:
            return ln
    return ''

def test_returns_first_unique_nonempty_line_from_file():
    input_text = 'a\nb\na\n'
    expect = first_unique_nonempty_line(input_text)
    with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False, newline='') as tf:
        tf.write(input_text)
        tf.flush()
        fname = tf.name
    try:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        code = main([fname])
        std_out = sys.stdout.getvalue()
        std_err = sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        os.unlink(fname)
    assert code == 0
    assert std_out == expect
    assert std_err == ''

def test_unique_among_unicode_lines_from_stdin():
    # beta, alpha, alpha, gamma, blank, beta
    input_text = '\u03b2\n\u03b1\n\u03b1\n\u03b3\n\n\u03b2\n'
    expect = first_unique_nonempty_line(input_text)
    stdin = io.StringIO(input_text)
    old_in, old_out, old_err = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = stdin, io.StringIO(), io.StringIO()
    code = main([])
    std_out = sys.stdout.getvalue()
    std_err = sys.stderr.getvalue()
    sys.stdin, sys.stdout, sys.stderr = old_in, old_out, old_err
    assert code == 0
    assert std_out == expect
    assert std_err == ''

def test_all_lines_only_whitespace_outputs_nothing():
    input_text = ' \n\t\n\r\n\u2003\n'
    expect = first_unique_nonempty_line(input_text)
    with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False, newline='') as tf:
        tf.write(input_text)
        tf.flush()
        fname = tf.name
    try:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        code = main([fname])
        std_out = sys.stdout.getvalue()
        std_err = sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        os.unlink(fname)
    assert code == 0
    assert std_out == ''
    assert expect == ''
    assert std_err == ''

def test_no_unique_lines_all_duplicates():
    input_text = 'repeat\nrepeat\nrepeat\n'
    expect = first_unique_nonempty_line(input_text)
    stdin = io.StringIO(input_text)
    old_in, old_out, old_err = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = stdin, io.StringIO(), io.StringIO()
    code = main([])
    std_out = sys.stdout.getvalue()
    std_err = sys.stderr.getvalue()
    sys.stdin, sys.stdout, sys.stderr = old_in, old_out, old_err
    assert code == 0
    assert std_out == ''
    assert expect == ''
    assert std_err == ''

def test_empty_input_returns_empty():
    input_text = ''
    expect = first_unique_nonempty_line(input_text)
    with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False, newline='') as tf:
        tf.write(input_text)
        tf.flush()
        fname = tf.name
    try:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        code = main([fname])
        std_out = sys.stdout.getvalue()
        std_err = sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        os.unlink(fname)
    assert code == 0
    assert std_out == ''
    assert expect == ''
    assert std_err == ''

def test_nonexistent_file_exits_2_and_error_stderr():
    missing = os.path.join(tempfile.gettempdir(), 'nonexistent_file_test_123.txt')
    if os.path.exists(missing):
        os.unlink(missing)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    code = main([missing])
    std_out = sys.stdout.getvalue()
    std_err = sys.stderr.getvalue()
    sys.stdout, sys.stderr = old_out, old_err
    assert code == 2
    assert std_out == ''
    assert std_err != ''
    assert any(s in std_err.lower() for s in ('no such file', 'not found', 'no such', 'cannot open'))

def test_multiple_unique_lines_only_first_is_output():
    input_text = 'red\ngreen\nblue\nred\n'
    expect = first_unique_nonempty_line(input_text)
    stdin = io.StringIO(input_text)
    old_in, old_out, old_err = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = stdin, io.StringIO(), io.StringIO()
    code = main([])
    std_out = sys.stdout.getvalue()
    std_err = sys.stderr.getvalue()
    sys.stdin, sys.stdout, sys.stderr = old_in, old_out, old_err
    assert code == 0
    assert std_out == expect
    # For this input, nonempty lines: [red, green, blue, red], first unique: green
    assert expect == 'green\n'
    assert std_err == ''

def test_preserves_original_whitespace_and_tabs():
    input_text = '   \tfoo\t  \nbar\n   \tfoo\t  \n'
    expect = first_unique_nonempty_line(input_text)
    with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False, newline='') as tf:
        tf.write(input_text)
        tf.flush()
        fname = tf.name
    try:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        code = main([fname])
        std_out = sys.stdout.getvalue()
        std_err = sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        os.unlink(fname)
    assert code == 0
    assert std_out == expect
    assert expect == 'bar\n'
    assert std_err == ''

def test_crlf_and_unicode_lines():
    # Greek alpha, beta, alpha, delta, beta, with CRLF endings
    input_text = '\u03b1\r\n\u03b2\r\n\u03b1\r\n\u03b4\r\n\u03b2\r\n'
    expect = first_unique_nonempty_line(input_text)
    with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False, newline='') as tf:
        tf.write(input_text)
        tf.flush()
        fname = tf.name
    try:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        code = main([fname])
        std_out = sys.stdout.getvalue()
        std_err = sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        os.unlink(fname)
    assert code == 0
    assert std_out == expect
    assert std_err == ''
    # The unique line should be delta\r\n (\u03b4\r\n)

def test_more_than_one_argument_is_refused():
    # Should fail if more than one argument is passed
    args = ['file1.txt', 'file2.txt']
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    code = main(args)
    std_out = sys.stdout.getvalue()
    std_err = sys.stderr.getvalue()
    sys.stdout, sys.stderr = old_out, old_err
    assert code != 0
    assert std_out == ''
    assert std_err != ''
    # Should print error about arguments
    assert 'usage' in std_err.lower() or 'argument' in std_err.lower()

def test_utf8_with_non_bmp_and_blank_lines_first_unique():
    # Non-BMP Unicode character and blank/space lines
    pileofpoo = chr(0x1F4A9)
    input_text = f'\n{pileofpoo}\n   \nfoo\n{pileofpoo}\nfoo\n'
    expect = first_unique_nonempty_line(input_text)
    stdin = io.StringIO(input_text)
    old_in, old_out, old_err = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = stdin, io.StringIO(), io.StringIO()
    code = main([])
    std_out = sys.stdout.getvalue()
    std_err = sys.stderr.getvalue()
    sys.stdin, sys.stdout, sys.stderr = old_in, old_out, old_err
    # There are two nonempty: pileofpoo+'\n' and 'foo\n', each occurs twice, so nothing is unique
    assert code == 0
    assert std_out == ''
    assert expect == ''
    assert std_err == ''

def test_utf8_with_non_bmp_and_blank_lines_first_unique_exists():
    pileofpoo = chr(0x1F4A9)
    input_text = f'{pileofpoo}\nfoo\nbar\nfoo\n'
    expect = first_unique_nonempty_line(input_text)
    stdin = io.StringIO(input_text)
    old_in, old_out, old_err = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = stdin, io.StringIO(), io.StringIO()
    code = main([])
    std_out = sys.stdout.getvalue()
    std_err = sys.stderr.getvalue()
    sys.stdin, sys.stdout, sys.stderr = old_in, old_out, old_err
    # Unique lines: pileofpoo+'\n', 'bar\n' (foo\n appears twice), so output: pileofpoo+'\n' (first unique)
    assert code == 0
    assert std_out == expect
    assert expect == pileofpoo + '\n'
    assert std_err == ''
