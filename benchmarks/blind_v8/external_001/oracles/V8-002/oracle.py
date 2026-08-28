import pytest
import io
import os
from contextlib import redirect_stdout, redirect_stderr
from uniq_lines import main

# Utility to run the CLI main(argv), capturing exit code, stdout, stderr

def run_cli(argv):
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            ret = main(argv)
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = ret if isinstance(ret, int) else 0
    return code, out.getvalue(), err.getvalue()


def test_empty_input_file(tmp_path):
    path = tmp_path / 'empty.txt'
    path.write_bytes(b'')
    # Direct invocation of the public target in the test as required
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([str(path)])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    assert code == 0
    assert out.getvalue() == ''
    assert err.getvalue() == ''


def test_only_repeated_lines(tmp_path):
    path = tmp_path / 'rep.txt'
    lines = [b'abc def\n', b'abc def\n', b'abc def\n']
    path.write_bytes(b''.join(lines))
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([str(path)])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    expected = lines[0].decode('utf-8')
    assert code == 0
    assert out.getvalue() == expected
    assert err.getvalue() == ''


def test_repeated_empty_lines(tmp_path):
    path = tmp_path / 'emptylines.txt'
    lines = [b'\n', b'\n', b'\n']
    path.write_bytes(b''.join(lines))
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([str(path)])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    expected = '\n'
    assert code == 0
    assert out.getvalue() == expected
    assert err.getvalue() == ''


def test_edge_lineendings_and_whitespace(tmp_path):
    path = tmp_path / 'lineends.txt'
    lines = [b'x\n', b'x \n', b'x\r\n', b'x\t\n', b'x\n', b'x \n']
    path.write_bytes(b''.join(lines))
    seen = set()
    uniq = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    expected = b''.join(uniq).decode('utf-8')
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([str(path)])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    assert code == 0
    assert out.getvalue() == expected
    assert err.getvalue() == ''


def test_crlf_and_lf_empty_lines(tmp_path):
    path = tmp_path / 'crlf_lf.txt'
    lines = [b'\n', b'\r\n', b'\n', b'\r\n']
    path.write_bytes(b''.join(lines))
    seen = set()
    uniq = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    expected = b''.join(uniq).decode('utf-8')
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([str(path)])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    assert code == 0
    assert out.getvalue() == expected
    assert err.getvalue() == ''


def test_unique_lines_and_order(tmp_path):
    path = tmp_path / 'mix.txt'
    lines = [
        b'A\n',    # 0
        b'B\r\n',  # 1
        b'C\n',    # 2
        b'A\n',    # 3
        b'C\n',    # 4
        b'B\r\n',  # 5
        b' D\n',   # 6
        b'\t\n',   # 7
        b' D\n',   # 8
        b'\n'      # 9
    ]
    path.write_bytes(b''.join(lines))
    seen = set()
    uniq = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    expected = b''.join(uniq).decode('utf-8')
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([str(path)])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    assert code == 0
    assert out.getvalue() == expected
    assert err.getvalue() == ''


def test_utf8_decode_error(tmp_path):
    path = tmp_path / 'badutf8.txt'
    content = b'good\nline\n\xe2\x28\xa1\n'
    path.write_bytes(content)
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([str(path)])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    assert code == 1
    assert out.getvalue() == ''
    assert err.getvalue() == 'error: cannot read file'


def test_missing_file():
    missing = 'definitelynotfound_XYYZnope.txt'
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([missing])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    assert code == 1
    assert out.getvalue() == ''
    assert err.getvalue() == 'error: cannot read file'


def test_input_is_dir(tmp_path):
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([str(tmp_path)])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    assert code == 1
    assert out.getvalue() == ''
    assert err.getvalue() == 'error: cannot read file'


def test_no_filename_arg():
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    assert code == 1
    assert out.getvalue() == ''
    assert err.getvalue() == 'error: cannot read file'


def test_file_with_one_line(tmp_path):
    path = tmp_path / 'one.txt'
    path.write_bytes(b'zzTOP\r\n')
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([str(path)])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    assert code == 0
    assert out.getvalue() == 'zzTOP\r\n'
    assert err.getvalue() == ''


def test_file_with_multiple_identical_empty_lines(tmp_path):
    path = tmp_path / 'manyempty.txt'
    lines = [b'\n', b'\r\n', b'\n', b'\r\n', b'\n', b'\r\n']
    path.write_bytes(b''.join(lines))
    seen = set()
    uniq = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    expected = b''.join(uniq).decode('utf-8')
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            result = main([str(path)])
    except SystemExit as se:
        code = se.code if isinstance(se.code, int) else 1
    else:
        code = result if isinstance(result, int) else 0
    assert code == 0
    assert out.getvalue() == expected
    assert err.getvalue() == ''

# END
