# test_pycolmask_oracle.py
import io
import sys
import os
import tempfile
import pytest
from pycolmask import main


def _write_csv(path, content):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)


def _read_csv_remove_cols(original_csv: str, mask: list[int]) -> str:
    # Remove columns by python reference logic for expected output, keeping delimiter locations.
    # We do NOT trim whitespace. Use only standard library.
    lines = original_csv.splitlines(keepends=True)
    if not lines:
        return "\n"
    header = lines[0]
    cols = header.rstrip('\n\r').split(',')
    # If all columns are masked, output '\n'
    if mask and set(mask) == set(range(len(cols))):
        return "\n"
    keep_indices = [i for i in range(len(cols)) if i not in mask]
    parts = []
    for line in lines:
        line_raw = line.rstrip('\n\r')
        fields = line_raw.split(',')
        kept = [fields[i] for i in keep_indices]
        parts.append(','.join(kept) + ('\n' if line.endswith('\n') or line.endswith('\r') else ''))
    return ''.join(parts)


def test_nominal_masking(tmp_path, monkeypatch):
    csv_content = 'A,  B ,C\n1,x,2\n3, y,4\n'
    mask_indices = [1] # remove column 1 ("  B ")
    file_path = tmp_path / 'in.csv'
    _write_csv(file_path, csv_content)
    # Build CLI args
    argv = ['pycolmask', str(file_path), '--mask=1']
    capout = io.StringIO()
    caperr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', capout)
    monkeypatch.setattr(sys, 'stderr', caperr)
    rc = main(argv)
    assert rc == 0
    # Calculate expected output independently
    expect = _read_csv_remove_cols(csv_content, mask_indices)
    assert capout.getvalue() == expect
    assert caperr.getvalue() == ''


def test_all_columns_masked(tmp_path, monkeypatch):
    csv_content = 'X,Y,Z\n1,2,3\n4,5,6\n'
    file_path = tmp_path / 'allmask.csv'
    _write_csv(file_path, csv_content)
    # Mask all columns (indices 0,1,2)
    argv = ['pycolmask', str(file_path), '--mask=0,1,2']
    capout = io.StringIO()
    caperr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', capout)
    monkeypatch.setattr(sys, 'stderr', caperr)
    rc = main(argv)
    assert rc == 0
    assert capout.getvalue() == '\n'
    assert caperr.getvalue() == ''


def test_invalid_mask_index_too_high(tmp_path, monkeypatch):
    csv_content = 'a,b,c\n1,2,3\n'
    file_path = tmp_path / 'badmask.csv'
    _write_csv(file_path, csv_content)
    # Try to mask column 5 (only 0,1,2 exist)
    argv = ['pycolmask', str(file_path), '--mask=2,5']
    capout = io.StringIO()
    caperr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', capout)
    monkeypatch.setattr(sys, 'stderr', caperr)
    rc = main(argv)
    assert rc == 1
    assert capout.getvalue() == ''  # Expect error to stderr only
    err = caperr.getvalue()
    assert 'mask' in err or 'column' in err or 'invalid' in err  # Some error clue


def test_empty_file(tmp_path, monkeypatch):
    file_path = tmp_path / 'empty.csv'
    _write_csv(file_path, '')
    argv = ['pycolmask', str(file_path), '--mask=0']
    capout = io.StringIO()
    caperr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', capout)
    monkeypatch.setattr(sys, 'stderr', caperr)
    rc = main(argv)
    assert rc == 0
    assert capout.getvalue() == '\n'
    assert caperr.getvalue() == ''


def test_malformed_mask_nonint(tmp_path, monkeypatch):
    csv_content = 'a,b\n1,2\n'
    file_path = tmp_path / 'badmask2.csv'
    _write_csv(file_path, csv_content)
    argv = ['pycolmask', str(file_path), '--mask=a,1']
    capout = io.StringIO()
    caperr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', capout)
    monkeypatch.setattr(sys, 'stderr', caperr)
    rc = main(argv)
    assert rc == 1
    assert capout.getvalue() == ''
    err = caperr.getvalue()
    assert 'mask' in err or 'integer' in err or 'invalid' in err


def test_missing_input_path(monkeypatch):
    argv = ['pycolmask', '--mask=1']
    capout = io.StringIO()
    caperr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', capout)
    monkeypatch.setattr(sys, 'stderr', caperr)
    rc = main(argv)
    assert rc == 1
    assert capout.getvalue() == ''
    err = caperr.getvalue()
    assert 'file' in err or 'path' in err or 'argument' in err


def test_unopenable_file(tmp_path, monkeypatch):
    # Path exists but no read permissions
    file_path = tmp_path / 'noperm.csv'
    _write_csv(file_path, 'foo,bar\n1,2\n')
    os.chmod(file_path, 0o000)
    try:
        argv = ['pycolmask', str(file_path), '--mask=0']
        capout = io.StringIO()
        caperr = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', capout)
        monkeypatch.setattr(sys, 'stderr', caperr)
        rc = main(argv)
        assert rc == 1
        assert capout.getvalue() == ''
        err = caperr.getvalue()
        assert 'file' in err or 'open' in err or 'permission' in err
    finally:
        os.chmod(file_path, 0o644)


def test_row_with_wrong_column_count(tmp_path, monkeypatch):
    csv_content = 'a,b,c\n1,2\n4,5,6\n'
    file_path = tmp_path / 'malformed.csv'
    _write_csv(file_path, csv_content)
    argv = ['pycolmask', str(file_path), '--mask=2']
    capout = io.StringIO()
    caperr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', capout)
    monkeypatch.setattr(sys, 'stderr', caperr)
    rc = main(argv)
    assert rc == 1
    assert capout.getvalue() == ''
    err = caperr.getvalue()
    assert 'row' in err or 'column' in err or 'malformed' in err


def test_missing_mask_argument(tmp_path, monkeypatch):
    file_path = tmp_path / 'foo.csv'
    _write_csv(file_path, 'x,y\n1,2\n')
    argv = ['pycolmask', str(file_path)]  # No --mask provided
    capout = io.StringIO()
    caperr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', capout)
    monkeypatch.setattr(sys, 'stderr', caperr)
    rc = main(argv)
    assert rc == 1
    assert capout.getvalue() == ''
    err = caperr.getvalue()
    assert 'mask' in err or 'argument' in err


def test_header_only_row(tmp_path, monkeypatch):
    csv_content = 'a,b,c\n'
    mask_indices = [1]
    file_path = tmp_path / 'headeronly.csv'
    _write_csv(file_path, csv_content)
    argv = ['pycolmask', str(file_path), '--mask=1']
    capout = io.StringIO()
    caperr = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', capout)
    monkeypatch.setattr(sys, 'stderr', caperr)
    rc = main(argv)
    assert rc == 0
    expect = _read_csv_remove_cols(csv_content, mask_indices)
    assert capout.getvalue() == expect
    assert caperr.getvalue() == ''
