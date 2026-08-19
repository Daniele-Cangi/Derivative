# V4-003: Oracle for 'colstats' CLI tool, main(argv: list[str] | None = None) -> int
import io
import os
import sys
import tempfile
import pytest
from colstats import main

# Helper for capturing stderr/stdout by redirecting in the current process
from contextlib import redirect_stdout, redirect_stderr

# --- Fixtures ---
@pytest.fixture
def input_file_path(tmp_path):
    return tmp_path / 'input.tsv'

@pytest.fixture
def output_file_path(tmp_path):
    return tmp_path / 'output.tsv'

# --- Test Cases ---

def test_non_numeric_value_triggers_error(input_file_path, output_file_path):
    # Column 2 has a non-numeric value
    input_file_path.write_text("1\t2.0\n3\tabs\n4\t5\n", encoding='utf-8')
    argv = [str(input_file_path), str(output_file_path)]
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        exit_code = main(argv)
    assert exit_code == 1
    # Should mention column 2 and the value 'abs' in the error message
    assert 'Non-numeric value in column 2: abs' in stderr.getvalue()
    with open(output_file_path, 'rb') as f:
        assert f.read() == b''  # No output file should be written on failure

def test_unequal_rows_and_blank_fields(input_file_path, output_file_path):
    # First line has 2 fields, second 1, third 3 (only 2nd column is empty)
    input_content = "1\t2\n3\n4\t\t6\n"
    input_file_path.write_text(input_content, encoding='utf-8')
    argv = [str(input_file_path), str(output_file_path)]
    exit_code = main(argv)
    assert exit_code == 0
    
    with open(output_file_path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    assert lines[0] == "column\tcount\tmin\tmax\tmean"
    # There are 3 columns in total
    res = [line.split('\t') for line in lines[1:]]
    # Column 1: values 1,3,4 (count=3)
    assert res[0][0] == '1'
    assert res[0][1] == '3'
    assert res[0][2] == '1'
    assert res[0][3] == '4'
    # mean = (1+3+4)/3 = 2.666... to 6 dp
    assert res[0][4] == f"{(1+3+4)/3:.6f}"
    # Column 2: values 2 (first row), empty 2nd/3rd row (count=1)
    assert res[1][0] == '2'
    assert res[1][1] == '1'
    assert res[1][2] == '2'
    assert res[1][3] == '2'
    assert res[1][4] == "2.000000"
    # Column 3: only one value (6, third row)
    assert res[2][0] == '3'
    assert res[2][1] == '1'
    assert res[2][2] == '6'
    assert res[2][3] == '6'
    assert res[2][4] == "6.000000"

def test_empty_input_file(input_file_path, output_file_path):
    input_file_path.write_text("", encoding='utf-8')
    argv = [str(input_file_path), str(output_file_path)]
    exit_code = main(argv)
    assert exit_code == 0
    with open(output_file_path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    assert lines == ["column\tcount\tmin\tmax\tmean"]  # Only header, no columns

def test_blank_lines_and_empty_fields(input_file_path, output_file_path):
    input_file_path.write_text("\n1\t\n\t2\n\n\t\t\n", encoding='utf-8')
    argv = [str(input_file_path), str(output_file_path)]
    exit_code = main(argv)
    assert exit_code == 0
    with open(output_file_path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    # 2 columns detected (row 2: col1=1; row3: col2=2)
    res = [line.split('\t') for line in lines[1:]]
    # col 1: 1 value, 1; col2: 1 value, 2
    assert res[0][0] == '1' and res[0][1] == '1' and res[0][2] == '1' and res[0][3] == '1' and res[0][4] == '1.000000'
    assert res[1][0] == '2' and res[1][1] == '1' and res[1][2] == '2' and res[1][3] == '2' and res[1][4] == '2.000000'

def test_input_path_dash_reads_stdin(output_file_path):
    in_rows = "1\t2\n3\t4\n"
    stdin_buf = io.StringIO(in_rows)
    argv = ['-', str(output_file_path)]
    old_stdin = sys.stdin
    sys.stdin = stdin_buf
    try:
        exit_code = main(argv)
    finally:
        sys.stdin = old_stdin
    assert exit_code == 0
    with open(output_file_path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    assert lines[0] == "column\tcount\tmin\tmax\tmean"
    res = [line.split('\t') for line in lines[1:]]
    assert res[0][0] == '1' and res[0][1] == '2' and res[0][4] == '2.000000'
    assert res[1][0] == '2' and res[1][1] == '2' and res[1][4] == '3.000000'

def test_output_path_dash_writes_stdout(input_file_path, capsys):
    # Output is printed to stdout, input is a file
    input_file_path.write_text("1\t2\n3\t4\n", encoding='utf-8')
    argv = [str(input_file_path), '-']
    exit_code = main(argv)
    assert exit_code == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines[0] == "column\tcount\tmin\tmax\tmean"
    # Confirm means, counts, and presence of both columns in output
    assert any(line.startswith('1\t2\t1') for line in lines)
    assert any(line.startswith('2\t2\t2') for line in lines)


def test_input_file_not_found(output_file_path):
    missing = '/no/such/file/hopefully.tsv'
    argv = [missing, str(output_file_path)]
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        exit_code = main(argv)
    assert exit_code == 1
    assert f'Input file not found or unreadable: {missing}' in stderr.getvalue()
    # No output file should be written
    assert not os.path.exists(output_file_path)

def test_both_input_and_output_dash_raises_usage():
    argv = ['-', '-']
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        with pytest.raises(SystemExit):
            ec = main(argv)
    val = stderr.getvalue()
    assert 'usage: colstats INPUT OUTPUT' in val or 'both input and output as -' in val  # Accept either phrasing

@pytest.mark.parametrize(
    'argv',
    [[], ['foo'], ['foo', 'bar', 'baz']]
)
def test_improper_argument_counts(argv):
    # Should return exit 2 and print usage
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        exit_code = main(argv)
    assert exit_code == 2
    assert 'usage: colstats INPUT OUTPUT' in stderr.getvalue()
