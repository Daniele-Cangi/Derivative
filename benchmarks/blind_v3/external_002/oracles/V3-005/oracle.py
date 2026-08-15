import io
import sys
import json
import pytest
import builtins
from contextlib import redirect_stdout

import jsoncompact

@pytest.fixture
def capture_stdout():
    f = io.StringIO()
    with redirect_stdout(f):
        yield f

@pytest.fixture
def example_input_and_expected():
    input_json = '[{"a":1,"b":null,"c":""},{"x":"value","y":null,"z":"nonempty"}, {}]'
    expected_output = '[{"a":1}, {"x":"value", "z":"nonempty"}, {}]'
    return input_json, expected_output

def test_compact_removes_null_and_empty_string(tmp_path, example_input_and_expected, capture_stdout):
    input_json, expected_output = example_input_and_expected

    # prepare stdin with input_json
    stdin_backup = sys.stdin
    sys.stdin = io.StringIO(input_json)

    try:
        # call main similarly to CLI
        jsoncompact.main()

        output = capture_stdout.getvalue()
        output_json = json.loads(output)
        expected_json = json.loads(expected_output)

        assert output_json == expected_json
    finally:
        sys.stdin = stdin_backup


def test_empty_array_input(capture_stdout):
    stdin_backup = sys.stdin
    sys.stdin = io.StringIO('[]')
    try:
        jsoncompact.main()
        output = capture_stdout.getvalue()
        output_json = json.loads(output)
        assert output_json == []
    finally:
        sys.stdin = stdin_backup


def test_invalid_json_exits_nonzero(monkeypatch):
    # Patch sys.exit to capture exit calls
    exit_calls = {}
    def fake_exit(code):
        exit_calls['code'] = code
        raise SystemExit(code)

    monkeypatch.setattr(sys, 'exit', fake_exit)
    stdin_backup = sys.stdin
    sys.stdin = io.StringIO('{invalid json}')
    with pytest.raises(SystemExit):
        jsoncompact.main()
    sys.stdin = stdin_backup
    assert 'code' in exit_calls
    assert exit_calls['code'] != 0


def test_no_null_no_empty_remain(capture_stdout):
    json_in = '[{"k1":"v1", "k2":123, "k3":false}]'
    expected = '[{"k1":"v1", "k2":123, "k3":false}]'
    stdin_backup = sys.stdin
    sys.stdin = io.StringIO(json_in)
    try:
        jsoncompact.main()
        output = capture_stdout.getvalue()
        assert json.loads(output) == json.loads(expected)
    finally:
        sys.stdin = stdin_backup


def test_preserve_order_of_keys(capture_stdout):
    json_in = '[{"a":null, "b":"", "c":1, "d":"keep"}]'
    expected = '[{"c":1, "d":"keep"}]'
    stdin_backup = sys.stdin
    sys.stdin = io.StringIO(json_in)
    try:
        jsoncompact.main()
        output = capture_stdout.getvalue()
        out_obj = json.loads(output)
        keys = list(out_obj[0].keys())
        assert keys == ['c', 'd']
    finally:
        sys.stdin = stdin_backup
