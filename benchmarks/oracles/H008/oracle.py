import json

import pytest

import cli


def test_recursive_json_merge_contract(tmp_path):
    base = tmp_path / "base.json"
    override = tmp_path / "override.json"
    output = tmp_path / "merged.json"
    base.write_text('{"db":{"host":"localhost","port":5432},"tags":["base"]}', encoding="utf-8")
    override.write_text('{"db":{"port":6432},"tags":["override"]}', encoding="utf-8")

    assert cli.main([str(base), str(override), str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "db": {"host": "localhost", "port": 6432},
        "tags": ["override"],
    }


def test_non_object_root_is_rejected(tmp_path):
    base = tmp_path / "base.json"
    override = tmp_path / "override.json"
    output = tmp_path / "merged.json"
    base.write_text("[]", encoding="utf-8")
    override.write_text("{}", encoding="utf-8")
    with pytest.raises((TypeError, ValueError, SystemExit)):
        cli.main([str(base), str(override), str(output)])
