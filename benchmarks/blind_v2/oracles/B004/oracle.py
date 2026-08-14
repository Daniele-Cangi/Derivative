import json

import pytest

import cli


def test_cli_orders_records_and_writes_json(tmp_path):
    source = tmp_path / "records.json"
    output = tmp_path / "ordered.json"
    source.write_text(
        json.dumps(
            [
                {"id": "b", "score": 10},
                {"id": "a", "score": 10},
                {"id": "c", "score": 12.5},
            ]
        ),
        encoding="utf-8",
    )
    assert cli.main([str(source), str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert [row["id"] for row in result] == ["c", "a", "b"]
    assert result[0]["score"] == 12.5


def test_duplicate_ids_and_malformed_records_are_rejected(tmp_path):
    output = tmp_path / "ordered.json"
    for payload in (
        [{"id": "a", "score": 1}, {"id": "a", "score": 2}],
        [{"id": "a"}],
        {"id": "not-an-array", "score": 1},
    ):
        source = tmp_path / "records.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            cli.main([str(source), str(output)])
