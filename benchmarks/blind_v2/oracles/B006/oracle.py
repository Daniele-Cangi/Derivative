import json

import pytest

from pipeline import run


def test_sensor_pipeline_aggregates_and_counts_malformed_events(tmp_path):
    source = tmp_path / "events.jsonl"
    output = tmp_path / "summary.json"
    source.write_text(
        "\n".join(
            [
                '{"sensor_id":"alpha","value":1}',
                '{"sensor_id":"beta","value":-1}',
                "not-json",
                '{"sensor_id":"alpha","value":3}',
                '{"sensor_id":"","value":9}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert run(str(source), str(output)) == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["valid_count"] == 3
    assert summary["malformed_count"] == 2
    assert list(summary["sensors"]) == ["alpha", "beta"]
    assert summary["sensors"]["alpha"] == {
        "count": 2,
        "min": 1,
        "max": 3,
        "mean": pytest.approx(2.0),
    }
    assert summary["sensors"]["beta"] == {
        "count": 1,
        "min": -1,
        "max": -1,
        "mean": pytest.approx(-1.0),
    }
