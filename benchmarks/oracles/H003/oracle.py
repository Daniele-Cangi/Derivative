import json

import cli


def test_jsonl_log_cli_contract(tmp_path):
    source = tmp_path / "events.jsonl"
    report = tmp_path / "report.json"
    source.write_text(
        '\n'.join([
            '{"level":"INFO","message":"started"}',
            'not-json',
            '{"level":"ERROR","message":"failed"}',
            '{"level":"INFO","message":"done"}',
        ]) + '\n',
        encoding="utf-8",
    )

    assert cli.main([str(source), str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload == {
        "counts_by_level": {"ERROR": 1, "INFO": 2},
        "malformed_count": 1,
        "total_valid": 3,
    }
