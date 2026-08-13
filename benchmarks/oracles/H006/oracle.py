import json

import pipeline


def test_sales_pipeline_contract(tmp_path):
    source = tmp_path / "sales.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    summary = tmp_path / "summary.json"
    source.write_text(
        '\n'.join([
            '{"customer_id":"A","amount":10.5}',
            '{"customer_id":"B","amount":3}',
            'broken',
            '{"customer_id":"A","amount":2.5}',
        ]) + '\n',
        encoding="utf-8",
    )

    assert pipeline.run(str(source), str(quarantine), str(summary)) == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["A"] == {"total_amount": 13.0, "transaction_count": 2}
    assert payload["B"] == {"total_amount": 3.0, "transaction_count": 1}
    assert len(quarantine.read_text(encoding="utf-8").splitlines()) == 1
