import csv

import cli


def test_duplicate_file_cli_contract(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "a.txt").write_text("same", encoding="utf-8")
    (source / "nested").mkdir()
    (source / "nested" / "b.txt").write_text("same", encoding="utf-8")
    (source / "unique.txt").write_text("different", encoding="utf-8")
    output = tmp_path / "duplicates.csv"

    assert cli.main([str(source), str(output)]) == 0
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["duplicate_count"] == "2"
    assert "a.txt" in rows[0]["paths"]
    assert "b.txt" in rows[0]["paths"]
