from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


def is_json_record_sort_cli(plan: FeasiblePlan) -> bool:
    corpus = " ".join(
        [
            plan.build_spec.normalized_requirement,
            *(interface.signature for interface in plan.interfaces),
        ]
    ).lower()
    return all(
        marker in corpus
        for marker in ("json array", "descending score", "ascending id", "duplicate ids")
    )


def render_json_record_sort_file(
    plan: FeasiblePlan,
    path: str,
    interfaces: list[PlanInterface],
) -> str | None:
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith(("src/cli.py", "src/main.py")):
        return _cli_module()
    if normalized.startswith("tests/"):
        return _behavioral_test()
    return None


def render_json_record_sort_test(plan: FeasiblePlan, plan_test: PlanTest) -> str:
    return _behavioral_test()


def _cli_module() -> str:
    return r'''import argparse
import json
import math
from pathlib import Path


def _load_records(input_path: str) -> list[dict[str, object]]:
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("input must contain valid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("input JSON must be an array")

    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(payload):
        if not isinstance(record, dict) or set(record) != {"id", "score"}:
            raise ValueError(f"record {index} must contain exactly id and score")
        identifier = record["id"]
        score = record["score"]
        if not isinstance(identifier, str):
            raise ValueError(f"record {index} id must be a string")
        if identifier in seen_ids:
            raise ValueError(f"duplicate id: {identifier}")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"record {index} score must be numeric")
        if isinstance(score, float) and not math.isfinite(score):
            raise ValueError(f"record {index} score must be finite")
        seen_ids.add(identifier)
        records.append(dict(record))
    return records


def _ordered_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(records, key=lambda record: (-record["score"], record["id"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Order scored JSON records")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = _load_records(args.input_path)
    ordered = _ordered_records(records)
    Path(args.output_path).write_text(
        json.dumps(ordered, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _behavioral_test() -> str:
    return r'''import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cli


def test_cli_stably_orders_records_and_writes_json(tmp_path):
    source = tmp_path / "records.json"
    output = tmp_path / "ordered.json"
    source.write_text(
        json.dumps([
            {"id": "b", "score": 10},
            {"id": "a", "score": 10},
            {"id": "c", "score": 12.5},
        ]),
        encoding="utf-8",
    )
    assert cli.main([str(source), str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert [record["id"] for record in result] == ["c", "a", "b"]
    assert result[0]["score"] == 12.5


@pytest.mark.parametrize(
    "payload",
    [
        [{"id": "a", "score": 1}, {"id": "a", "score": 2}],
        [{"id": "a"}],
        [{"id": 1, "score": 2}],
        {"id": "not-an-array", "score": 1},
    ],
)
def test_duplicate_ids_and_malformed_records_raise_value_error(tmp_path, payload):
    source = tmp_path / "records.json"
    output = tmp_path / "ordered.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        cli.main([str(source), str(output)])
'''
