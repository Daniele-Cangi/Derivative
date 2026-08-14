from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


def is_json_log_cli(plan: FeasiblePlan) -> bool:
    corpus = " ".join(
        [
            plan.build_spec.normalized_requirement,
            plan.architecture_summary,
            *(atom.text for atom in plan.build_spec.requirement_atoms),
        ]
    ).lower()
    return all(
        token in corpus
        for token in ("json lines", "level", "message", "total_valid", "counts_by_level")
    )


def render_json_log_file(
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


def render_json_log_test(plan: FeasiblePlan, plan_test: PlanTest) -> str:
    return _behavioral_test()


def _cli_module() -> str:
    return r'''import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def process_jsonl(input_path: str) -> dict[str, Any]:
    counts_by_level: Counter[str] = Counter()
    total_valid = 0
    malformed_count = 0
    with Path(input_path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            if (
                not isinstance(event, dict)
                or not isinstance(event.get("level"), str)
                or not isinstance(event.get("message"), str)
            ):
                malformed_count += 1
                continue
            total_valid += 1
            counts_by_level[event["level"]] += 1
    return {
        "counts_by_level": dict(sorted(counts_by_level.items())),
        "malformed_count": malformed_count,
        "total_valid": total_valid,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a JSON Lines application log.")
    parser.add_argument("input_jsonl")
    parser.add_argument("output_json")
    args = parser.parse_args(argv)
    report = process_jsonl(args.input_jsonl)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _behavioral_test() -> str:
    return r'''import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cli


def test_behavioral_jsonl_log_report_counts_valid_and_malformed_records(tmp_path):
    input_jsonl = tmp_path / "application.jsonl"
    output_json = tmp_path / "report.json"
    input_jsonl.write_text(
        '{"level":"INFO","message":"started"}\n'
        'not-json\n'
        '{"level":"ERROR","message":"failed"}\n'
        '{"level":"INFO","message":"done"}\n',
        encoding="utf-8",
    )
    assert cli.main([str(input_jsonl), str(output_json)]) == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report == {
        "counts_by_level": {"ERROR": 1, "INFO": 2},
        "malformed_count": 1,
        "total_valid": 3,
    }
'''
