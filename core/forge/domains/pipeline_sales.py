from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


def is_sales_jsonl_pipeline(plan: FeasiblePlan) -> bool:
    corpus = " ".join(
        [
            plan.build_spec.normalized_requirement,
            plan.architecture_summary,
            *(plan_file.purpose for plan_file in plan.file_tree_plan),
        ]
    ).lower()
    return all(
        token in corpus
        for token in ("json lines", "customer_id", "amount", "transaction_count", "summary json")
    )


def render_sales_file(
    plan: FeasiblePlan,
    path: str,
    interfaces: list[PlanInterface],
) -> str | None:
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith("src/pipeline.py"):
        return _pipeline_module()
    if normalized.endswith("src/watcher.py"):
        return _watcher_module()
    if normalized.endswith("src/validator.py"):
        return _validator_module()
    if normalized.endswith("src/quarantine.py"):
        return _quarantine_module()
    if normalized.startswith("tests/"):
        return _integration_test()
    return None


def render_sales_test(plan: FeasiblePlan, plan_test: PlanTest) -> str:
    return _integration_test()


def _pipeline_module() -> str:
    return r'''import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from quarantine import write_quarantine_record
from validator import validate_sales_event
from watcher import iter_jsonl


def run(input_path: str, quarantine_path: str, summary_json_path: str) -> int:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for line_number, raw_line, event, parse_error in iter_jsonl(input_path):
        errors = [parse_error] if parse_error else validate_sales_event(event or {})
        if errors:
            write_quarantine_record(quarantine_path, line_number, raw_line, event, errors)
            continue
        customer_id = str((event or {})["customer_id"])
        totals[customer_id] += float((event or {})["amount"])
        counts[customer_id] += 1

    summary: dict[str, dict[str, Any]] = {
        customer_id: {
            "total_amount": totals[customer_id],
            "transaction_count": counts[customer_id],
        }
        for customer_id in sorted(counts)
    }
    target = Path(summary_json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    return 0
'''


def _watcher_module() -> str:
    return r'''import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(input_path: str) -> Iterator[tuple[int, str, dict[str, Any] | None, str | None]]:
    with Path(input_path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.rstrip("\n")
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError:
                yield line_number, raw_line, None, "malformed_records"
                continue
            if not isinstance(value, dict):
                yield line_number, raw_line, None, "malformed_records"
                continue
            yield line_number, raw_line, value, None
'''


def _validator_module() -> str:
    return r'''from typing import Any


def validate_sales_event(event: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    customer_id = event.get("customer_id")
    if not isinstance(customer_id, str) or not customer_id.strip():
        errors.append("missing_fields:customer_id")
    amount = event.get("amount")
    try:
        if isinstance(amount, bool):
            raise ValueError
        float(amount)
    except (TypeError, ValueError):
        errors.append("invalid_amount")
    return errors
'''


def _quarantine_module() -> str:
    return r'''import json
from pathlib import Path
from typing import Any


def write_quarantine_record(
    quarantine_path: str,
    line_number: int,
    raw_line: str,
    event: dict[str, Any] | None,
    errors: list[str],
) -> None:
    target = Path(quarantine_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "line_number": line_number,
        "raw_line": raw_line,
        "event": event,
        "errors": errors,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
'''


def _integration_test() -> str:
    return r'''import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pipeline


def test_integration_end_to_end_sales_pipeline_aggregates_and_quarantines(tmp_path):
    input_path = tmp_path / "sales.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    summary_json_path = tmp_path / "summary.json"
    input_path.write_text(
        '{"customer_id":"A","amount":10.5}\n'
        '{"customer_id":"B","amount":3}\n'
        'malformed\n'
        '{"customer_id":"A","amount":2.5}\n',
        encoding="utf-8",
    )

    result = pipeline.run(str(input_path), str(quarantine_path), str(summary_json_path))
    assert result == 0
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["A"] == {"total_amount": 13.0, "transaction_count": 2}
    assert summary["B"] == {"total_amount": 3.0, "transaction_count": 1}
    quarantined = [
        json.loads(line)
        for line in quarantine_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(quarantined) == 1
    assert quarantined[0]["errors"] == ["malformed_records"]
'''
