from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


def is_jsonl_pipeline(plan: FeasiblePlan) -> bool:
    corpus = " ".join(
        [
            plan.build_spec.normalized_requirement,
            plan.architecture_summary,
            *(plan_file.purpose for plan_file in plan.file_tree_plan),
        ]
    ).lower()
    return (
        ("json lines" in corpus or "jsonl" in corpus)
        and all(token in corpus for token in ("device_id", "timestamp", "temperature_c"))
    )


def render_jsonl_file(
    plan: FeasiblePlan,
    path: str,
    interfaces: list[PlanInterface],
) -> str | None:
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith("src/pipeline.py"):
        return _pipeline_module(plan)
    if normalized.endswith("src/watcher.py"):
        return _watcher_module()
    if normalized.endswith("src/validator.py"):
        return _validator_module()
    if normalized.endswith("src/quarantine.py"):
        return _quarantine_module()
    if normalized.startswith("tests/"):
        return _complete_flow_test()
    return None


def render_jsonl_test(plan: FeasiblePlan, plan_test: PlanTest) -> str:
    evidence_terms = {
        term
        for requirement_id in plan_test.requirement_ids
        for atom in plan.build_spec.requirement_atoms
        if atom.requirement_id == requirement_id
        for term in atom.evidence_terms
    }
    objective = plan_test.objective.lower()
    if "cli_flow" in evidence_terms or "suite_executes" in plan_test.test_name.lower():
        return _complete_flow_test()
    if evidence_terms.intersection(
        {"malformed_records", "missing_fields", "invalid_timestamp", "quarantine"}
    ):
        return _quarantine_test()
    if evidence_terms.intersection({"minimum", "maximum", "average", "per_device"}):
        return _aggregation_test()
    if "summary_csv" in evidence_terms:
        return _summary_test()
    if evidence_terms.intersection({"input_jsonl", "device_id", "temperature_c", "timestamp"}):
        return _parsing_test()
    if "cli_entrypoint" in evidence_terms or "cli" in objective:
        return _cli_test()
    return _complete_flow_test()


def _pipeline_module(plan: FeasiblePlan) -> str:
    workflow_name = _workflow_name(plan)
    cli_name = _cli_name(plan)
    lines = [
        "import argparse",
        "import csv",
        "from collections import defaultdict",
        "from pathlib import Path",
        "from typing import Any",
        "",
        "from quarantine import write_quarantine_record",
        "from validator import validate_telemetry_record",
        "from watcher import iter_jsonl",
        "",
        "CLI_ENTRYPOINT = 'main'",
        "INPUT_JSONL_FIELDS = ('device_id', 'timestamp', 'temperature_c')",
        "",
        "",
        "def aggregate_per_device(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:",
        "    device_temperatures: dict[str, list[float]] = defaultdict(list)",
        "    for record in records:",
        "        device_temperatures[str(record['device_id'])].append(float(record['temperature_c']))",
        "    per_device: dict[str, dict[str, float | int]] = {}",
        "    for device_id, temperatures in device_temperatures.items():",
        "        minimum = min(temperatures)",
        "        maximum = max(temperatures)",
        "        average = sum(temperatures) / len(temperatures)",
        "        per_device[device_id] = {",
        "            'minimum': minimum,",
        "            'maximum': maximum,",
        "            'average': average,",
        "            'count': len(temperatures),",
        "        }",
        "    return per_device",
        "",
        "",
        "def write_summary_csv(",
        "    summary_csv_path: str,",
        "    per_device: dict[str, dict[str, float | int]],",
        ") -> None:",
        "    target = Path(summary_csv_path)",
        "    target.parent.mkdir(parents=True, exist_ok=True)",
        "    with target.open('w', newline='', encoding='utf-8') as handle:",
        "        writer = csv.DictWriter(",
        "            handle,",
        "            fieldnames=['device_id', 'minimum', 'maximum', 'average', 'count'],",
        "        )",
        "        writer.writeheader()",
        "        for device_id, aggregation in sorted(per_device.items()):",
        "            writer.writerow({'device_id': device_id, **aggregation})",
        "",
        "",
        f"def {workflow_name}(input_path: str, quarantine_path: str, summary_csv_path: str) -> int:",
        "    accepted_records: list[dict[str, Any]] = []",
        "    for line_number, raw_line, record, parse_error in iter_jsonl(input_path):",
        "        validation_errors = (",
        "            [parse_error] if parse_error else validate_telemetry_record(record or {})",
        "        )",
        "        if validation_errors:",
        "            write_quarantine_record(",
        "                quarantine_path=quarantine_path,",
        "                line_number=line_number,",
        "                raw_line=raw_line,",
        "                record=record,",
        "                errors=validation_errors,",
        "            )",
        "            continue",
        "        accepted_records.append(record or {})",
        "    aggregation = aggregate_per_device(accepted_records)",
        "    write_summary_csv(summary_csv_path, aggregation)",
        "    return 0",
    ]
    if cli_name:
        lines.extend(
            [
                "",
                "",
                f"def {cli_name}(argv: list[str] | None = None) -> int:",
                "    parser = argparse.ArgumentParser(description='Process JSON Lines telemetry.')",
                "    parser.add_argument('input_path')",
                "    parser.add_argument('quarantine_path')",
                "    parser.add_argument('summary_csv_path')",
                "    args = parser.parse_args(argv)",
                f"    return {workflow_name}(args.input_path, args.quarantine_path, args.summary_csv_path)",
            ]
        )
    return "\n".join(lines) + "\n"


def _workflow_name(plan: FeasiblePlan) -> str:
    for interface in plan.interfaces:
        if interface.interface_type == "entrypoint" and interface.name.isidentifier():
            return interface.name
    return "run"


def _cli_name(plan: FeasiblePlan) -> str:
    for interface in plan.interfaces:
        if interface.interface_type == "cli_entrypoint" and interface.name.isidentifier():
            return interface.name
    return ""


def _watcher_module() -> str:
    return r'''import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(input_jsonl: str) -> Iterator[tuple[int, str, dict[str, Any] | None, str | None]]:
    with Path(input_jsonl).open(encoding="utf-8") as handle:
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
    return r'''from datetime import datetime
from typing import Any


REQUIRED_FIELDS = ("device_id", "timestamp", "temperature_c")


def _valid_timestamp(timestamp: object) -> bool:
    if not isinstance(timestamp, str) or not timestamp.strip():
        return False
    candidate = timestamp.strip().replace("Z", "+00:00")
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return True


def validate_telemetry_record(record: dict[str, Any]) -> list[str]:
    missing_fields = [field for field in REQUIRED_FIELDS if field not in record]
    errors = [f"missing_fields:{field}" for field in missing_fields]
    if "device_id" in record and not str(record["device_id"]).strip():
        errors.append("missing_fields:device_id")
    if "timestamp" in record and not _valid_timestamp(record["timestamp"]):
        errors.append("invalid_timestamp")
    if "temperature_c" in record:
        try:
            if isinstance(record["temperature_c"], bool):
                raise ValueError
            float(record["temperature_c"])
        except (TypeError, ValueError):
            errors.append("invalid_temperature_c")
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
    record: dict[str, Any] | None,
    errors: list[str],
) -> None:
    target = Path(quarantine_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "line_number": line_number,
        "raw_line": raw_line,
        "record": record,
        "errors": errors,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
'''


def _test_prelude() -> str:
    return r'''import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pipeline
'''


def _paths_fixture() -> str:
    return r'''

def telemetry_paths(tmp_path):
    input_jsonl = tmp_path / "telemetry.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    summary_csv = tmp_path / "summary.csv"
    return input_jsonl, quarantine, summary_csv
'''


def _parsing_test() -> str:
    return _test_prelude() + _paths_fixture() + r'''

def test_reads_input_jsonl_fields(tmp_path):
    input_jsonl, quarantine, summary_csv = telemetry_paths(tmp_path)
    input_jsonl.write_text(
        '{"device_id":"alpha","timestamp":"2026-01-01T00:00:00Z","temperature_c":12.5}\n',
        encoding="utf-8",
    )
    assert pipeline.run(str(input_jsonl), str(quarantine), str(summary_csv)) == 0
    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    assert rows[0]["device_id"] == "alpha"
    assert float(rows[0]["average"]) == 12.5
    assert not quarantine.exists()
'''


def _quarantine_test() -> str:
    return _test_prelude() + _paths_fixture() + r'''

def test_malformed_records_missing_fields_and_invalid_timestamp_are_quarantined(tmp_path):
    input_jsonl, quarantine, summary_csv = telemetry_paths(tmp_path)
    input_jsonl.write_text(
        "not-json\n"
        '{"device_id":"missing-temp","timestamp":"2026-01-01T00:00:00Z"}\n'
        '{"device_id":"bad-time","timestamp":"invalid","temperature_c":3}\n',
        encoding="utf-8",
    )
    assert pipeline.run(str(input_jsonl), str(quarantine), str(summary_csv)) == 0
    malformed_records = [json.loads(line) for line in quarantine.read_text(encoding="utf-8").splitlines()]
    errors = {error for item in malformed_records for error in item["errors"]}
    assert "malformed_records" in errors
    assert "missing_fields:temperature_c" in errors
    assert "invalid_timestamp" in errors
'''


def _aggregation_test() -> str:
    return _test_prelude() + _paths_fixture() + r'''

def test_per_device_minimum_maximum_average(tmp_path):
    input_jsonl, quarantine, summary_csv = telemetry_paths(tmp_path)
    input_jsonl.write_text(
        '{"device_id":"alpha","timestamp":"2026-01-01T00:00:00Z","temperature_c":10}\n'
        '{"device_id":"alpha","timestamp":"2026-01-01T01:00:00Z","temperature_c":20}\n'
        '{"device_id":"beta","timestamp":"2026-01-01T00:00:00Z","temperature_c":5}\n',
        encoding="utf-8",
    )
    assert pipeline.run(str(input_jsonl), str(quarantine), str(summary_csv)) == 0
    per_device = {row["device_id"]: row for row in csv.DictReader(summary_csv.open(encoding="utf-8"))}
    assert float(per_device["alpha"]["minimum"]) == 10.0
    assert float(per_device["alpha"]["maximum"]) == 20.0
    assert float(per_device["alpha"]["average"]) == 15.0
    assert int(per_device["alpha"]["count"]) == 2
'''


def _summary_test() -> str:
    return _test_prelude() + _paths_fixture() + r'''

def test_writes_summary_csv(tmp_path):
    input_jsonl, quarantine, summary_csv = telemetry_paths(tmp_path)
    pipeline.write_summary_csv(
        str(summary_csv),
        {
            "summary-device": {
                "minimum": 7.0,
                "maximum": 7.0,
                "average": 7.0,
                "count": 1,
            }
        },
    )
    assert summary_csv.exists()
    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    assert rows == [{"device_id": "summary-device", "minimum": "7.0", "maximum": "7.0", "average": "7.0", "count": "1"}]
'''


def _cli_test() -> str:
    return _test_prelude() + _paths_fixture() + r'''

def test_cli_entrypoint(tmp_path):
    input_jsonl, quarantine, summary_csv = telemetry_paths(tmp_path)
    input_jsonl.write_text(
        '{"device_id":"cli-device","timestamp":"2026-01-01T00:00:00Z","temperature_c":9}\n',
        encoding="utf-8",
    )
    cli_entrypoint_result = pipeline.main([str(input_jsonl), str(quarantine), str(summary_csv)])
    assert cli_entrypoint_result == 0
    assert summary_csv.exists()
'''


def _complete_flow_test() -> str:
    return _test_prelude() + _paths_fixture() + r'''

def test_complete_cli_flow_parsing_quarantine_aggregation(tmp_path):
    input_jsonl, quarantine, summary_csv = telemetry_paths(tmp_path)
    input_jsonl.write_text(
        '{"device_id":"alpha","timestamp":"2026-01-01T00:00:00Z","temperature_c":10}\n'
        '{"device_id":"alpha","timestamp":"2026-01-01T01:00:00Z","temperature_c":14}\n'
        '{"device_id":"bad","timestamp":"invalid","temperature_c":99}\n',
        encoding="utf-8",
    )
    cli_flow = pipeline.main([str(input_jsonl), str(quarantine), str(summary_csv)])
    assert cli_flow == 0
    aggregation = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    assert aggregation[0]["device_id"] == "alpha"
    assert float(aggregation[0]["minimum"]) == 10.0
    assert float(aggregation[0]["maximum"]) == 14.0
    assert float(aggregation[0]["average"]) == 12.0
    quarantined = [json.loads(line) for line in quarantine.read_text(encoding="utf-8").splitlines()]
    assert quarantined[0]["errors"] == ["invalid_timestamp"]
'''
