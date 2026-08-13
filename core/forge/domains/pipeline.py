from typing import List

from core.forge.contracts import ArtifactTargetType, FeasiblePlan, PlanInterface, PlanTest
from core.forge.domains.base import BaseDomainAdapter


class PipelineDomainAdapter(BaseDomainAdapter):
    name = "pipeline"

    def matches(self, plan: FeasiblePlan) -> bool:
        if plan.build_spec.target_artifact_type == ArtifactTargetType.PIPELINE:
            return True
        paths = {item.path.replace("\\", "/").lower() for item in plan.file_tree_plan}
        return "src/pipeline.py" in paths or "src/quarantine.py" in paths

    def render_file(self, plan: FeasiblePlan, path: str, interfaces: List[PlanInterface]) -> str:
        normalized = path.replace("\\", "/").lower()
        if normalized.endswith("src/pipeline.py"):
            return self._template_pipeline_entrypoint(plan)
        if normalized.endswith("src/watcher.py"):
            return self._template_pipeline_watcher()
        if normalized.endswith("src/validator.py"):
            return self._template_pipeline_validator()
        if normalized.endswith("src/quarantine.py"):
            return self._template_pipeline_quarantine()
        if normalized.startswith("tests/"):
            return self._template_pipeline_plan_test_module(plan)
        return self._template_generic_module(path, interfaces)

    def render_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        name = plan_test.test_name.lower()
        objective = plan_test.objective.lower()
        if "suite_executes" in name or any(
            token in objective
            for token in ("pipeline", "quarantine", "schema", "watch", "health endpoint", "health")
        ):
            return self._template_pipeline_suite_executes_test(plan)
        return self._template_pipeline_requirement_test(plan, plan_test)

    def _template_pipeline_entrypoint(self, plan: FeasiblePlan) -> str:
        quality = plan.quality_contract
        entrypoint_name = self._pipeline_workflow_name(plan)
        lines = [
            "import csv",
            "import json",
            "import os",
            "import sqlite3",
            "import time",
            "from pathlib import Path",
            "from typing import Any",
            "",
            "from quarantine import quarantine_row",
            "from validator import DEFAULT_SCHEMA, validate_row",
            "from watcher import discover_csv_files",
            "",
            "DB_PATH = os.environ.get('FORGE_PIPELINE_DB', 'pipeline.db')",
            "WATCH_DIR = os.environ.get('FORGE_WATCH_DIR', 'watch')",
            "QUARANTINE_DIR = os.environ.get('FORGE_QUARANTINE_DIR', 'quarantine')",
            f"QUALITY_LEVEL = {quality.overall_level}",
            f"QUALITY_AUDIT_TRAIL = {str(bool(quality.audit_trail))}",
            f"QUALITY_HEALTH_ENDPOINT = {str(bool(quality.health_endpoint))}",
            "",
            "",
            "def init_db(db_path: str = DB_PATH) -> None:",
            "    with sqlite3.connect(db_path) as conn:",
            "        conn.execute(",
            "            'CREATE TABLE IF NOT EXISTS records ('",
            "            'id INTEGER PRIMARY KEY AUTOINCREMENT, '",
            "            'source_file TEXT NOT NULL, '",
            "            'row_index INTEGER NOT NULL, '",
            "            'payload_json TEXT NOT NULL, '",
            "            'created_at REAL NOT NULL)'",
            "        )",
            "        conn.execute(",
            "            'CREATE TABLE IF NOT EXISTS audit_events ('",
            "            'id INTEGER PRIMARY KEY AUTOINCREMENT, '",
            "            'created_at REAL NOT NULL, '",
            "            'source_file TEXT NOT NULL, '",
            "            'row_index INTEGER NOT NULL, '",
            "            'status TEXT NOT NULL, '",
            "            'message TEXT NOT NULL)'",
            "        )",
            "        conn.execute(",
            "            'CREATE TABLE IF NOT EXISTS schema_meta ('",
            "            'name TEXT PRIMARY KEY, '",
            "            'version INTEGER NOT NULL)'",
            "        )",
            "        conn.execute(",
            "            'INSERT OR REPLACE INTO schema_meta(name, version) VALUES (?, ?)',",
            "            ('pipeline', 1),",
            "        )",
            "        conn.commit()",
            "",
            "",
            "def _insert_record(",
            "    conn: sqlite3.Connection,",
            "    source_file: str,",
            "    row_index: int,",
            "    row: dict[str, str],",
            ") -> None:",
            "    conn.execute(",
            "        'INSERT INTO records(source_file, row_index, payload_json, created_at) VALUES (?, ?, ?, ?)',",
            "        (source_file, row_index, json.dumps(row, sort_keys=True), time.time()),",
            "    )",
            "",
            "",
            "def _insert_audit(",
            "    conn: sqlite3.Connection,",
            "    source_file: str,",
            "    row_index: int,",
            "    status: str,",
            "    message: str,",
            ") -> None:",
            "    conn.execute(",
            "        'INSERT INTO audit_events(created_at, source_file, row_index, status, message) VALUES (?, ?, ?, ?, ?)',",
            "        (time.time(), source_file, row_index, status, message),",
            "    )",
            "",
            "",
            "def process_row(",
            "    conn: sqlite3.Connection,",
            "    row: dict[str, str],",
            "    row_index: int,",
            "    source_file: str,",
            "    schema: dict[str, str] | None,",
            "    quarantine_dir: str,",
            ") -> dict[str, object]:",
            "    valid, error = validate_row(row, schema=schema)",
            "    if not valid:",
            "        quarantine_path = quarantine_row(",
            "            row,",
            "            error=error or 'schema_validation_failed',",
            "            source_file=source_file,",
            "            row_index=row_index,",
            "            quarantine_dir=quarantine_dir,",
            "        )",
            "        _insert_audit(conn, source_file, row_index, 'quarantined', str(error or 'schema_validation_failed'))",
            "        return {'accepted': False, 'quarantine_path': str(quarantine_path)}",
            "    _insert_record(conn, source_file, row_index, row)",
            "    _insert_audit(conn, source_file, row_index, 'accepted', 'ok')",
            "    return {'accepted': True, 'quarantine_path': ''}",
            "",
            "",
            "def process_file(",
            "    csv_path: str,",
            "    db_path: str = DB_PATH,",
            "    schema: dict[str, str] | None = None,",
            "    quarantine_dir: str = QUARANTINE_DIR,",
            ") -> dict[str, int]:",
            "    init_db(db_path)",
            "    accepted = 0",
            "    quarantined = 0",
            "    with sqlite3.connect(db_path) as conn:",
            "        with open(csv_path, 'r', encoding='utf-8', newline='') as handle:",
            "            reader = csv.DictReader(handle)",
            "            for idx, row in enumerate(reader, start=1):",
            "                normalized = {str(k): ('' if v is None else str(v).strip()) for k, v in row.items() if k is not None}",
            "                outcome = process_row(",
            "                    conn,",
            "                    normalized,",
            "                    row_index=idx,",
            "                    source_file=Path(csv_path).name,",
            "                    schema=schema or DEFAULT_SCHEMA,",
            "                    quarantine_dir=quarantine_dir,",
            "                )",
            "                if bool(outcome.get('accepted')):",
            "                    accepted += 1",
            "                else:",
            "                    quarantined += 1",
            "        conn.commit()",
            "    return {'accepted': accepted, 'quarantined': quarantined}",
            "",
            "",
            "def get_pipeline_stats(db_path: str = DB_PATH) -> dict[str, int]:",
            "    init_db(db_path)",
            "    with sqlite3.connect(db_path) as conn:",
            "        accepted = conn.execute(",
            "            \"SELECT COUNT(1) FROM audit_events WHERE status = 'accepted'\"",
            "        ).fetchone()",
            "        quarantined = conn.execute(",
            "            \"SELECT COUNT(1) FROM audit_events WHERE status = 'quarantined'\"",
            "        ).fetchone()",
            "        records = conn.execute('SELECT COUNT(1) FROM records').fetchone()",
            "    return {",
            "        'accepted_count': int(accepted[0]) if accepted else 0,",
            "        'quarantined_count': int(quarantined[0]) if quarantined else 0,",
            "        'record_count': int(records[0]) if records else 0,",
            "    }",
            "",
            "",
            f"def {entrypoint_name}(",
            "    watch_dir: str = WATCH_DIR,",
            "    db_path: str = DB_PATH,",
            "    quarantine_dir: str = QUARANTINE_DIR,",
            "    schema: dict[str, str] | None = None,",
            "    poll_once: bool = True,",
            ") -> int:",
            "    processed_any = False",
            "    seen: set[str] = set()",
            "    while True:",
            "        files = discover_csv_files(watch_dir, seen=seen)",
            "        for csv_file in files:",
            "            process_file(",
            "                str(csv_file),",
            "                db_path=db_path,",
            "                schema=schema or DEFAULT_SCHEMA,",
            "                quarantine_dir=quarantine_dir,",
            "            )",
            "            seen.add(str(csv_file))",
            "            processed_any = True",
            "        if poll_once:",
            "            break",
            "        time.sleep(1.0)",
            "    if not processed_any:",
            "        init_db(db_path)",
            "    return 0",
            "",
            "",
            "try:",
            "    from fastapi import FastAPI",
            "",
            "    app = FastAPI(title='Forge Pipeline')",
            "",
            "    @app.get('/health')",
            "    def health() -> dict[str, Any]:",
            "        return get_pipeline_stats(DB_PATH)",
            "except Exception:",
            "    app = None",
            "",
        ]
        cli_entrypoint = next(
            (
                interface.name
                for interface in plan.interfaces
                if interface.interface_type == "cli_entrypoint" and interface.name.isidentifier()
            ),
            "",
        )
        if cli_entrypoint:
            lines.extend(
                [
                    "",
                    f"def {cli_entrypoint}(argv: list[str] | None = None) -> int:",
                    "    import argparse",
                    "",
                    "    parser = argparse.ArgumentParser()",
                    "    parser.add_argument('input_path')",
                    "    parser.add_argument('quarantine_path')",
                    "    parser.add_argument('summary_csv_path')",
                    "    args = parser.parse_args(argv)",
                    f"    return {entrypoint_name}(",
                    "        watch_dir=str(Path(args.input_path).parent),",
                    "        db_path=str(Path(args.summary_csv_path).with_suffix('.db')),",
                    "        quarantine_dir=str(Path(args.quarantine_path).parent),",
                    "        poll_once=True,",
                    "    )",
                    "",
                ]
            )
        return "\n".join(lines)

    def _pipeline_workflow_name(self, plan: FeasiblePlan) -> str:
        for interface in plan.interfaces:
            if interface.interface_type == "entrypoint" and interface.name.isidentifier():
                return interface.name
        for interface in plan.interfaces:
            if interface.interface_type == "function" and interface.name == "run":
                return interface.name
        return "run"

    def _template_pipeline_watcher(self) -> str:
        return (
            "from pathlib import Path\n"
            "\n"
            "\n"
            "def discover_csv_files(watch_dir: str, seen: set[str] | None = None) -> list[Path]:\n"
            "    base = Path(watch_dir)\n"
            "    if not base.exists() or not base.is_dir():\n"
            "        return []\n"
            "    known = seen or set()\n"
            "    files: list[Path] = []\n"
            "    for candidate in sorted(base.glob('*.csv')):\n"
            "        if str(candidate) in known:\n"
            "            continue\n"
            "        files.append(candidate)\n"
            "    return files\n"
        )
    def _template_pipeline_validator(self) -> str:
        return (
            "from datetime import datetime\n"
            "\n"
            "DEFAULT_SCHEMA: dict[str, str] = {\n"
            "    'invoice_id': 'str',\n"
            "    'due_date': 'date',\n"
            "    'amount': 'float',\n"
            "    'customer_name': 'str',\n"
            "}\n"
            "\n"
            "\n"
            "def _is_valid_date(value: str) -> bool:\n"
            "    candidate = (value or '').strip()\n"
            "    if not candidate:\n"
            "        return False\n"
            "    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'):\n"
            "        try:\n"
            "            datetime.strptime(candidate, fmt)\n"
            "            return True\n"
            "        except ValueError:\n"
            "            continue\n"
            "    return False\n"
            "\n"
            "\n"
            "def validate_row(row: dict[str, str], schema: dict[str, str] | None = None) -> tuple[bool, str]:\n"
            "    active_schema = schema or DEFAULT_SCHEMA\n"
            "    for field, expected_type in active_schema.items():\n"
            "        if field not in row:\n"
            "            return False, f'missing_field:{field}'\n"
            "        value = (row.get(field) or '').strip()\n"
            "        if expected_type == 'str':\n"
            "            if not value:\n"
            "                return False, f'empty_field:{field}'\n"
            "        elif expected_type == 'float':\n"
            "            try:\n"
            "                float(value)\n"
            "            except ValueError:\n"
            "                return False, f'invalid_float:{field}'\n"
            "        elif expected_type == 'date':\n"
            "            if not _is_valid_date(value):\n"
            "                return False, f'invalid_date:{field}'\n"
            "    return True, ''\n"
        )

    def _template_pipeline_quarantine(self) -> str:
        return (
            "import json\n"
            "from pathlib import Path\n"
            "import time\n"
            "\n"
            "\n"
            "def quarantine_row(\n"
            "    row: dict[str, str],\n"
            "    error: str,\n"
            "    source_file: str,\n"
            "    row_index: int,\n"
            "    quarantine_dir: str,\n"
            ") -> Path:\n"
            "    base = Path(quarantine_dir)\n"
            "    base.mkdir(parents=True, exist_ok=True)\n"
            "    payload = {\n"
            "        'created_at': time.time(),\n"
            "        'source_file': source_file,\n"
            "        'row_index': row_index,\n"
            "        'error': error,\n"
            "        'row': row,\n"
            "    }\n"
            "    target = base / f\"{Path(source_file).stem}_{row_index}.json\"\n"
            "    target.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')\n"
            "    log_file = base / 'quarantine.log.jsonl'\n"
            "    with log_file.open('a', encoding='utf-8') as handle:\n"
            "        handle.write(json.dumps(payload, sort_keys=True) + '\\n')\n"
            "    return target\n"
        )

    def _template_pipeline_plan_test_module(self, plan: FeasiblePlan) -> str:
        entrypoint_name = self._pipeline_workflow_name(plan)
        return (
            "import sqlite3\n"
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "\n"
            "import pipeline\n"
            "\n"
            "\n"
            "def test_pipeline_plan_smoke(tmp_path):\n"
            "    watch_dir = tmp_path / 'watch'\n"
            "    watch_dir.mkdir()\n"
            "    quarantine_dir = tmp_path / 'quarantine'\n"
            "    db_path = tmp_path / 'pipeline.db'\n"
            "    (watch_dir / 'rows.csv').write_text(\n"
            "        'invoice_id,due_date,amount,customer_name\\nINV-1,2026-01-15,10,Acme\\n',\n"
            "        encoding='utf-8',\n"
            "    )\n"
            f"    rc = pipeline.{entrypoint_name}(\n"
            "        watch_dir=str(watch_dir),\n"
            "        db_path=str(db_path),\n"
            "        quarantine_dir=str(quarantine_dir),\n"
            "        poll_once=True,\n"
            "    )\n"
            "    assert rc == 0\n"
            "    stats = pipeline.get_pipeline_stats(str(db_path))\n"
            "    assert stats['accepted_count'] >= 1\n"
            "    with sqlite3.connect(str(db_path)) as conn:\n"
            "        audit_rows = conn.execute('SELECT COUNT(1) FROM audit_events').fetchone()\n"
            "    assert int(audit_rows[0]) >= 1\n"
        )

    def _template_pipeline_suite_executes_test(self, plan: FeasiblePlan) -> str:
        entrypoint_name = self._pipeline_workflow_name(plan)
        return (
            "import sqlite3\n"
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "\n"
            "import pipeline\n"
            "\n"
            "\n"
            "def test_integration_pipeline_processes_valid_and_invalid_rows(tmp_path):\n"
            "    watch_dir = tmp_path / 'watch'\n"
            "    watch_dir.mkdir()\n"
            "    quarantine_dir = tmp_path / 'quarantine'\n"
            "    db_path = tmp_path / 'pipeline.db'\n"
            "    (watch_dir / 'batch.csv').write_text(\n"
            "        'invoice_id,due_date,amount,customer_name\\n'\n"
            "        'INV-1,2026-01-15,10,Acme\\n'\n"
            "        'INV-2,not-a-date,12,Beta\\n',\n"
            "        encoding='utf-8',\n"
            "    )\n"
            f"    rc = pipeline.{entrypoint_name}(\n"
            "        watch_dir=str(watch_dir),\n"
            "        db_path=str(db_path),\n"
            "        quarantine_dir=str(quarantine_dir),\n"
            "        poll_once=True,\n"
            "    )\n"
            "    assert rc == 0\n"
            "    stats = pipeline.get_pipeline_stats(str(db_path))\n"
            "    assert stats['accepted_count'] >= 1\n"
            "    assert stats['quarantined_count'] >= 1\n"
            "    assert (quarantine_dir / 'quarantine.log.jsonl').exists()\n"
            "    with sqlite3.connect(str(db_path)) as conn:\n"
            "        audit_count = conn.execute('SELECT COUNT(1) FROM audit_events').fetchone()\n"
            "    assert int(audit_count[0]) >= 2\n"
            "\n"
            "\n"
            "def test_health_endpoint_stats_function(tmp_path):\n"
            "    db_path = tmp_path / 'pipeline.db'\n"
            "    pipeline.init_db(str(db_path))\n"
            "    stats = pipeline.get_pipeline_stats(str(db_path))\n"
            "    assert {'accepted_count', 'quarantined_count', 'record_count'} <= set(stats.keys())\n"
        )

    def _template_pipeline_requirement_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        entrypoint_name = self._pipeline_workflow_name(plan)
        objective = plan_test.objective.lower()
        if any(token in objective for token in ("quarantine", "invalid", "schema")):
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "import pipeline\n"
                "\n"
                "\n"
                "def test_pipeline_quarantines_invalid_rows(tmp_path):\n"
                "    watch_dir = tmp_path / 'watch'\n"
                "    watch_dir.mkdir()\n"
                "    quarantine_dir = tmp_path / 'quarantine'\n"
                "    db_path = tmp_path / 'pipeline.db'\n"
                "    (watch_dir / 'invalid.csv').write_text(\n"
                "        'invoice_id,due_date,amount,customer_name\\nINV-1,bad-date,10,Acme\\n',\n"
                "        encoding='utf-8',\n"
                "    )\n"
                f"    rc = pipeline.{entrypoint_name}(\n"
                "        watch_dir=str(watch_dir),\n"
                "        db_path=str(db_path),\n"
                "        quarantine_dir=str(quarantine_dir),\n"
                "        poll_once=True,\n"
                "    )\n"
                "    assert rc == 0\n"
                "    stats = pipeline.get_pipeline_stats(str(db_path))\n"
                "    assert stats['quarantined_count'] >= 1\n"
                "    assert (quarantine_dir / 'quarantine.log.jsonl').exists()\n"
            )
        if "health" in objective:
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "import pipeline\n"
                "\n"
                "\n"
                "def test_pipeline_health_stats_are_exposed(tmp_path):\n"
                "    db_path = tmp_path / 'pipeline.db'\n"
                "    pipeline.init_db(str(db_path))\n"
                "    stats = pipeline.get_pipeline_stats(str(db_path))\n"
                "    assert stats['accepted_count'] >= 0\n"
                "    assert stats['quarantined_count'] >= 0\n"
                "    assert stats['record_count'] >= 0\n"
            )
        return (
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "\n"
            "import pipeline\n"
            "\n"
            "\n"
            "def test_pipeline_requirement_smoke(tmp_path):\n"
            "    watch_dir = tmp_path / 'watch'\n"
            "    watch_dir.mkdir()\n"
            "    quarantine_dir = tmp_path / 'quarantine'\n"
            "    db_path = tmp_path / 'pipeline.db'\n"
            "    (watch_dir / 'batch.csv').write_text(\n"
            "        'invoice_id,due_date,amount,customer_name\\nINV-1,2026-01-15,10,Acme\\n',\n"
            "        encoding='utf-8',\n"
            "    )\n"
            f"    rc = pipeline.{entrypoint_name}(\n"
            "        watch_dir=str(watch_dir),\n"
            "        db_path=str(db_path),\n"
            "        quarantine_dir=str(quarantine_dir),\n"
            "        poll_once=True,\n"
            "    )\n"
            "    assert rc == 0\n"
            "    stats = pipeline.get_pipeline_stats(str(db_path))\n"
            "    assert stats['accepted_count'] >= 1\n"
        )
