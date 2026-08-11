from typing import List

from core.forge.contracts import FeasiblePlan


def render_audit(plan: FeasiblePlan) -> str:
    quality = plan.quality_contract
    lines: List[str] = [
        "import json",
        "import logging",
        "import sqlite3",
        "import time",
        "",
        "from storage import DB_PATH, init_db",
        "",
        "_logger = logging.getLogger('forge.service')",
        "if not _logger.handlers:",
        "    logging.basicConfig(level=logging.INFO)",
        "",
        "",
        "def record_event(",
        "    user_id: str | None,",
        "    status_code: int,",
        "    detail: str,",
        "    db_path: str = DB_PATH,",
        ") -> None:",
    ]
    if quality.audit_trail:
        lines.extend(
            [
                "    init_db(db_path)",
                "    with sqlite3.connect(db_path) as conn:",
                "        conn.execute(",
                "            'INSERT INTO events(created_at, user_id, status_code, detail) VALUES (?, ?, ?, ?)',",
                "            (time.time(), user_id, status_code, detail),",
                "        )",
                "        conn.commit()",
            ]
        )
    if quality.structured_logging:
        lines.extend(
            [
                "    _logger.info(",
                "        json.dumps(",
                "            {'user_id': user_id, 'status_code': status_code, 'detail': detail},",
                "            sort_keys=True,",
                "        )",
                "    )",
            ]
        )
    else:
        lines.append("    _logger.info('user=%s status=%s detail=%s', user_id, status_code, detail)")
    lines.extend(
        [
            "",
            "",
            "def get_recent_events(db_path: str = DB_PATH, limit: int = 50) -> list[dict[str, object]]:",
        ]
    )
    if quality.audit_trail:
        lines.extend(
            [
                "    init_db(db_path)",
                "    with sqlite3.connect(db_path) as conn:",
                "        rows = conn.execute(",
                "            'SELECT created_at, user_id, status_code, detail FROM events ORDER BY id DESC LIMIT ?',",
                "            (limit,),",
                "        ).fetchall()",
                "    return [",
                "        {'created_at': float(row[0]), 'user_id': row[1], 'status_code': int(row[2]), 'detail': str(row[3])}",
                "        for row in rows",
                "    ]",
            ]
        )
    else:
        lines.append("    return []")
    lines.append("")
    return "\n".join(lines)
