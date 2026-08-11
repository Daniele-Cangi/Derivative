from typing import List

from core.forge.contracts import FeasiblePlan


def render_observability(plan: FeasiblePlan) -> str:
    quality = plan.quality_contract
    lines: List[str] = [
        "import sqlite3",
        "",
        "from storage import DB_PATH, init_db",
        "",
        "",
        "def health_status(db_path: str = DB_PATH) -> dict[str, object]:",
        "    init_db(db_path)",
        "    with sqlite3.connect(db_path) as conn:",
        "        users = conn.execute('SELECT COUNT(1) FROM users').fetchone()",
    ]
    if quality.audit_trail:
        lines.append("        events = conn.execute('SELECT COUNT(1) FROM events').fetchone()")
    else:
        lines.append("        events = (0,)")
    lines.extend(
        [
            "    return {",
            "        'status': 'ok',",
            "        'user_count': int(users[0]) if users else 0,",
            "        'event_count': int(events[0]) if events else 0,",
            f"        'structured_logging': {quality.structured_logging!r},",
            "    }",
            "",
        ]
    )
    return "\n".join(lines)
