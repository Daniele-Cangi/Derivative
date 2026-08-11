from typing import List

from core.forge.contracts import FeasiblePlan


def render_storage(plan: FeasiblePlan) -> str:
    quality = plan.quality_contract
    plaintext = bool(quality.secrets_in_plaintext and quality.auth_level == "plaintext")
    persistent_rate = bool(
        quality.rate_limit_persistent or quality.rate_limit_scope == "distributed"
    )
    user_column = "api_key TEXT UNIQUE NOT NULL" if plaintext else "api_key_hash TEXT UNIQUE NOT NULL"
    lines: List[str] = [
        "import os",
        "import sqlite3",
        "",
        "DB_PATH = os.environ.get('FORGE_SERVICE_DB', 'service.db')",
        "",
        "",
        "def init_db(db_path: str = DB_PATH) -> None:",
        "    with sqlite3.connect(db_path) as conn:",
        "        conn.execute(",
        "            'CREATE TABLE IF NOT EXISTS users ('",
        "            'username TEXT PRIMARY KEY, '",
        f"            '{user_column})'",
        "        )",
    ]
    if persistent_rate:
        lines.extend(
            [
                "        conn.execute(",
                "            'CREATE TABLE IF NOT EXISTS rate_limit_hits ('",
                "            'user_id TEXT NOT NULL, '",
                "            'ts REAL NOT NULL)'",
                "        )",
            ]
        )
    if quality.audit_trail:
        lines.extend(
            [
                "        conn.execute(",
                "            'CREATE TABLE IF NOT EXISTS events ('",
                "            'id INTEGER PRIMARY KEY AUTOINCREMENT, '",
                "            'created_at REAL NOT NULL, '",
                "            'user_id TEXT, '",
                "            'status_code INTEGER NOT NULL, '",
                "            'detail TEXT NOT NULL)'",
                "        )",
            ]
        )
    if quality.schema_versioned:
        lines.extend(
            [
                "        conn.execute(",
                "            'CREATE TABLE IF NOT EXISTS schema_meta ('",
                "            'name TEXT PRIMARY KEY, '",
                "            'version INTEGER NOT NULL)'",
                "        )",
                "        conn.execute(",
                "            'INSERT OR REPLACE INTO schema_meta(name, version) VALUES (?, ?)',",
                "            ('service', 1),",
                "        )",
            ]
        )
    lines.extend(["        conn.commit()", ""])
    return "\n".join(lines)
