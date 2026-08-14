from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


def is_idempotent_event_service(plan: FeasiblePlan) -> bool:
    corpus = " ".join(
        [
            plan.build_spec.normalized_requirement,
            plan.architecture_summary,
            *(interface.signature for interface in plan.interfaces),
        ]
    ).lower()
    return all(token in corpus for token in ("create_event", "event_id", "idempotent", "sqlite"))


def render_event_service_file(
    plan: FeasiblePlan,
    path: str,
    interfaces: list[PlanInterface],
) -> str | None:
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith("src/service.py"):
        return _service_module()
    if normalized.endswith("src/storage.py"):
        return _storage_module()
    if normalized.endswith("src/auth.py"):
        return _auth_module()
    if normalized.startswith("tests/"):
        return _integration_test()
    return None


def render_event_service_test(plan: FeasiblePlan, plan_test: PlanTest) -> str:
    return _integration_test()


def _service_module() -> str:
    return r'''from typing import Any

from auth import authenticate, register_user as _register_user
from storage import insert_event_once


QUALITY_AUTH_MODE = "plaintext"


def register_user(user_id: str, api_key: str, db_path: str) -> None:
    _register_user(user_id, api_key, db_path)


def create_event(
    api_key: str,
    event_id: str,
    payload: dict[str, Any],
    db_path: str,
) -> tuple[int, dict[str, Any]]:
    user_id = authenticate(api_key, db_path)
    if user_id is None:
        return 401, {"error": "unauthorized"}
    created, stored_payload = insert_event_once(event_id, user_id, payload, db_path)
    return (201 if created else 200), {
        "event_id": event_id,
        "payload": stored_payload,
        "created": created,
    }
'''


def _storage_module() -> str:
    return r'''import json
import sqlite3
from typing import Any


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "user_id TEXT PRIMARY KEY, api_key TEXT NOT NULL UNIQUE)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "event_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, payload_json TEXT NOT NULL)"
        )


def insert_event_once(
    event_id: str,
    user_id: str,
    payload: dict[str, Any],
    db_path: str,
) -> tuple[bool, dict[str, Any]]:
    init_db(db_path)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO events(event_id, user_id, payload_json) VALUES (?, ?, ?)",
            (event_id, user_id, encoded),
        )
        row = connection.execute(
            "SELECT payload_json FROM events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError("event insert did not produce a stored row")
    return cursor.rowcount == 1, json.loads(str(row[0]))
'''


def _auth_module() -> str:
    return r'''import sqlite3

from storage import init_db


def register_user(user_id: str, api_key: str, db_path: str) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO users(user_id, api_key) VALUES (?, ?)",
            (user_id, api_key),
        )


def authenticate(api_key: str, db_path: str) -> str | None:
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT user_id FROM users WHERE api_key = ?",
            (api_key,),
        ).fetchone()
    return str(row[0]) if row is not None else None
'''


def _integration_test() -> str:
    return r'''import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import service


def test_integration_create_event_is_authenticated_and_idempotent(tmp_path):
    db_path = str(tmp_path / "events.sqlite3")
    api_key = "key-alice"
    service.register_user("alice", api_key, db_path)

    first_status, first_payload = service.create_event(
        api_key, "evt-1", {"amount": 10}, db_path
    )
    repeated_status, repeated_payload = service.create_event(
        api_key, "evt-1", {"amount": 99}, db_path
    )
    assert first_status == 201
    assert repeated_status == 200
    assert first_payload["event_id"] == repeated_payload["event_id"] == "evt-1"
    assert repeated_payload["payload"] == {"amount": 10}

    with sqlite3.connect(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_id = ?", ("evt-1",)
        ).fetchone()[0]
    assert count == 1

    status, payload = service.create_event("invalid", "evt-2", {}, db_path)
    assert status == 401
    assert payload == {"error": "unauthorized"}
'''
