import sqlite3

import service


def test_authenticated_idempotent_event_contract(tmp_path):
    db_path = str(tmp_path / "events.sqlite3")
    service.register_user("alice", "key-alice", db_path=db_path)

    first_status, first_payload = service.create_event(
        "key-alice", "evt-1", {"amount": 10}, db_path
    )
    second_status, second_payload = service.create_event(
        "key-alice", "evt-1", {"amount": 99}, db_path
    )
    assert first_status in {200, 201}
    assert second_status == 200
    assert second_payload["event_id"] == first_payload["event_id"] == "evt-1"

    with sqlite3.connect(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_id = ?", ("evt-1",)
        ).fetchone()[0]
    assert count == 1

    status, payload = service.create_event("invalid", "evt-2", {}, db_path)
    assert status == 401
    assert payload["error"] == "unauthorized"
