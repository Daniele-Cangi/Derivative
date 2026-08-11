from core.forge.contracts import FeasiblePlan


def render_domain(plan: FeasiblePlan) -> str:
    _ = plan
    return (
        "from typing import Any\n"
        "\n"
        "from audit import record_event\n"
        "from auth import authenticate\n"
        "from rate_limit import enforce_rate_limit\n"
        "from storage import DB_PATH\n"
        "\n"
        "\n"
        "def handle_request(\n"
        "    api_key: str,\n"
        "    payload: dict[str, Any],\n"
        "    db_path: str = DB_PATH,\n"
        "    now: float | None = None,\n"
        ") -> tuple[int, dict[str, Any]]:\n"
        "    user_id = authenticate(api_key, db_path=db_path)\n"
        "    if user_id is None:\n"
        "        record_event(None, 401, 'unauthorized', db_path)\n"
        "        return 401, {'error': 'unauthorized'}\n"
        "    if not enforce_rate_limit(user_id, now=now, db_path=db_path):\n"
        "        record_event(user_id, 429, 'rate_limit_exceeded', db_path)\n"
        "        return 429, {'error': 'rate_limit_exceeded'}\n"
        "    record_event(user_id, 200, 'accepted', db_path)\n"
        "    return 200, {'accepted': True, 'user': user_id, 'payload': payload}\n"
    )
