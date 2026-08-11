from typing import List

from core.forge.contracts import FeasiblePlan


def render_service_api(plan: FeasiblePlan) -> str:
    quality = plan.quality_contract
    lines: List[str] = [
        "from typing import Any",
        "",
        "from audit import get_recent_events, record_event",
        "from auth import authenticate, register_user",
        "from domain import handle_request",
        "from observability import health_status",
        "from rate_limit import RATE_LIMIT_PER_MINUTE, enforce_rate_limit",
        "from storage import DB_PATH, init_db",
        "",
        f"QUALITY_AUTH_MODE = {quality.auth_level!r}",
        f"QUALITY_RATE_SCOPE = {quality.rate_limit_scope!r}",
        f"QUALITY_RATE_PERSISTENT = {quality.rate_limit_persistent!r}",
        f"QUALITY_SCHEMA_VERSIONED = {quality.schema_versioned!r}",
        f"QUALITY_AUDIT_TRAIL = {quality.audit_trail!r}",
        f"QUALITY_HEALTH_ENDPOINT = {quality.health_endpoint!r}",
        f"QUALITY_STRUCTURED_LOGGING = {quality.structured_logging!r}",
        f"QUALITY_INTEGRATION_TESTS = {quality.integration_tests!r}",
        f"QUALITY_LEVEL = {quality.overall_level}",
        "",
        "",
        "def create_app(db_path: str = DB_PATH) -> dict[str, object]:",
        "    init_db(db_path)",
        "    return {",
        "        'db_path': db_path,",
        "        'auth_mode': QUALITY_AUTH_MODE,",
        "        'rate_scope': QUALITY_RATE_SCOPE,",
        "        'quality_level': QUALITY_LEVEL,",
        "    }",
        "",
        "",
        "def run() -> int:",
        "    config = create_app(DB_PATH)",
        "    return 0 if config.get('db_path') else 1",
        "",
        "",
        "try:",
        "    from fastapi import FastAPI, Header",
        "",
        "    app = FastAPI(title='Forge Service')",
        "",
        "    @app.post('/events')",
        "    def post_event(payload: dict[str, Any], x_api_key: str = Header(default='')):",
        "        status_code, body = handle_request(x_api_key, payload, db_path=DB_PATH)",
        "        return {'status_code': status_code, **body}",
    ]
    if quality.health_endpoint:
        lines.extend(
            [
                "",
                "    @app.get('/health')",
                "    def health() -> dict[str, object]:",
                "        return health_status(DB_PATH)",
            ]
        )
    lines.extend(
        [
            "except Exception:",
            "    app = None",
            "",
            "__all__ = [",
            "    'DB_PATH', 'RATE_LIMIT_PER_MINUTE', 'QUALITY_AUTH_MODE', 'QUALITY_LEVEL',",
            "    'init_db', 'register_user', 'authenticate', 'enforce_rate_limit',",
            "    'record_event', 'get_recent_events', 'handle_request', 'health_status',",
            "    'create_app', 'run',",
            "]",
            "",
        ]
    )
    return "\n".join(lines)
