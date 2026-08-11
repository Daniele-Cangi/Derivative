from typing import List

from core.forge.contracts import ArtifactTargetType, FeasiblePlan, PlanInterface, PlanTest
from core.forge.domains.base import BaseDomainAdapter


class ServiceDomainAdapter(BaseDomainAdapter):
    name = "service"

    def matches(self, plan: FeasiblePlan) -> bool:
        if plan.build_spec.target_artifact_type == ArtifactTargetType.SERVICE:
            return True
        paths = {item.path.replace("\\", "/").lower() for item in plan.file_tree_plan}
        return "src/service.py" in paths

    def render_file(self, plan: FeasiblePlan, path: str, interfaces: List[PlanInterface]) -> str:
        normalized = path.replace("\\", "/").lower()
        if normalized.endswith("src/service.py"):
            return self._template_service_entrypoint(plan)
        if normalized.endswith("src/domain.py"):
            return self._template_service_domain(plan)
        if normalized.startswith("tests/"):
            return self._template_service_plan_test_module(plan)
        return self._template_generic_module(path, interfaces)

    def render_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        name = plan_test.test_name.lower()
        objective = plan_test.objective.lower()
        if "suite_executes" in name or any(
            token in objective
            for token in ("service", "rest", "api", "authentication", "rate limit", "persistent")
        ):
            return self._template_service_suite_executes_test(plan)
        return self._template_service_requirement_test(plan)

    def _template_service_entrypoint(self, plan: FeasiblePlan) -> str:
        quality = plan.quality_contract
        auth_level = quality.auth_level.lower()
        use_hashed_auth = auth_level == "hashed"
        use_jwt_auth = auth_level == "jwt"
        secrets_plaintext = bool(quality.secrets_in_plaintext and auth_level == "plaintext")
        distributed_rate = quality.rate_limit_scope == "distributed"
        persistent_rate = bool(quality.rate_limit_persistent or distributed_rate)
        audit_trail = bool(quality.audit_trail)
        schema_versioned = bool(quality.schema_versioned)
        health_endpoint = bool(quality.health_endpoint)
        structured_logging = bool(quality.structured_logging)
        integration_tests = bool(quality.integration_tests)
        storage_column = "api_key TEXT UNIQUE NOT NULL" if secrets_plaintext else "api_key_hash TEXT UNIQUE NOT NULL"
        auth_mode_label = "jwt" if use_jwt_auth else ("hashed" if use_hashed_auth else "plaintext")

        lines: List[str] = [
            "import base64",
            "import hashlib",
            "import hmac",
            "import json",
            "import logging",
            "import os",
            "import sqlite3",
            "import time",
            "from typing import Any",
            "",
            "try:",
            "    import bcrypt  # type: ignore",
            "except Exception:",
            "    bcrypt = None",
            "",
            "try:",
            "    import redis  # type: ignore",
            "except Exception:",
            "    redis = None",
            "",
            "DB_PATH = os.environ.get('FORGE_SERVICE_DB', 'service.db')",
            "REDIS_URL = os.environ.get('FORGE_REDIS_URL', 'redis://localhost:6379/0')",
            "JWT_SECRET = os.environ.get('FORGE_JWT_SECRET', 'forge-dev-secret')",
            "RATE_LIMIT_PER_MINUTE = int(os.environ.get('FORGE_RATE_LIMIT_PER_MINUTE', '100'))",
            f"QUALITY_AUTH_MODE = {auth_mode_label!r}",
            f"QUALITY_RATE_SCOPE = {quality.rate_limit_scope!r}",
            f"QUALITY_RATE_PERSISTENT = {str(persistent_rate)}",
            f"QUALITY_SCHEMA_VERSIONED = {str(schema_versioned)}",
            f"QUALITY_AUDIT_TRAIL = {str(audit_trail)}",
            f"QUALITY_HEALTH_ENDPOINT = {str(health_endpoint)}",
            f"QUALITY_STRUCTURED_LOGGING = {str(structured_logging)}",
            f"QUALITY_INTEGRATION_TESTS = {str(integration_tests)}",
            f"QUALITY_LEVEL = {quality.overall_level}",
            "_logger = logging.getLogger('forge.service')",
            "if not _logger.handlers:",
            "    logging.basicConfig(level=logging.INFO)",
            "",
            "",
            "def init_db(db_path: str = DB_PATH) -> None:",
            "    with sqlite3.connect(db_path) as conn:",
            "        conn.execute(",
            "            'CREATE TABLE IF NOT EXISTS users ('",
            "            'username TEXT PRIMARY KEY, '",
            f"            '{storage_column})'",
            "        )",
        ]
        if schema_versioned:
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
        if audit_trail:
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
        lines.extend(
            [
                "        conn.commit()",
                "",
                "",
                "def _hash_api_key(api_key: str) -> str:",
                "    value = api_key.encode('utf-8')",
            ]
        )
        if use_hashed_auth:
            lines.extend(
                [
                    "    if bcrypt is None:",
                    "        raise RuntimeError('bcrypt required but not available')",
                    "    token = bcrypt.hashpw(value, bcrypt.gensalt(rounds=4)).decode('utf-8')",
                    "    return f'bcrypt${token}'",
                ]
            )
        else:
            lines.extend(
                [
                    "    if bcrypt is not None and os.environ.get('FORGE_USE_BCRYPT', '0') == '1':",
                    "        token = bcrypt.hashpw(value, bcrypt.gensalt(rounds=4)).decode('utf-8')",
                    "        return f'bcrypt${token}'",
                    "    digest = hashlib.sha256(value + b'forge-fallback-salt').hexdigest()",
                    "    return f'sha256${digest}'",
                ]
            )
        lines.extend(["", "", "def _verify_api_key(api_key: str, stored: str) -> bool:", "    value = api_key.encode('utf-8')"])
        if use_hashed_auth:
            lines.extend(
                [
                    "    if not stored.startswith('bcrypt$'):",
                    "        return False",
                    "    if bcrypt is None:",
                    "        return False",
                    "    candidate = stored.split('$', 1)[1]",
                    "    return bool(bcrypt.checkpw(value, candidate.encode('utf-8')))",
                ]
            )
        else:
            lines.extend(
                [
                    "    if stored.startswith('bcrypt$') and bcrypt is not None:",
                    "        candidate = stored.split('$', 1)[1]",
                    "        return bool(bcrypt.checkpw(value, candidate.encode('utf-8')))",
                    "    if stored.startswith('sha256$'):",
                    "        expected = hashlib.sha256(value + b'forge-fallback-salt').hexdigest()",
                    "        return stored.split('$', 1)[1] == expected",
                    "    return False",
                ]
            )
        lines.extend(
            [
                "",
                "",
                "def _verify_jwt_token(token: str) -> str | None:",
                "    candidate = token.strip()",
                "    if candidate.lower().startswith('bearer '):",
                "        candidate = candidate.split(' ', 1)[1].strip()",
                "    parts = candidate.split('.', 1)",
                "    if len(parts) != 2:",
                "        return None",
                "    username, signature = parts",
                "    expected = hmac.new(",
                "        JWT_SECRET.encode('utf-8'),",
                "        username.encode('utf-8'),",
                "        hashlib.sha256,",
                "    ).digest()",
                "    expected_token = base64.urlsafe_b64encode(expected).decode('utf-8').rstrip('=')",
                "    if hmac.compare_digest(signature, expected_token):",
                "        return username",
                "    return None",
                "",
                "",
                "def register_user(username: str, api_key: str, db_path: str = DB_PATH) -> None:",
                "    init_db(db_path)",
                "    with sqlite3.connect(db_path) as conn:",
            ]
        )
        if secrets_plaintext:
            lines.extend(
                [
                    "        conn.execute(",
                    "            'INSERT OR REPLACE INTO users(username, api_key) VALUES (?, ?)',",
                    "            (username, api_key),",
                    "        )",
                ]
            )
        else:
            lines.extend(
                [
                    "        conn.execute(",
                    "            'INSERT OR REPLACE INTO users(username, api_key_hash) VALUES (?, ?)',",
                    "            (username, _hash_api_key(api_key)),",
                    "        )",
                ]
            )
        lines.extend(
            [
                "        conn.commit()",
                "",
                "",
                "def authenticate(api_key: str, db_path: str = DB_PATH) -> str | None:",
                "    init_db(db_path)",
            ]
        )
        if use_jwt_auth:
            lines.extend(
                [
                    "    jwt_user = _verify_jwt_token(api_key)",
                    "    if jwt_user is not None:",
                    "        return jwt_user",
                ]
            )
        lines.extend(
            [
                "    with sqlite3.connect(db_path) as conn:",
            ]
        )
        if secrets_plaintext:
            lines.extend(
                [
                    "        row = conn.execute(",
                    "            'SELECT username FROM users WHERE api_key = ?',",
                    "            (api_key,),",
                    "        ).fetchone()",
                    "    return str(row[0]) if row else None",
                ]
            )
        else:
            lines.extend(
                [
                    "        rows = conn.execute(",
                    "            'SELECT username, api_key_hash FROM users'",
                    "        ).fetchall()",
                    "    for username, api_key_hash in rows:",
                    "        if _verify_api_key(api_key, str(api_key_hash)):",
                    "            return str(username)",
                    "    return None",
                ]
            )
        lines.extend(
            [
                "",
                "",
                "def _redis_client():",
                "    if redis is None:",
                "        return None",
                "    try:",
                "        return redis.from_url(REDIS_URL)",
                "    except Exception:",
                "        return None",
                "",
                "",
                "def _allow_rate_with_sqlite(user_id: str, limit: int, timestamp: float, db_path: str) -> bool:",
                "    init_db(db_path)",
                "    window_start = timestamp - 60.0",
                "    with sqlite3.connect(db_path) as conn:",
                "        conn.execute('DELETE FROM rate_limit_hits WHERE ts < ?', (window_start,))",
                "        current = conn.execute(",
                "            'SELECT COUNT(1) FROM rate_limit_hits WHERE user_id = ? AND ts >= ?',",
                "            (user_id, window_start),",
                "        ).fetchone()",
                "        count = int(current[0]) if current else 0",
                "        if count >= limit:",
                "            conn.commit()",
                "            return False",
                "        conn.execute(",
                "            'INSERT INTO rate_limit_hits(user_id, ts) VALUES (?, ?)',",
                "            (user_id, timestamp),",
                "        )",
                "        conn.commit()",
                "    return True",
            ]
        )
        if not persistent_rate:
            lines.extend(
                [
                    "_RATE_LIMIT_BUCKETS: dict[str, list[float]] = {}",
                    "",
                    "",
                    "def _allow_rate_with_memory(user_id: str, limit: int, timestamp: float) -> bool:",
                    "    window_start = timestamp - 60.0",
                    "    bucket = _RATE_LIMIT_BUCKETS.setdefault(user_id, [])",
                    "    bucket[:] = [entry for entry in bucket if entry >= window_start]",
                    "    if len(bucket) >= limit:",
                    "        return False",
                    "    bucket.append(timestamp)",
                    "    return True",
                ]
            )
            lines.extend(
                [
                    "",
                    "",
                    "def enforce_rate_limit(",
                    "    user_id: str,",
                    "    limit: int = RATE_LIMIT_PER_MINUTE,",
                    "    now: float | None = None,",
                    "    db_path: str = DB_PATH,",
                    ") -> bool:",
                    "    timestamp = time.time() if now is None else float(now)",
                    "    return _allow_rate_with_memory(user_id, limit, timestamp)",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "",
                    "def enforce_rate_limit(",
                    "    user_id: str,",
                    "    limit: int = RATE_LIMIT_PER_MINUTE,",
                    "    now: float | None = None,",
                    "    db_path: str = DB_PATH,",
                    ") -> bool:",
                    "    timestamp = time.time() if now is None else float(now)",
                ]
            )
        if distributed_rate:
            lines.extend(
                [
                    "    client = _redis_client()",
                    "    if client is not None:",
                    "        key = f'rl:{user_id}'",
                    "        window_start = timestamp - 60.0",
                    "        try:",
                    "            client.zremrangebyscore(key, 0, window_start)",
                    "            count = int(client.zcard(key))",
                    "            if count >= limit:",
                    "                return False",
                    "            client.zadd(key, {str(timestamp): timestamp})",
                    "            client.expire(key, 120)",
                    "            return True",
                    "        except Exception:",
                    "            pass",
                    "    return _allow_rate_with_sqlite(user_id, limit, timestamp, db_path)",
                ]
            )
        elif persistent_rate:
            lines.append("    return _allow_rate_with_sqlite(user_id, limit, timestamp, db_path)")
        lines.extend(
            [
                "",
                "",
                "def _log_event(user_id: str | None, status_code: int, detail: str, db_path: str) -> None:",
            ]
        )
        if audit_trail:
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
        if structured_logging:
            lines.extend(
                [
                    "    _logger.info(",
                    "        json.dumps(",
                    "            {",
                    "                'user_id': user_id,",
                    "                'status_code': status_code,",
                    "                'detail': detail,",
                    "            },",
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
        if audit_trail:
            lines.extend(
                [
                    "    init_db(db_path)",
                    "    with sqlite3.connect(db_path) as conn:",
                    "        rows = conn.execute(",
                    "            'SELECT created_at, user_id, status_code, detail FROM events ORDER BY id DESC LIMIT ?',",
                    "            (limit,),",
                    "        ).fetchall()",
                    "    return [",
                    "        {",
                    "            'created_at': float(row[0]),",
                    "            'user_id': row[1],",
                    "            'status_code': int(row[2]),",
                    "            'detail': str(row[3]),",
                    "        }",
                    "        for row in rows",
                    "    ]",
                ]
            )
        else:
            lines.append("    return []")
        lines.extend(
            [
                "",
                "",
                "def create_app(db_path: str = DB_PATH) -> dict[str, str]:",
                "    init_db(db_path)",
                "    return {'db_path': db_path, 'auth_mode': QUALITY_AUTH_MODE, 'rate_scope': QUALITY_RATE_SCOPE}",
                "",
                "",
                "def handle_request(",
                "    api_key: str,",
                "    payload: dict[str, Any],",
                "    db_path: str = DB_PATH,",
                "    now: float | None = None,",
                ") -> tuple[int, dict[str, Any]]:",
                "    user_id = authenticate(api_key, db_path=db_path)",
                "    if user_id is None:",
                "        _log_event(None, 401, 'unauthorized', db_path)",
                "        return 401, {'error': 'unauthorized'}",
                "    if not enforce_rate_limit(user_id, now=now, db_path=db_path):",
                "        _log_event(user_id, 429, 'rate_limit_exceeded', db_path)",
                "        return 429, {'error': 'rate_limit_exceeded'}",
                "    _log_event(user_id, 200, 'accepted', db_path)",
                "    return 200, {'status': 'ok', 'user': user_id, 'payload': payload}",
                "",
                "",
                "try:",
                "    from fastapi import FastAPI, Header, HTTPException",
                "",
                "    app = FastAPI(title='Forge Service')",
            ]
        )
        if health_endpoint:
            lines.extend(
                [
                    "",
                    "    @app.get('/health')",
                    "    def health() -> dict[str, object]:",
                    "        return {'status': 'ok', 'quality_level': QUALITY_LEVEL}",
                ]
            )
        lines.extend(
            [
                "",
                "    @app.post('/events')",
                "    def events(payload: dict[str, Any], x_api_key: str | None = Header(default=None)) -> dict[str, Any]:",
                "        if not x_api_key:",
                "            raise HTTPException(status_code=401, detail='missing_api_key')",
                "        code, body = handle_request(x_api_key, payload)",
                "        if code != 200:",
                "            raise HTTPException(status_code=code, detail=str(body.get('error', 'request_rejected')))",
                "        return body",
                "except Exception:",
                "    app = None",
                "",
                "",
                "def run() -> int:",
                "    config = create_app(DB_PATH)",
                "    return 0 if 'db_path' in config else 1",
            ]
        )
        return "\n".join(lines) + "\n"

    def _template_service_domain(self, plan: FeasiblePlan) -> str:
        _ = plan
        return (
            "from service import (\n"
            "    authenticate,\n"
            "    create_app,\n"
            "    enforce_rate_limit,\n"
            "    get_recent_events,\n"
            "    handle_request,\n"
            "    init_db,\n"
            "    register_user,\n"
            "    run,\n"
            ")\n"
            "\n"
            "__all__ = [\n"
            "    'init_db',\n"
            "    'register_user',\n"
            "    'authenticate',\n"
            "    'enforce_rate_limit',\n"
            "    'handle_request',\n"
            "    'get_recent_events',\n"
            "    'create_app',\n"
            "    'run',\n"
            "]\n"
        )
    def _template_service_plan_test_module(self, plan: FeasiblePlan) -> str:
        quality = plan.quality_contract
        extra_assert = "    assert service.QUALITY_LEVEL >= 8\n" if quality.overall_level >= 8 else ""
        return (
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "\n"
            "import service\n"
            "\n"
            "\n"
            "def test_service_module_importable():\n"
            "    assert callable(service.run)\n"
            "    assert callable(service.handle_request)\n"
            + extra_assert
        )

    def _template_service_suite_executes_test(self, plan: FeasiblePlan) -> str:
        quality = plan.quality_contract
        lines = [
            "from pathlib import Path",
            "import sys",
            "",
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))",
            "",
            "import service",
            "",
            "",
            "def test_rejects_unauthorized_requests(tmp_path):",
            "    db_path = tmp_path / 'service.sqlite3'",
            "    service.init_db(str(db_path))",
            "    code, payload = service.handle_request(",
            "        'invalid-key',",
            "        {'event': 'x'},",
            "        db_path=str(db_path),",
            "        now=1000.0,",
            "    )",
            "    assert code == 401",
            "    assert payload['error'] == 'unauthorized'",
            "",
            "",
            "def test_enforces_rate_limit(tmp_path):",
            "    db_path = tmp_path / 'service.sqlite3'",
            "    service.register_user('alice', 'key-alice', db_path=str(db_path))",
            "    for index in range(100):",
            "        code, _ = service.handle_request(",
            "            'key-alice',",
            "            {'index': index},",
            "            db_path=str(db_path),",
            "            now=1000.0 + index * 0.1,",
            "        )",
            "        assert code == 200",
            "    code, payload = service.handle_request(",
            "        'key-alice',",
            "        {'index': 101},",
            "        db_path=str(db_path),",
            "        now=1011.0,",
            "    )",
            "    assert code == 429",
            "    assert payload['error'] == 'rate_limit_exceeded'",
            "",
            "",
            "def test_persists_user_credentials_across_reloads(tmp_path):",
            "    db_path = tmp_path / 'service.sqlite3'",
            "    service.register_user('bob', 'key-bob', db_path=str(db_path))",
            "    user = service.authenticate('key-bob', db_path=str(db_path))",
            "    assert user == 'bob'",
            "    app_config = service.create_app(str(db_path))",
            "    assert app_config['db_path'] == str(db_path)",
            "    assert service.run() == 0",
        ]
        if quality.audit_trail:
            lines.extend(
                [
                    "",
                    "",
                    "def test_audit_trail_records_requests(tmp_path):",
                    "    db_path = tmp_path / 'service.sqlite3'",
                    "    service.register_user('eve', 'key-eve', db_path=str(db_path))",
                    "    service.handle_request('key-eve', {'hello': 'world'}, db_path=str(db_path), now=1000.0)",
                    "    events = service.get_recent_events(str(db_path), limit=5)",
                    "    assert events",
                    "    assert any(event['status_code'] in {200, 401, 429} for event in events)",
                ]
            )
        if quality.integration_tests:
            lines.extend(
                [
                    "",
                    "",
                    "def test_integration_flow_handles_valid_request(tmp_path):",
                    "    db_path = tmp_path / 'service.sqlite3'",
                    "    service.register_user('intg', 'key-intg', db_path=str(db_path))",
                    "    code, payload = service.handle_request(",
                    "        'key-intg',",
                    "        {'integration': True},",
                    "        db_path=str(db_path),",
                    "        now=1000.0,",
                    "    )",
                    "    assert code == 200",
                    "    assert payload['user'] == 'intg'",
                ]
            )
        if quality.auth_level == "hashed":
            lines.extend(
                [
                    "",
                    "",
                    "def test_hashed_auth_path_stores_hashes(tmp_path):",
                    "    db_path = tmp_path / 'service.sqlite3'",
                    "    service.register_user('hash', 'key-hash', db_path=str(db_path))",
                    "    assert service.authenticate('key-hash', db_path=str(db_path)) == 'hash'",
                ]
            )
        return "\n".join(lines) + "\n"

    def _template_service_requirement_test(self, plan: FeasiblePlan) -> str:
        quality = plan.quality_contract
        auth_assert = (
            "    assert service.QUALITY_AUTH_MODE in {'hashed', 'jwt'}\n"
            if quality.auth_level in {"hashed", "jwt"}
            else "    assert service.QUALITY_AUTH_MODE == 'plaintext'\n"
        )
        return (
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "\n"
            "import service\n"
            "\n"
            "\n"
            "def test_service_requirement_smoke(tmp_path):\n"
            "    db_path = tmp_path / 'service.sqlite3'\n"
            "    service.register_user('smoke', 'key-smoke', db_path=str(db_path))\n"
            "    code, payload = service.handle_request('key-smoke', {'ok': True}, db_path=str(db_path), now=1000.0)\n"
            "    assert code == 200\n"
            "    assert payload['user'] == 'smoke'\n"
            + auth_assert
        )
