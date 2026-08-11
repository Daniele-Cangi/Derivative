from typing import List

from core.forge.contracts import FeasiblePlan


def render_auth(plan: FeasiblePlan) -> str:
    quality = plan.quality_contract
    auth_level = quality.auth_level.lower()
    plaintext = bool(quality.secrets_in_plaintext and auth_level == "plaintext")
    lines: List[str] = [
        "import base64",
        "import hashlib",
        "import hmac",
        "import os",
        "import sqlite3",
    ]
    if auth_level == "hashed":
        lines.append("import bcrypt")
    lines.extend(
        [
            "",
            "from storage import DB_PATH, init_db",
            "",
            "JWT_SECRET = os.environ.get('FORGE_JWT_SECRET', 'forge-dev-secret')",
            f"AUTH_LEVEL = {auth_level!r}",
            "",
            "",
            "def _hash_api_key(api_key: str) -> str:",
        ]
    )
    if auth_level == "hashed":
        lines.extend(
            [
                "    token = bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt(rounds=4))",
                "    return token.decode('utf-8')",
            ]
        )
    else:
        lines.extend(
            [
                "    return hmac.new(",
                "        JWT_SECRET.encode('utf-8'),",
                "        api_key.encode('utf-8'),",
                "        hashlib.sha256,",
                "    ).hexdigest()",
            ]
        )
    lines.extend(
        [
            "",
            "",
            "def _verify_api_key(api_key: str, stored: str) -> bool:",
        ]
    )
    if auth_level == "hashed":
        lines.append("    return bool(bcrypt.checkpw(api_key.encode('utf-8'), stored.encode('utf-8')))")
    else:
        lines.append("    return hmac.compare_digest(_hash_api_key(api_key), stored)")
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
            "    return username if hmac.compare_digest(signature, expected_token) else None",
            "",
            "",
            "def register_user(username: str, api_key: str, db_path: str = DB_PATH) -> None:",
            "    init_db(db_path)",
            "    with sqlite3.connect(db_path) as conn:",
        ]
    )
    if plaintext:
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
    if auth_level == "jwt":
        lines.extend(
            [
                "    jwt_user = _verify_jwt_token(api_key)",
                "    if jwt_user is not None:",
                "        return jwt_user",
            ]
        )
    lines.append("    with sqlite3.connect(db_path) as conn:")
    if plaintext:
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
                "        rows = conn.execute('SELECT username, api_key_hash FROM users').fetchall()",
                "    for username, stored_hash in rows:",
                "        if _verify_api_key(api_key, str(stored_hash)):",
                "            return str(username)",
                "    return None",
            ]
        )
    lines.append("")
    return "\n".join(lines)
