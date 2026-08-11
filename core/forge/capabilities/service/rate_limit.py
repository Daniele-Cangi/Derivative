from typing import List

from core.forge.contracts import FeasiblePlan


def render_rate_limit(plan: FeasiblePlan) -> str:
    quality = plan.quality_contract
    distributed = quality.rate_limit_scope == "distributed"
    persistent = bool(quality.rate_limit_persistent or distributed)
    lines: List[str] = [
        "import os",
        "import sqlite3",
        "import time",
    ]
    if distributed:
        lines.extend(
            [
                "try:",
                "    import redis",
                "except Exception:",
                "    redis = None",
            ]
        )
    lines.extend(
        [
            "",
            "from storage import DB_PATH, init_db",
            "",
            "RATE_LIMIT_PER_MINUTE = int(os.environ.get('FORGE_RATE_LIMIT_PER_MINUTE', '100'))",
            f"RATE_LIMIT_SCOPE = {quality.rate_limit_scope!r}",
            f"RATE_LIMIT_PERSISTENT = {persistent!r}",
        ]
    )
    if persistent:
        lines.extend(
            [
                "",
                "",
                "def _allow_sqlite(user_id: str, limit: int, timestamp: float, db_path: str) -> bool:",
                "    init_db(db_path)",
                "    window_start = timestamp - 60.0",
                "    with sqlite3.connect(db_path) as conn:",
                "        conn.execute('DELETE FROM rate_limit_hits WHERE ts < ?', (window_start,))",
                "        row = conn.execute(",
                "            'SELECT COUNT(1) FROM rate_limit_hits WHERE user_id = ? AND ts >= ?',",
                "            (user_id, window_start),",
                "        ).fetchone()",
                "        if row and int(row[0]) >= limit:",
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
    else:
        lines.extend(
            [
                "",
                "_RATE_LIMIT_BUCKETS: dict[str, list[float]] = {}",
                "",
                "",
                "def _allow_memory(user_id: str, limit: int, timestamp: float) -> bool:",
                "    window_start = timestamp - 60.0",
                "    bucket = _RATE_LIMIT_BUCKETS.setdefault(user_id, [])",
                "    bucket[:] = [entry for entry in bucket if entry >= window_start]",
                "    if len(bucket) >= limit:",
                "        return False",
                "    bucket.append(timestamp)",
                "    return True",
            ]
        )
    if distributed:
        lines.extend(
            [
                "",
                "",
                "def _redis_client():",
                "    if redis is None:",
                "        return None",
                "    try:",
                "        return redis.from_url(os.environ.get('FORGE_REDIS_URL', 'redis://localhost:6379/0'))",
                "    except Exception:",
                "        return None",
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
        ]
    )
    if distributed:
        lines.extend(
            [
                "    client = _redis_client()",
                "    if client is not None:",
                "        key = f'rl:{user_id}'",
                "        try:",
                "            client.zremrangebyscore(key, 0, timestamp - 60.0)",
                "            if int(client.zcard(key)) >= limit:",
                "                return False",
                "            client.zadd(key, {str(timestamp): timestamp})",
                "            client.expire(key, 120)",
                "            return True",
                "        except Exception:",
                "            pass",
                "    return _allow_sqlite(user_id, limit, timestamp, db_path)",
            ]
        )
    elif persistent:
        lines.append("    return _allow_sqlite(user_id, limit, timestamp, db_path)")
    else:
        lines.append("    return _allow_memory(user_id, limit, timestamp)")
    lines.append("")
    return "\n".join(lines)
