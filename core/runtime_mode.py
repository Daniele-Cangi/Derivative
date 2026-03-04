import os
from typing import Optional

VALID_EXECUTION_MODES = ("local-only", "hybrid", "remote-only")


def normalize_execution_mode(mode: Optional[str] = None) -> str:
    candidate = (mode or os.getenv("DERIVATIVE_EXECUTION_MODE") or "").strip().lower()
    if not candidate:
        return "hybrid"
    if candidate not in VALID_EXECUTION_MODES:
        supported = ", ".join(VALID_EXECUTION_MODES)
        raise ValueError(f"Unsupported execution mode '{candidate}'. Supported modes: {supported}.")
    return candidate
