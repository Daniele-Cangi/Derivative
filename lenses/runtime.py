from functools import lru_cache
from importlib import import_module
from types import ModuleType


@lru_cache(maxsize=None)
def load_optional_module(module_name: str) -> tuple[ModuleType | None, str]:
    try:
        return import_module(module_name), ""
    except ImportError as exc:
        return None, str(exc)
