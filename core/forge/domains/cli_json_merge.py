from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


def is_recursive_json_merge_cli(plan: FeasiblePlan) -> bool:
    corpus = " ".join(
        [
            plan.build_spec.normalized_requirement,
            plan.architecture_summary,
            *(atom.text for atom in plan.build_spec.requirement_atoms),
        ]
    ).lower()
    return all(
        token in corpus
        for token in ("json", "merge", "recurs", "replaces lists", "non-object root")
    )


def render_json_merge_file(
    plan: FeasiblePlan,
    path: str,
    interfaces: list[PlanInterface],
) -> str | None:
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith(("src/cli.py", "src/main.py")):
        return _cli_module()
    if normalized.startswith("tests/"):
        return _behavioral_test()
    return None


def render_json_merge_test(plan: FeasiblePlan, plan_test: PlanTest) -> str:
    return _behavioral_test()


def _cli_module() -> str:
    return r'''import argparse
import json
from pathlib import Path
from typing import Any


def replace_lists(override: list[Any]) -> list[Any]:
    return list(override)


def recursive_json_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = recursive_json_merge(merged[key], value) if key in merged else value
        return merged
    if isinstance(override, list):
        return replace_lists(override)
    return override


def load_json_object(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recursively merge two JSON objects.")
    parser.add_argument("base_json_path")
    parser.add_argument("override_json_path")
    parser.add_argument("output_json_path")
    args = parser.parse_args(argv)
    base = load_json_object(args.base_json_path)
    override = load_json_object(args.override_json_path)
    merged = recursive_json_merge(base, override)
    output = Path(args.output_json_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _behavioral_test() -> str:
    return r'''import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cli


def test_recursive_json_merge_replaces_lists_and_rejects_non_object_root(tmp_path):
    base = tmp_path / "base.json"
    override = tmp_path / "override.json"
    output = tmp_path / "merged.json"
    base.write_text('{"db":{"host":"localhost","port":5432},"tags":["base"]}', encoding="utf-8")
    override.write_text('{"db":{"port":6432},"tags":["override"]}', encoding="utf-8")
    assert cli.main([str(base), str(override), str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "db": {"host": "localhost", "port": 6432},
        "tags": ["override"],
    }
    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["tags"] == ["override"]

    base.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        cli.main([str(base), str(override), str(output)])
'''
