from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_RUNTIME_ROOTS = {
    "dowhy",
    "networkx",
    "openai",
    "pgmpy",
    "pint",
    "qiskit",
    "qiskit_aer",
    "scipy",
    "sympy",
    "z3",
}


def _loaded_optional_runtimes() -> list[str]:
    return sorted(name for name in OPTIONAL_RUNTIME_ROOTS if name in sys.modules)


def main() -> int:
    sys.path.insert(0, str(REPOSITORY_ROOT))

    import forge
    from core.forge.contracts import FeasiblePlan
    from core.forge.planner_stage import PlannerStage
    from core.forge.requirement_compiler import RequirementCompiler

    loaded_after_import = _loaded_optional_runtimes()
    if loaded_after_import:
        raise RuntimeError(
            f"Minimal Forge import loaded optional runtimes: {loaded_after_import}"
        )

    with tempfile.TemporaryDirectory(prefix="forge-minimal-") as temp_directory:
        state_root = Path(temp_directory)
        planner = PlannerStage(
            execution_mode="local-only",
            audit_log_file=str(state_root / "audit.json"),
            memory_file=str(state_root / "memory.json"),
            gene_pool_file=str(state_root / "gene-pool.json"),
        )
        build_spec = RequirementCompiler().compile(
            "Build a Python CLI that echoes one text argument and includes tests."
        )
        plan = planner.plan(build_spec)

    if not isinstance(plan, FeasiblePlan):
        raise RuntimeError(f"Minimal local planning returned {type(plan).__name__}")

    loaded_after_plan = _loaded_optional_runtimes()
    if loaded_after_plan:
        raise RuntimeError(
            f"Minimal local planning loaded optional runtimes: {loaded_after_plan}"
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "forge_module": forge.__name__,
                "plan_type": type(plan).__name__,
                "planned_files": len(plan.file_tree_plan),
                "optional_runtimes_loaded": loaded_after_plan,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
