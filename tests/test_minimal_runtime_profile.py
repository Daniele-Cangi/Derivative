import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPOSITORY_ROOT / ".github" / "scripts" / "minimal_forge_smoke.py"
CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "forge-ci.yml"


def test_minimal_forge_smoke_builds_a_local_plan_without_optional_runtimes():
    result = subprocess.run(
        [sys.executable, "-B", str(SMOKE_SCRIPT)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["plan_type"] == "FeasiblePlan"
    assert payload["planned_files"] > 0
    assert payload["optional_runtimes_loaded"] == []


def test_ci_has_an_independent_minimal_profile_gate():
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    minimal_job = workflow.split("  minimal-forge-runtime:", maxsplit=1)[1].split(
        "  test-and-gate:", maxsplit=1
    )[0]

    assert "python -m pip install -r requirements/forge.txt" in minimal_job
    assert "requirements/all.txt" not in minimal_job
    assert "minimal_forge_smoke.py" in minimal_job
    assert "python -B forge.py --help" in minimal_job
