import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_SUBSTRATE_ROOTS = {
    "dowhy",
    "networkx",
    "pgmpy",
    "pint",
    "qiskit",
    "qiskit_aer",
    "scipy",
    "sympy",
}


def test_importing_forge_does_not_load_optional_substrate_runtimes():
    script = (
        "import json, sys; import forge; "
        f"roots = {sorted(OPTIONAL_SUBSTRATE_ROOTS)!r}; "
        "print(json.dumps(sorted(name for name in roots if name in sys.modules)))"
    )

    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
