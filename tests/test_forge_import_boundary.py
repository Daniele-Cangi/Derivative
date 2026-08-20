import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_SUBSTRATE_ROOTS = {
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


def test_constructing_substrate_does_not_load_optional_lens_runtimes():
    script = (
        "import json, sys; from core.substrate import CognitiveSubstrate; "
        "substrate = CognitiveSubstrate(execution_mode='local-only'); "
        f"roots = {sorted(OPTIONAL_SUBSTRATE_ROOTS)!r}; "
        "print(json.dumps({'lens_count': len(substrate.lenses), "
        "'loaded': sorted(name for name in roots if name in sys.modules)}))"
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
    assert json.loads(result.stdout) == {"lens_count": 7, "loaded": []}


def test_symbolic_problem_loads_sympy_without_loading_unrelated_lens_runtimes():
    script = (
        "import json, sys; from core.substrate import CognitiveSubstrate; "
        "CognitiveSubstrate(execution_mode='local-only').decompose("
        "'Derive the symbolic formula x = y + 1 and verify the invariant'); "
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
    assert json.loads(result.stdout) == ["sympy", "z3"]


def test_hybrid_kernel_loads_openai_client_without_making_a_request():
    script = (
        "import json, sys; from core.kernel import ReasoningKernel; "
        "before = 'openai' in sys.modules; "
        "kernel = ReasoningKernel(api_key='sk-test', execution_mode='hybrid'); "
        "print(json.dumps({'before': before, 'after': 'openai' in sys.modules, "
        "'client_created': kernel.client is not None, 'live': kernel.use_live_model}))"
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
    assert json.loads(result.stdout) == {
        "before": False,
        "after": True,
        "client_created": True,
        "live": True,
    }


def test_constructing_planner_does_not_load_networkx():
    script = (
        "import json, sys; from core.forge.planner_stage import PlannerStage; "
        "planner = PlannerStage(); "
        "print(json.dumps({'lens_count': len(planner.substrate.lenses), "
        "'networkx_loaded': 'networkx' in sys.modules}))"
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
    assert json.loads(result.stdout) == {"lens_count": 7, "networkx_loaded": False}


def test_topology_solver_loads_networkx_only_when_solving():
    script = (
        "import json, sys; "
        "from core.topology_solver import TopologySearchQuery, solve_topology_search; "
        "before = 'networkx' in sys.modules; "
        "query = TopologySearchQuery(node_count=4, physical_error_rate=0.001, "
        "fidelity_threshold=0.95, gate_operations=10, latency_limit_ms=50.0, "
        "entanglement_factor_limit=3.0); "
        "result = solve_topology_search(query); "
        "print(json.dumps({'before': before, 'after': 'networkx' in sys.modules, "
        "'evaluated': result.evaluated_topologies}))"
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
    payload = json.loads(result.stdout)
    assert payload["before"] is False
    assert payload["after"] is True
    assert payload["evaluated"] > 0
