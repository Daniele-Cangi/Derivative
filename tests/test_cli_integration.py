import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from core.execution_loop import ExecutionLoop
from lenses.base import CognitiveLens


SYMBOLIC_PROMPT = (
    "A recursive function f(n) is defined as f(0)=1, f(1)=2, f(n)=3*f(n-1) - f(n-2) for n>=2. "
    "Find the closed-form expression for f(n), verify it satisfies the recurrence for n=0 through n=10, "
    "and determine for which values of n the function exceeds 10^6. Then prove that the ratio f(n+1)/f(n) "
    "converges and find the limit."
)

SURVIVAL_PROMPT = (
    "A reliability network has N=5 channels with per-hour failure probability p=0.03. "
    "Compute the probability the system survives 8 consecutive hours, and compute the expected number "
    "of hours before failure under independent and correlated failure models."
)

SURVIVAL_COMBINATORIAL_PROMPT = (
    "A system has 4 independent components, each failing with probability p=0.1 per hour. "
    "The system fails if 2 or more components fail in the same hour. "
    "Find the probability that the system survives exactly 8 consecutive hours without failure. "
    "Then find the expected number of hours until first system failure."
)

REGULAR_ENUMERATION_PROMPT = (
    "Find all non-isomorphic connected graphs on exactly 6 nodes where every node has degree exactly 3."
)

IMPOSSIBLE_PROMPT = (
    "Design a network on exactly 4 nodes such that: every pair of nodes is directly connected, "
    "the network diameter is strictly greater than 2, the vertex connectivity is at least 3, "
    "and the total number of edges does not exceed 3."
)


ANSI_PATTERN = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
REPO_ROOT = Path(__file__).resolve().parents[1]
DERIVATIVE_ENTRYPOINT = REPO_ROOT / "derivative.py"


def _normalize_cli_output(raw: str) -> str:
    return " ".join(ANSI_PATTERN.sub("", raw).split())


def _run_derivative_cli(prompt: str, workdir: Path) -> str:
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = "dummy_key_for_testing"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(REPO_ROOT)
    )
    completed = subprocess.run(
        [sys.executable, str(DERIVATIVE_ENTRYPOINT), prompt, "--mode", "local-only"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
        env=env,
    )
    output = _normalize_cli_output(f"{completed.stdout}\n{completed.stderr}")
    assert completed.returncode == 0, output
    assert "Derivative - Self-Executing Reasoning Engine" in output
    return output


def _inject_memory_seed(workdir: Path, prompt: str) -> None:
    payload = [
        {
            "timestamp": "2026-03-05T00:00:00+00:00",
            "problem_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "lens_contributions": {"Lineage": 1.0},
            "reasoning_delta": "Seeded context for integration validation.",
            "confidence_delta": 0.5,
            "confidence_score": 0.95,
            "conclusion_snapshot": "Seed snapshot.",
            "top_design_titles": ["Seed Design A"],
            "top_design_primitives": ["Mutate survival via prior genome"],
            "execution_cycle_summaries": [],
            "verified_hypotheses": [],
        }
    ]
    (workdir / "memory_deltas.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _diagnostic_survival_values(prompt: str) -> tuple[float, float]:
    loop = ExecutionLoop()
    lenses = [
        CognitiveLens("Probabilistic", "framing", ["c1"], ["b1"], 0.9, "probabilistic"),
        CognitiveLens("Topological", "framing", ["c2"], ["b2"], 0.7, "deductive"),
    ]
    result = loop.run(prompt, lenses, bypass_seeds=True)
    payload = json.loads(result.final_output)
    values = payload.get("result", {})
    return float(values["survival_probability"]), float(values["expected_hours"])


def test_cli_symbolic_prompt_end_to_end(tmp_path):
    output = _run_derivative_cli(SYMBOLIC_PROMPT, tmp_path)

    assert "Status: converged." in output
    assert "closed form f(n) =" in output
    assert "threshold 1000000 reached at n=15" in output
    assert "ratio limit" in output
    assert "sqrt(5)" in output
    assert "f(n) = n + 1" not in output


def test_cli_survival_prompt_end_to_end(tmp_path):
    output = _run_derivative_cli(SURVIVAL_PROMPT, tmp_path)

    assert "Status: converged." in output
    assert "survival probability" in output
    assert "expected runtime" in output
    assert "over 8 hour(s)" in output
    assert "over 1 hour(s)" not in output

    survival_match = re.search(r"survival probability of ([0-9]*\.[0-9]+) over 8 hour\(s\)", output)
    expected_match = re.search(r"expected runtime of ([0-9]*\.[0-9]+) hours", output)
    assert survival_match is not None
    assert expected_match is not None
    assert 0.60 <= float(survival_match.group(1)) <= 0.70
    assert 18.0 <= float(expected_match.group(1)) <= 21.0


def test_cli_survival_combinatorial_prompt_end_to_end(tmp_path):
    output = _run_derivative_cli(SURVIVAL_COMBINATORIAL_PROMPT, tmp_path)

    assert "Status: converged." in output
    assert "over 8 hour(s)" in output
    survival_match = re.search(r"survival probability of ([0-9]*\.[0-9]+) over 8 hour\(s\)", output)
    expected_match = re.search(r"expected runtime of ([0-9]*\.[0-9]+) hours", output)
    assert survival_match is not None
    assert expected_match is not None
    survival_value = float(survival_match.group(1))
    expected_value = float(expected_match.group(1))
    assert 0.64 <= survival_value <= 0.67
    assert 19.0 <= expected_value <= 19.3
    assert abs(survival_value - 0.430467) > 0.1


def test_cli_survival_seed_immune_matches_diagnostic_path(tmp_path):
    _inject_memory_seed(tmp_path, SURVIVAL_PROMPT)
    diagnostic_survival, diagnostic_expected = _diagnostic_survival_values(SURVIVAL_PROMPT)

    output = _run_derivative_cli(SURVIVAL_PROMPT, tmp_path)
    assert "evolutionary seed snapshot" in output

    survival_match = re.search(r"survival probability of ([0-9]*\.[0-9]+) over 8 hour\(s\)", output)
    expected_match = re.search(r"expected runtime of ([0-9]*\.[0-9]+) hours", output)
    assert survival_match is not None
    assert expected_match is not None

    cli_survival = float(survival_match.group(1))
    cli_expected = float(expected_match.group(1))
    assert abs(cli_survival - round(diagnostic_survival, 6)) <= 1e-6
    assert abs(cli_expected - round(diagnostic_expected, 3)) <= 1e-6


def test_cli_regular_enumeration_prompt_end_to_end(tmp_path):
    output = _run_derivative_cli(REGULAR_ENUMERATION_PROMPT, tmp_path)

    assert "Status: converged." in output
    count_match = re.search(r"found (\d+) satisfiable 3-regular topology configuration\(s\)", output)
    diameter_match = re.search(r"diameter (\d+)", output)
    connectivity_match = re.search(r"connectivity (\d+)", output)
    assert count_match is not None
    assert int(count_match.group(1)) == 2
    assert diameter_match is not None
    assert connectivity_match is not None
    assert int(diameter_match.group(1)) > 0
    assert int(connectivity_match.group(1)) > 0
    assert "candidate is G001" in output


def test_cli_impossible_prompt_end_to_end(tmp_path):
    output = _run_derivative_cli(IMPOSSIBLE_PROMPT, tmp_path)
    lowered = output.lower()

    assert "unsatisfiable" in lowered or "infeasible" in lowered or "no valid configuration" in lowered
    confidence_match = re.search(r"confidence:\s*([0-9]*\.[0-9]+)", lowered)
    assert confidence_match is not None
    assert float(confidence_match.group(1)) < 0.50
    assert "complete graph has diameter 1" in lowered or "needs 6 edges" in lowered
