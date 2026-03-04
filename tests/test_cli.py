from typer.testing import CliRunner

import derivative
from audit.trail import AuditTrail
from core.workspace import ArtifactWorkspace
from memory.delta import DeltaMemory
from memory.gene_pool import DesignGenePool


EXACT_QUERY = """
Given a quantum error-corrected distributed system with N=4 nodes,
where each node maintains a logical qubit with physical error rate p=0.001,
find all satisfiable topological configurations such that:

1. End-to-end fidelity > 0.95 after 50 gate operations
2. No single node failure causes system-wide decoherence
3. Classical control latency between any two nodes < 50ms
4. Total entanglement overhead does not exceed 3x the minimum spanning tree

Which configurations exist? Which is optimal under simultaneous
novelty + feasibility scoring? Emit a verifiable circuit.
""".strip()


class QuietProgress:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *args, **kwargs):
        return 1

    def update(self, *args, **kwargs):
        return None


def test_cli_emits_prose_without_table_delimiters(tmp_path, monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(derivative, "memory", DeltaMemory(storage_file=str(tmp_path / "memory.json")))
    monkeypatch.setattr(derivative, "audit", AuditTrail(log_file=str(tmp_path / "audit.json")))
    monkeypatch.setattr(derivative, "gene_pool", DesignGenePool(storage_file=str(tmp_path / "gene_pool.json")))
    monkeypatch.setattr(
        derivative,
        "ArtifactWorkspace",
        lambda: ArtifactWorkspace(base_dir=str(tmp_path / "generated_artifacts")),
    )
    monkeypatch.setattr(derivative, "Progress", QuietProgress)

    result = runner.invoke(derivative.app, [EXACT_QUERY, "--mode", "local-only"])

    assert result.exit_code == 0
    assert "Derivative - Self-Executing Reasoning Engine" in result.output
    assert "Cycles used:" in result.output
    assert "Artifacts saved to:" in result.output
    assert "|" not in result.output
