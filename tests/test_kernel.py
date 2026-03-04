import pytest
from types import SimpleNamespace

from core.kernel import ReasoningKernel
from lenses.base import CognitiveLens


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


def test_kernel_synthesis_uses_local_fallback():
    kernel = ReasoningKernel(api_key="dummy_key_for_testing")
    lenses = [
        CognitiveLens("MockLens1", "framing1", ["constraint1"], ["spot1"], 0.8, "deductive"),
        CognitiveLens("MockLens2", "framing2", ["constraint2"], ["spot2"], 0.7, "causal"),
    ]

    result = kernel.synthesize("What are the failure modes of this code?", lenses)

    assert result.epistemic_confidence > 0.5
    assert len(result.reasoning_chain) >= 4
    assert result.violated_constraints == []
    assert set(result.lens_contributions) == {"MockLens1", "MockLens2"}
    assert abs(sum(result.lens_contributions.values()) - 1.0) < 1e-9
    assert result.generated_designs
    assert result.generated_designs[0].composite_score >= 0.0
    assert result.generated_designs[0].artifacts
    assert result.execution_result is not None
    assert result.execution_result.cycles_used >= 1
    assert result.execution_result.final_output.startswith("{")
    assert "lead design" in result.conclusion.lower()


def test_kernel_parses_live_json_and_falls_back_if_needed():
    kernel = ReasoningKernel(api_key="live-key")

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        text=(
                            '{"conclusion": "Parsed result", '
                            '"reasoning_chain": [{"step_id": "1", "description": "step"}], '
                            '"violated_constraints": [], '
                            '"epistemic_confidence": 0.7, '
                            '"lens_contributions": {"MockLens1": 1.0}}'
                        )
                    )
                ]
            )

    kernel.client = SimpleNamespace(messages=FakeMessages())
    lenses = [CognitiveLens("MockLens1", "framing1", ["constraint1"], ["spot1"], 0.8, "deductive")]

    result = kernel.synthesize("Analyze this system", lenses)

    assert result.execution_result is not None
    assert result.execution_result.converged is True
    assert "Parsed result" not in result.conclusion
    assert "dominant framing" in result.conclusion.lower()
    assert len(result.reasoning_chain) >= 2
    assert result.epistemic_confidence >= 0.7
    assert result.generated_designs


def test_kernel_recombines_prior_design_context():
    kernel = ReasoningKernel(api_key="dummy_key_for_testing", execution_mode="local-only")
    lenses = [CognitiveLens("MockLens1", "framing1", ["constraint1"], ["spot1"], 0.8, "deductive")]

    result = kernel.synthesize(
        "Analyze this system",
        lenses,
        design_context=[
            {
                "titles": ["Constraint Lattice Compiler"],
                "primitives": ["Rewire the prior invariant graph"],
                "confidence": 0.9,
            }
        ],
    )

    assert any("Lineage Mutation Reactor" == design.title for design in result.generated_designs)
    assert any(design.lineage_titles for design in result.generated_designs)


def test_kernel_runs_exact_topology_search_for_structured_query():
    kernel = ReasoningKernel(api_key="dummy_key_for_testing", execution_mode="local-only")
    lenses = [
        CognitiveLens("MockLens1", "framing1", ["constraint1"], ["spot1"], 0.8, "deductive"),
        CognitiveLens("MockLens2", "framing2", ["constraint2"], ["spot2"], 0.7, "physical"),
        CognitiveLens("MockLens3", "framing3", ["constraint3"], ["spot3"], 0.75, "quantum"),
    ]

    result = kernel.synthesize(EXACT_QUERY, lenses)

    assert result.topology_search is not None
    assert result.execution_result is not None
    assert result.execution_result.cycles_used >= 2
    assert result.topology_search.evaluated_topologies > 0
    assert len(result.topology_search.satisfiable_topologies) == 3
    assert result.generated_designs
    assert result.generated_designs[0].title.endswith("T001")
    assert any(artifact.language == "json" for artifact in result.generated_designs[0].artifacts)
    assert any(artifact.artifact_type == "quantum_circuit" for artifact in result.generated_designs[0].artifacts)
    assert any(artifact.artifact_type == "execution_verification" for artifact in result.generated_designs[0].artifacts)
    assert "Exact exhaustive search found 3 satisfiable topology configuration(s)" in result.conclusion


def test_kernel_remote_only_requires_live_credentials():
    kernel = ReasoningKernel(api_key="dummy_key_for_testing", execution_mode="remote-only")
    lenses = [CognitiveLens("MockLens1", "framing1", ["constraint1"], ["spot1"], 0.8, "deductive")]

    with pytest.raises(RuntimeError):
        kernel.synthesize("Analyze this system", lenses)
