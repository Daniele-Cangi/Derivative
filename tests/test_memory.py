import pytest

from core.kernel import ReasoningResult, ReasoningStep
from memory.delta import DeltaMemory


def test_memory_records_absolute_confidence(tmp_path):
    storage_file = tmp_path / "memory.json"
    memory = DeltaMemory(storage_file=str(storage_file))

    first = ReasoningResult(
        conclusion="First conclusion",
        reasoning_chain=[ReasoningStep("1", "step")],
        violated_constraints=[],
        epistemic_confidence=0.9,
        lens_contributions={"LensA": 1.0},
        generated_designs=[],
    )
    second = ReasoningResult(
        conclusion="Second conclusion",
        reasoning_chain=[ReasoningStep("1", "step")],
        violated_constraints=[],
        epistemic_confidence=0.7,
        lens_contributions={"LensA": 1.0},
        generated_designs=[],
    )

    delta1 = memory.record(first, "same problem")
    delta2 = memory.record(second, "same problem")

    assert delta1.confidence_score == 0.9
    assert delta1.confidence_delta == 0.9
    assert delta2.confidence_score == 0.7
    assert delta2.confidence_delta == pytest.approx(-0.2)
    assert "Conclusion changed" in delta2.reasoning_delta


def test_memory_returns_design_context(tmp_path):
    storage_file = tmp_path / "memory.json"
    memory = DeltaMemory(storage_file=str(storage_file))

    seeded = ReasoningResult(
        conclusion="Seeded conclusion",
        reasoning_chain=[ReasoningStep("1", "step")],
        violated_constraints=[],
        epistemic_confidence=0.8,
        lens_contributions={"LensA": 1.0},
        generated_designs=[],
    )
    memory.record(seeded, "same problem")
    memory.history[-1].top_design_titles = ["Quantum Branch Forge"]
    memory.history[-1].top_design_primitives = ["Spawn parallel design states"]

    context = memory.retrieve_design_context("same problem")

    assert context
    assert context[0]["titles"] == ["Quantum Branch Forge"]
    assert context[0]["primitives"] == ["Spawn parallel design states"]
