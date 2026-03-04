import pytest

from lenses.causal import CausalLens
from lenses.formal import FormalLens
from lenses.physical import PhysicalLens
from lenses.probabilistic import ProbabilisticLens
from lenses.quantum import QuantumLens
from lenses.symbolic import SymbolicLens
from lenses.topological import TopologicalLens


def test_all_lenses_produce_local_framings():
    lenses = [
        CausalLens(api_key="dummy_key_for_testing"),
        SymbolicLens(api_key="dummy_key_for_testing"),
        TopologicalLens(api_key="dummy_key_for_testing"),
        ProbabilisticLens(api_key="dummy_key_for_testing"),
        QuantumLens(api_key="dummy_key_for_testing"),
        FormalLens(api_key="dummy_key_for_testing"),
        PhysicalLens(api_key="dummy_key_for_testing"),
    ]

    for lens in lenses:
        assert lens.lens_name != "BaseLens"
        assert lens.epistemic_tag != "unknown"
        assert lens.is_applicable("Analyze the failure modes of a distributed system.") is True

        framing = lens.frame("Analyze the failure modes of a distributed system.")
        assert framing.lens_name == lens.lens_name
        assert framing.confidence > 0.0
        assert framing.constraints
        assert framing.blind_spots
        assert framing.operator_primitives
        assert framing.design_affordances
        assert lens.analysis_focus.lower() in framing.framing.lower()


def test_quantum_lens_exposes_runtime_status():
    lens = QuantumLens(api_key="dummy_key_for_testing")

    assert lens.runtime_status() in {"ready", "degraded", "unavailable"}
    assert isinstance(lens.runtime_detail(), str)


def test_lens_remote_only_requires_live_credentials():
    lens = SymbolicLens(api_key="dummy_key_for_testing", execution_mode="remote-only")

    with pytest.raises(RuntimeError):
        lens.frame("x = y + 1")
