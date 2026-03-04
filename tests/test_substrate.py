from core.substrate import CognitiveSubstrate
from lenses.base import BaseLens
from lenses.formal import FormalLens
from lenses.symbolic import SymbolicLens


class BrokenLens(BaseLens):
    lens_name = "Broken Lens"
    epistemic_tag = "deductive"

    def frame(self, problem: str):
        raise RuntimeError("boom")


def test_substrate_loads_lenses():
    substrate = CognitiveSubstrate()
    assert len(substrate.lenses) == 7


def test_substrate_decompose_returns_ranked_framings():
    substrate = CognitiveSubstrate()
    framings = substrate.decompose("Analyze the failure modes of a distributed system")

    assert len(framings) >= 2
    assert framings[0].confidence >= framings[-1].confidence
    for framing in framings:
        assert framing.confidence > 0.0
        assert framing.epistemic_tag != "unknown"


def test_substrate_recovers_from_failing_lens():
    substrate = CognitiveSubstrate()
    substrate.lenses = [
        BrokenLens(api_key="dummy_key_for_testing"),
        SymbolicLens(api_key="dummy_key_for_testing"),
        FormalLens(api_key="dummy_key_for_testing"),
    ]

    framings = substrate.decompose("Test problem")

    assert len(framings) >= 2
    assert substrate.last_errors
