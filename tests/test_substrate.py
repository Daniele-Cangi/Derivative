import threading

from core.substrate import CognitiveSubstrate
from lenses.base import BaseLens
from lenses.formal import FormalLens
from lenses.symbolic import SymbolicLens


class BrokenLens(BaseLens):
    lens_name = "Broken Lens"
    epistemic_tag = "deductive"

    def frame(self, problem: str):
        raise RuntimeError("boom")


class ThreadBoundLens(BaseLens):
    lens_name = "Thread Bound Lens"
    epistemic_tag = "physical"
    parallel_safe = False

    def __init__(self):
        super().__init__(api_key="dummy_key_for_testing", execution_mode="local-only")
        self.thread_id = None

    def frame(self, problem: str):
        self.thread_id = threading.get_ident()
        return super().frame(problem)


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


def test_substrate_runs_non_parallel_safe_lens_on_calling_thread():
    substrate = CognitiveSubstrate()
    thread_bound_lens = ThreadBoundLens()
    substrate.lenses = [
        thread_bound_lens,
        SymbolicLens(api_key="dummy_key_for_testing", execution_mode="local-only"),
        FormalLens(api_key="dummy_key_for_testing", execution_mode="local-only"),
    ]
    calling_thread_id = threading.get_ident()

    substrate.decompose("Analyze a quantum system architecture")

    assert thread_bound_lens.thread_id == calling_thread_id
