from lenses.base import BaseLens
try:
    from qiskit import QuantumCircuit
    QUANTUM_IMPORT_ERROR = ""
except ImportError as exc:
    QuantumCircuit = None
    QUANTUM_IMPORT_ERROR = str(exc)

class QuantumLens(BaseLens):
    epistemic_tag = "quantum"
    lens_name = "Quantum Logic"
    library_focus = "Qiskit"
    analysis_focus = "state ambiguity, branching alternatives, and non-classical interactions"
    keywords = (
        "quantum",
        "superposition",
        "state",
        "branch",
        "uncertain",
        "entangled",
        "parallel",
        "alternative",
    )
    default_constraints = (
        "Track competing states before collapsing to a single explanation.",
        "Do not assume one linear path when the problem clearly branches.",
        "Separate observation from the underlying state when evidence is partial.",
    )
    default_blind_spots = (
        "Mostly a metaphorical stress test outside true quantum domains.",
        "Can overcomplicate simple deterministic problems.",
    )

    def _collect_library_signals(self, problem: str):
        if QuantumCircuit is None:
            notes = ["Qiskit runtime unavailable; the quantum layer is operating in degraded design-only mode."]
            blind_spots = []
            if QUANTUM_IMPORT_ERROR:
                blind_spots.append(f"Quantum runtime import error: {QUANTUM_IMPORT_ERROR}")
            return notes, [], blind_spots, 0.0

        width = 2 if len(problem) < 220 else 3
        circuit = QuantumCircuit(width)
        circuit.h(0)
        if width > 1:
            circuit.cx(0, 1)
        if width > 2:
            circuit.cx(1, 2)

        ops = circuit.count_ops()
        notes = [
            f"Qiskit constructed a {width}-qubit branching scaffold with depth {circuit.depth()} and ops {dict(ops)}."
        ]
        constraints = ["Keep multiple candidate states alive until cross-lens collapse is justified."]
        blind_spots = []
        return notes, constraints, blind_spots, 0.08

    def _build_operator_primitives(self, problem: str) -> list[str]:
        primitives = [
            "Spawn parallel design states before collapsing on a single implementation path.",
            "Entangle two subsystems so a change in one propagates as a deliberate design signal.",
        ]
        if "parallel" in problem.lower() or "distributed" in problem.lower():
            primitives.append("Route concurrent branches through controlled synchronization instead of early serialization.")
        return primitives

    def _build_design_affordances(self, problem: str) -> list[str]:
        return [
            "Use Qiskit-inspired branching to evaluate mutually incompatible architectures before committing resources.",
            "Model candidate designs as state vectors whose collapse is delayed until formal and physical checks complete.",
        ]

    def runtime_status(self) -> str:
        if QuantumCircuit is not None:
            return "ready"
        if QUANTUM_IMPORT_ERROR:
            return "degraded"
        return "unavailable"

    def runtime_detail(self) -> str:
        if QuantumCircuit is not None:
            return "QuantumCircuit import ok"
        return QUANTUM_IMPORT_ERROR or "QuantumCircuit unavailable"
