from lenses.base import BaseLens
try:
    import z3  # noqa: F401
except ImportError:
    z3 = None

class FormalLens(BaseLens):
    epistemic_tag = "formal"
    lens_name = "Formal Verification"
    library_focus = "Z3 Theorem Prover"
    analysis_focus = "invariants, satisfiability, and contradiction detection"
    keywords = (
        "proof",
        "verify",
        "guarantee",
        "invariant",
        "safety",
        "correctness",
        "assert",
        "formal",
    )
    default_constraints = (
        "Reject internally contradictory conclusions.",
        "Respect hard safety and correctness invariants.",
        "Treat unverifiable assumptions as risks, not facts.",
    )
    default_blind_spots = (
        "Can miss pragmatic tradeoffs when strict proof is unavailable.",
        "May underweight probabilistic evidence that still matters operationally.",
    )

    def _collect_library_signals(self, problem: str):
        if z3 is None:
            return [], [], [], 0.0

        normalized = problem.lower()
        contradiction_pairs = (
            ("always", "never"),
            ("must", "cannot"),
            ("required", "forbidden"),
            ("all", "none"),
        )

        conflicts = [pair for pair in contradiction_pairs if pair[0] in normalized and pair[1] in normalized]
        if conflicts:
            solver = z3.Solver()
            marker = z3.Bool("requirement_marker")
            solver.add(marker)
            solver.add(z3.Not(marker))
            solver.check()
            notes = [
                "Z3 detected an unsatisfiable requirement cue pair: "
                + ", ".join(f"{left}/{right}" for left, right in conflicts)
                + "."
            ]
            constraints = ["Resolve contradictory requirements before synthesis."]
            return notes, constraints, [], 0.07

        solver = z3.Solver()
        assumptions = [z3.Bool(f"constraint_{index}") for index in range(2)]
        solver.add(z3.And(*assumptions))
        status = solver.check()
        note = f"Z3 baseline satisfiability check returned {status} for the extracted requirement shell."
        return [note], ["Keep hard requirements mutually satisfiable."], [], 0.04

    def _build_operator_primitives(self, problem: str) -> list[str]:
        return [
            "Synthesize only architectures whose core invariants remain satisfiable under composition.",
            "Treat each bold design move as a candidate theorem obligation before adoption.",
        ]

    def _build_design_affordances(self, problem: str) -> list[str]:
        return [
            "Use satisfiability as a generator that prunes impossible inventions while preserving novel valid ones.",
            "Search for new mechanisms by solving for what must be true, then building directly against that model.",
        ]
