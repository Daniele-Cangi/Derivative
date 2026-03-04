from lenses.base import BaseLens
try:
    import sympy  # noqa: F401
except ImportError:
    sympy = None

class SymbolicLens(BaseLens):
    epistemic_tag = "symbolic"
    lens_name = "Symbolic Logic"
    library_focus = "SymPy"
    analysis_focus = "definitions, invariants, and algebraic consistency"
    keywords = (
        "equation",
        "logic",
        "formula",
        "invariant",
        "constraint",
        "proof",
        "derive",
        "symbolic",
    )
    default_constraints = (
        "Avoid conclusions that violate stated definitions.",
        "Track invariants before optimizing implementation details.",
        "Reduce ambiguous wording into explicit logical claims.",
    )
    default_blind_spots = (
        "Can underweight messy operational realities or human factors.",
        "May treat approximate behavior too rigidly when uncertainty dominates.",
    )

    def _collect_library_signals(self, problem: str):
        if sympy is None:
            return [], [], [], 0.0

        tokens = []
        for token in self._problem_tokens(problem):
            if token.isidentifier() and token not in tokens:
                tokens.append(token)
            if len(tokens) >= 4:
                break

        if not tokens:
            return [], [], [], 0.0

        symbols = sympy.symbols(" ".join(tokens))
        if not isinstance(symbols, tuple):
            symbols = (symbols,)

        operator_count = sum(problem.count(operator) for operator in ("=", "<", ">", "+", "-", "*", "/"))
        notes = [f"SymPy normalized {len(symbols)} symbolic variable(s) from the prompt."]
        if operator_count:
            notes.append(f"Detected {operator_count} explicit operator token(s) that can anchor invariants.")

        constraints = ["Maintain explicit invariants for the extracted symbolic variables."]
        blind_spots = []
        if operator_count == 0:
            blind_spots.append("The prompt did not expose concrete expressions, so symbolic reduction stays coarse.")

        return notes, constraints, blind_spots, 0.05

    def _build_operator_primitives(self, problem: str) -> list[str]:
        primitives = [
            "Lift the problem into explicit variables, invariants, and transformations before implementation.",
            "Rewrite ambiguous requirements into symbolic constraints that can be recombined safely.",
        ]
        if any(operator in problem for operator in ("=", "+", "-", "*", "/")):
            primitives.append("Exploit the detected operator structure to synthesize equivalent but more stable formulations.")
        return primitives

    def _build_design_affordances(self, problem: str) -> list[str]:
        return [
            "Use symbolic rewriting to search for architecture variants that preserve invariants while changing execution shape.",
            "Treat formulas and constraints as editable program matter, not just descriptive metadata.",
        ]
