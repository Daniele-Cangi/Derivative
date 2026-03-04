from lenses.base import BaseLens
try:
    import pgmpy  # noqa: F401
except ImportError:
    pgmpy = None

class ProbabilisticLens(BaseLens):
    epistemic_tag = "probabilistic"
    lens_name = "Bayesian Probabilistic"
    library_focus = "pgmpy"
    analysis_focus = "uncertainty, priors, likelihoods, and confidence calibration"
    keywords = (
        "probability",
        "likely",
        "uncertain",
        "risk",
        "confidence",
        "estimate",
        "prior",
        "bayes",
    )
    default_constraints = (
        "Do not present uncertain claims as deterministic facts.",
        "Separate priors from observed evidence.",
        "Update confidence when new evidence changes the likelihood.",
    )
    default_blind_spots = (
        "May underweight hard impossibility constraints.",
        "Can be less useful when the main issue is definitional rather than statistical.",
    )

    def _build_operator_primitives(self, problem: str) -> list[str]:
        primitives = [
            "Maintain multiple weighted hypotheses instead of committing to a single deterministic forecast.",
            "Promote uncertainty into an explicit tuning surface for architecture selection.",
        ]
        if "risk" in problem.lower() or "failure" in problem.lower():
            primitives.append("Allocate redundancy in proportion to posterior failure mass.")
        return primitives

    def _build_design_affordances(self, problem: str) -> list[str]:
        return [
            "Use probability mass to rebalance effort across candidate architectures rather than treating all options equally.",
            "Encode uncertain modules as adaptive components whose operating mode shifts with observed evidence.",
        ]
