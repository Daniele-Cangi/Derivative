from lenses.base import BaseLens
try:
    import dowhy  # noqa: F401
except ImportError:
    dowhy = None

class CausalLens(BaseLens):
    epistemic_tag = "causal"
    lens_name = "Causal Inference"
    library_focus = "DoWhy"
    analysis_focus = "cause-and-effect structure, interventions, and upstream drivers"
    keywords = (
        "cause",
        "effect",
        "impact",
        "driver",
        "root cause",
        "dependency",
        "intervention",
        "feedback",
    )
    default_constraints = (
        "Do not confuse correlation with causation.",
        "Separate observed symptoms from upstream drivers.",
        "Check whether interventions actually change the outcome.",
    )
    default_blind_spots = (
        "Weak on exact numerical bounds without supporting measurements.",
        "May miss symbolic contradictions when causal structure is underspecified.",
    )

    def _build_operator_primitives(self, problem: str) -> list[str]:
        primitives = [
            "Separate upstream drivers from observed symptoms before changing the design.",
            "Design an intervention point that can falsify the suspected root cause.",
        ]
        if "feedback" in problem.lower() or "loop" in problem.lower():
            primitives.append("Break reinforcing loops with a controllable damping mechanism.")
        return primitives

    def _build_design_affordances(self, problem: str) -> list[str]:
        return [
            "Use causal intervention mapping to create mechanisms that change outcomes instead of only describing them.",
            "Treat each major subsystem as a node in an interventional graph and optimize intervention leverage.",
        ]
