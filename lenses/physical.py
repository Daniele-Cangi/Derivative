from lenses.base import BaseLens
import re
try:
    import scipy  # noqa: F401
except ImportError:
    scipy = None

try:
    import pint  # noqa: F401
except ImportError:
    pint = None

class PhysicalLens(BaseLens):
    epistemic_tag = "physical"
    lens_name = "Physical Constraints"
    library_focus = "SciPy and Pint"
    analysis_focus = "resource limits, throughput, latency, and real-world bounds"
    keywords = (
        "latency",
        "memory",
        "cpu",
        "bandwidth",
        "throughput",
        "resource",
        "scale",
        "physical",
    )
    default_constraints = (
        "Do not propose solutions that ignore finite resources.",
        "Respect latency, throughput, and capacity ceilings.",
        "Separate idealized behavior from real operating conditions.",
    )
    default_blind_spots = (
        "May underweight logical elegance when resources are not the bottleneck.",
        "Can miss semantic correctness issues if the design is efficient but wrong.",
    )

    def _collect_library_signals(self, problem: str):
        if pint is None:
            return [], [], [], 0.0

        matches = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds|kb|mb|gb|tb|hz|mhz|ghz)", problem.lower())
        if not matches:
            return [], [], [], 0.0

        registry = pint.UnitRegistry()
        normalized_values = []
        normalized_units = []
        for value, unit in matches[:4]:
            try:
                if unit in {"ms"}:
                    quantity = float(value) * registry.millisecond
                    normalized_values.append(quantity.to(registry.second).magnitude)
                    normalized_units.append("seconds")
                elif unit in {"s", "sec", "seconds"}:
                    quantity = float(value) * registry.second
                    normalized_values.append(quantity.to(registry.second).magnitude)
                    normalized_units.append("seconds")
                elif unit in {"kb", "mb", "gb", "tb"}:
                    quantity = float(value) * getattr(registry, unit)
                    normalized_values.append(quantity.to(registry.byte).magnitude)
                    normalized_units.append("bytes")
                elif unit in {"hz", "mhz", "ghz"}:
                    quantity = float(value) * getattr(registry, unit)
                    normalized_values.append(quantity.to(registry.hertz).magnitude)
                    normalized_units.append("hertz")
            except Exception:
                continue

        if not normalized_values:
            return [], [], [], 0.0

        note = f"Pint normalized {len(normalized_values)} explicit resource quantity signal(s)."
        notes = [note]
        if scipy is not None and len(normalized_values) > 1:
            try:
                spread = float(scipy.stats.variation(normalized_values))
                notes.append(f"SciPy estimated a resource spread coefficient of {spread:.2f}.")
            except Exception:
                pass

        constraints = ["Respect the explicit quantitative resource bounds mentioned in the prompt."]
        blind_spots = []
        if len(set(normalized_units)) > 1:
            blind_spots.append("Mixed units can hide tradeoffs across latency, capacity, and frequency.")

        return notes, constraints, blind_spots, 0.05

    def _build_operator_primitives(self, problem: str) -> list[str]:
        primitives = [
            "Turn resource ceilings into shape constraints for the architecture instead of post-hoc limits.",
            "Co-design latency, throughput, and capacity so the mechanism fits the operating envelope.",
        ]
        if re.search(r"\d", problem):
            primitives.append("Use the explicit quantities in the prompt as hard anchors for the generated design.")
        return primitives

    def _build_design_affordances(self, problem: str) -> list[str]:
        return [
            "Generate architectures that are physically shaped by measured budgets, not idealized abstractions.",
            "Exploit unit-normalized resource signals to search for designs that remain stable under scale.",
        ]
