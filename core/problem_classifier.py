from dataclasses import dataclass
from enum import Enum
import re


class ProblemType(Enum):
    NUMERIC = "numeric"
    PROBABILISTIC = "probabilistic"
    STRUCTURAL = "structural"
    COMBINATORIAL = "combinatorial"
    SYMBOLIC = "symbolic"
    HYBRID = "hybrid"


@dataclass
class ProblemClassification:
    primary_type: ProblemType
    secondary_types: list[ProblemType]
    requires_numeric_answer: bool
    numeric_keywords: list[str]
    recommended_lens: str


class ProblemClassifier:
    NUMERIC_TRIGGERS = [
        "find the minimum",
        "find the maximum",
        "find the optimal",
        "what is the minimum",
        "what is the maximum",
        "how many",
        "how much",
        "calculate",
        "compute",
        "minimum number",
        "maximum number",
        "exact number",
        "probability of",
        "probability that",
        "expected value",
        "at least",
        "at most",
        "no more than",
        "no less than",
        "falls below",
        "exceeds",
        "threshold",
        "bound",
        "minimum rounds",
        "minimum hops",
        "minimum steps",
        "verify that",
        "show that",
        "prove that",
    ]

    STRUCTURAL_TRIGGERS = [
        "topology",
        "architecture",
        "design",
        "configure",
        "network",
        "graph",
        "nodes",
        "edges",
        "connections",
        "distribute",
        "schedule",
        "assign",
        "allocate",
        "ring topology",
        "topological configuration",
        "topological configurations",
        "minimum spanning tree",
    ]

    PROBABILISTIC_TRIGGERS = [
        "probability",
        "chance",
        "likelihood",
        "p=",
        "error rate",
        "failure rate",
        "interception",
        "noise",
        "fidelity",
        "independent",
        "correlated",
        "distribution",
        "stochastic",
    ]

    NUMERIC_PATTERNS = (
        (re.compile(r"\b(?:minimum|maximum)\s+(?:number|rounds|hops|steps)\b"), "derived-numeric-target"),
        (re.compile(r"[<>]=?\s*\d"), "numeric-comparison"),
        (re.compile(r"\bfalls below\s+[0-9]*\.?[0-9]+"), "falls-below-threshold"),
        (re.compile(r"\bp\s*=\s*[0-9]*\.?[0-9]+"), "explicit-parameter"),
    )

    def classify(self, problem: str) -> ProblemClassification:
        problem_lower = problem.lower()

        triggered_keywords = [
            keyword
            for keyword in self.NUMERIC_TRIGGERS
            if self._contains_trigger(keyword, problem_lower)
        ]
        for pattern, label in self.NUMERIC_PATTERNS:
            if pattern.search(problem_lower) and label not in triggered_keywords:
                triggered_keywords.append(label)

        requires_numeric = len(triggered_keywords) > 0
        structural_hits = [
            trigger
            for trigger in self.STRUCTURAL_TRIGGERS
            if self._contains_trigger(trigger, problem_lower)
        ]
        structural_hits = self._collapse_overlapping_hits(structural_hits)
        probabilistic_hits = [
            trigger
            for trigger in self.PROBABILISTIC_TRIGGERS
            if self._contains_trigger(trigger, problem_lower)
        ]

        has_structural = bool(structural_hits)
        has_probabilistic = bool(probabilistic_hits)

        if has_probabilistic and requires_numeric and len(structural_hits) <= 1:
            primary = ProblemType.PROBABILISTIC
            recommended_lens = "probabilistic"
        elif has_structural and requires_numeric and len(structural_hits) >= 2:
            primary = ProblemType.HYBRID
            recommended_lens = "topological"
        elif requires_numeric and not has_structural:
            primary = ProblemType.NUMERIC
            recommended_lens = "symbolic"
        elif has_structural and not requires_numeric:
            primary = ProblemType.STRUCTURAL
            recommended_lens = "topological"
        elif has_probabilistic:
            primary = ProblemType.PROBABILISTIC
            recommended_lens = "probabilistic"
        else:
            primary = ProblemType.STRUCTURAL
            recommended_lens = "topological"

        secondary: list[ProblemType] = []
        if has_structural and primary != ProblemType.STRUCTURAL:
            secondary.append(ProblemType.STRUCTURAL)
        if has_probabilistic and primary != ProblemType.PROBABILISTIC:
            secondary.append(ProblemType.PROBABILISTIC)

        return ProblemClassification(
            primary_type=primary,
            secondary_types=secondary,
            requires_numeric_answer=requires_numeric,
            numeric_keywords=triggered_keywords,
            recommended_lens=recommended_lens,
        )

    def _collapse_overlapping_hits(self, hits: list[str]) -> list[str]:
        collapsed: list[str] = []
        for hit in sorted(hits, key=len, reverse=True):
            if any(hit in existing for existing in collapsed):
                continue
            collapsed.append(hit)
        return collapsed

    def _contains_trigger(self, trigger: str, problem_lower: str) -> bool:
        pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(trigger)}(?![a-z0-9_])")
        return bool(pattern.search(problem_lower))
