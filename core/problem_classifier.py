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
    explicit_objectives: list[str]


class ProblemClassifier:
    DIRECTIVE_PATTERNS = (
        re.compile(
            r"\b(find|determine|prove|verify|show|compute|calculate|derive)\b([^.!?\n]*)",
            re.IGNORECASE,
        ),
    )

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
        "total count",
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

    SYMBOLIC_TRIGGERS = [
        "recurrence",
        "recursive",
        "closed form",
        "explicit formula",
        "characteristic polynomial",
        "companion matrix",
        "eigenvalue",
        "eigenvalues",
        "sympy",
        "algebraic system",
        "system of equations",
    ]

    NUMERIC_PATTERNS = (
        (re.compile(r"\b(?:minimum|maximum)\s+(?:number|rounds|hops|steps)\b"), "derived-numeric-target"),
        (re.compile(r"[<>]=?\s*\d"), "numeric-comparison"),
        (re.compile(r"\bfalls below\s+[0-9]*\.?[0-9]+"), "falls-below-threshold"),
        (re.compile(r"\bp\s*=\s*[0-9]*\.?[0-9]+"), "explicit-parameter"),
    )

    SYMBOLIC_PATTERNS = (
        (
            re.compile(
                r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*n\s*\)\s*=\s*.*\1\s*\(\s*n\s*-\s*1\s*\)",
                re.IGNORECASE,
            ),
            "linear-recurrence",
        ),
        (re.compile(r"\bf\(\s*n\s*\+\s*1\s*\)\s*/\s*f\(\s*n\s*\)", re.IGNORECASE), "ratio-limit"),
        (re.compile(r"\bsolve\b[^.!?\n]*\b(?:system|equations?)\b", re.IGNORECASE), "algebraic-system"),
    )

    def classify(self, problem: str) -> ProblemClassification:
        problem_lower = problem.lower()
        objectives = self.extract_explicit_objectives(problem)

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
        symbolic_hits = [
            trigger
            for trigger in self.SYMBOLIC_TRIGGERS
            if self._contains_trigger(trigger, problem_lower)
        ]
        for pattern, label in self.SYMBOLIC_PATTERNS:
            if pattern.search(problem) and label not in symbolic_hits:
                symbolic_hits.append(label)

        has_structural = bool(structural_hits)
        has_probabilistic = bool(probabilistic_hits)
        has_symbolic = bool(symbolic_hits)
        has_linear_recurrence = "linear-recurrence" in symbolic_hits

        if has_symbolic and (has_linear_recurrence or "recurrence" in problem_lower or "recursive" in problem_lower):
            primary = ProblemType.SYMBOLIC
            recommended_lens = "symbolic"
        elif has_symbolic and not has_structural:
            primary = ProblemType.SYMBOLIC
            recommended_lens = "symbolic"
        elif has_probabilistic and requires_numeric and len(structural_hits) <= 1:
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
        if has_symbolic and primary != ProblemType.SYMBOLIC:
            secondary.append(ProblemType.SYMBOLIC)

        return ProblemClassification(
            primary_type=primary,
            secondary_types=secondary,
            requires_numeric_answer=requires_numeric,
            numeric_keywords=triggered_keywords,
            recommended_lens=recommended_lens,
            explicit_objectives=objectives,
        )

    def extract_explicit_objectives(self, problem: str) -> list[str]:
        objectives: list[str] = []
        seen: set[str] = set()
        for raw_line in problem.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            numbered = re.match(r"^\d+[\).\:]?\s*(.+)$", line)
            candidate = numbered.group(1).strip() if numbered else ""
            if candidate:
                normalized = self._normalize_objective(candidate)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    objectives.append(candidate)

        for pattern in self.DIRECTIVE_PATTERNS:
            for match in pattern.finditer(problem):
                clause = f"{match.group(1)} {match.group(2)}".strip(" .,:;\n\t")
                for segment in self._split_compound_objective(clause):
                    normalized = self._normalize_objective(segment)
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        objectives.append(segment)

        return objectives

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

    def _normalize_objective(self, objective: str) -> str:
        return " ".join(objective.lower().split())

    def _split_compound_objective(self, objective: str) -> list[str]:
        segments = re.split(
            r"\s*(?:,|\band\b|\bthen\b)\s+(?=(?:find|determine|prove|verify|show|compute|calculate|derive)\b)",
            objective,
            flags=re.IGNORECASE,
        )
        return [segment.strip(" .,:;\n\t") for segment in segments if segment.strip()]
