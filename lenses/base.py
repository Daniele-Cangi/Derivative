import os
import re
from dataclasses import dataclass, field
from typing import List, Optional
import anthropic

from core.json_utils import clamp_float, ensure_string_list, extract_json_object
from core.runtime_mode import normalize_execution_mode

@dataclass
class CognitiveLens:
    lens_name: str
    framing: str                    # How this lens sees the problem
    constraints: List[str]          # What this lens forbids / makes impossible
    blind_spots: List[str]          # What this lens cannot see
    confidence: float               # 0.0 - 1.0
    epistemic_tag: str              # "deductive" | "probabilistic" | "symbolic" | "quantum" | "physical" | "causal" | "formal"
    operator_primitives: List[str] = field(default_factory=list)
    design_affordances: List[str] = field(default_factory=list)

class BaseLens:
    """Base class for all cognitive lenses."""
    epistemic_tag = "unknown"
    lens_name = "BaseLens"
    library_focus = "None"
    analysis_focus = "general system behavior"
    keywords: tuple[str, ...] = ()
    default_constraints: tuple[str, ...] = ()
    default_blind_spots: tuple[str, ...] = ()
    
    def __init__(self, api_key: Optional[str] = None, execution_mode: Optional[str] = None):
        resolved_key = (api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
        self.api_key = resolved_key
        self.execution_mode = normalize_execution_mode(execution_mode)
        self.allow_local_fallback = self.execution_mode != "remote-only"
        self.use_live_model = (
            self.execution_mode in {"hybrid", "remote-only"}
            and bool(resolved_key)
            and resolved_key != "dummy_key_for_testing"
        )
        self.client = anthropic.Anthropic(api_key=resolved_key) if self.use_live_model else None

    def frame(self, problem: str) -> CognitiveLens:
        local_lens = self._frame_locally(problem)
        if self.execution_mode == "local-only":
            return local_lens
        if not self.use_live_model:
            if self.allow_local_fallback:
                return local_lens
            raise RuntimeError(f"{self.lens_name} requires remote mode with a valid ANTHROPIC_API_KEY.")

        system_prompt = f"""You are a cognitive lens for an AI agent. 
Your domain is {self.lens_name}, heavily influenced by the principles of the {self.library_focus} library.
Your epistemic tag is: {self.epistemic_tag}.

Analyze the user's problem uniquely from your domain's perspective. Do not solve the problem directly.
Extract the framing, constraints, and blind spots of the problem from your perspective.
Return exactly one JSON object using this schema:
{{
    "framing": "How this lens sees the problem (string)",
    "constraints": ["constraint 1", "constraint 2"],
    "blind_spots": ["blind spot 1"],
    "confidence": 0.85
}}"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Problem: {problem}"}
                ]
            )
            data = extract_json_object(response.content[0].text)

            return CognitiveLens(
                lens_name=self.lens_name,
                framing=str(data.get("framing") or local_lens.framing).strip(),
                constraints=ensure_string_list(data.get("constraints")) or local_lens.constraints,
                blind_spots=ensure_string_list(data.get("blind_spots")) or local_lens.blind_spots,
                confidence=clamp_float(
                    data.get("confidence"),
                    default=local_lens.confidence,
                    minimum=0.0,
                    maximum=1.0,
                ),
                epistemic_tag=self.epistemic_tag,
                operator_primitives=list(local_lens.operator_primitives),
                design_affordances=list(local_lens.design_affordances),
            )
        except Exception:
            if self.allow_local_fallback:
                return local_lens
            raise
    
    def is_applicable(self, problem: str) -> bool:
        """Returns True if this lens is relevant for the given problem."""
        return self._relevance_score(problem) >= 0.2

    def _frame_locally(self, problem: str) -> CognitiveLens:
        normalized_problem = " ".join(problem.split())
        problem_summary = normalized_problem[:160]
        if len(normalized_problem) > 160:
            problem_summary += "..."

        matched_keywords = self._matched_keywords(problem)
        library_notes, extra_constraints, extra_blind_spots, confidence_bonus = self._collect_library_signals(
            problem
        )
        operator_primitives = self._build_operator_primitives(problem)
        design_affordances = self._build_design_affordances(problem)
        if matched_keywords:
            signal_text = f"Detected domain signals: {', '.join(matched_keywords)}."
        else:
            signal_text = (
                "No explicit domain keywords were present, so this lens is used as a structural cross-check."
            )
        if library_notes:
            signal_text += f" Library-backed checks: {' '.join(library_notes)}"
        if operator_primitives:
            signal_text += f" Operator primitives: {'; '.join(operator_primitives[:2])}."

        constraints = self._unique_items(list(self.default_constraints) + list(extra_constraints)) or [
            f"Do not ignore {self.analysis_focus} when forming conclusions."
        ]
        blind_spots = self._unique_items(list(self.default_blind_spots) + list(extra_blind_spots)) or [
            "Context outside this analytical lens may be underweighted."
        ]

        confidence = clamp_float(
            0.55
            + (0.08 * len(matched_keywords))
            + (0.05 if len(normalized_problem) > 80 else 0.0)
            + confidence_bonus,
            default=0.55,
            minimum=0.35,
            maximum=0.92,
        )
        framing = (
            f"{self.lens_name} frames the problem around {self.analysis_focus}. "
            f"{signal_text} Working summary: {problem_summary}"
        )

        return CognitiveLens(
            lens_name=self.lens_name,
            framing=framing,
            constraints=constraints,
            blind_spots=blind_spots,
            confidence=confidence,
            epistemic_tag=self.epistemic_tag,
            operator_primitives=operator_primitives,
            design_affordances=design_affordances,
        )

    def _relevance_score(self, problem: str) -> float:
        normalized_problem = problem.lower()
        score = 0.15
        if len(problem.split()) >= 5:
            score += 0.1

        general_tokens = (
            "why",
            "how",
            "risk",
            "failure",
            "system",
            "reason",
            "analyze",
            "validate",
            "architecture",
            "code",
            "problem",
        )
        if any(token in normalized_problem for token in general_tokens):
            score += 0.2

        score += min(0.5, 0.15 * len(self._matched_keywords(problem)))
        return clamp_float(score, default=0.15)

    def _matched_keywords(self, problem: str) -> list[str]:
        normalized_problem = problem.lower()
        return [keyword for keyword in self.keywords if keyword.lower() in normalized_problem][:3]

    def _collect_library_signals(
        self,
        problem: str,
    ) -> tuple[list[str], list[str], list[str], float]:
        return [], [], [], 0.0

    def _build_operator_primitives(self, problem: str) -> list[str]:
        matched_keywords = self._matched_keywords(problem)
        if matched_keywords:
            return [
                f"Transform the problem by explicitly manipulating {keyword} as a {self.lens_name} control surface."
                for keyword in matched_keywords[:2]
            ]
        return [
            f"Project the problem into {self.analysis_focus} and iterate on the resulting design variables."
        ]

    def _build_design_affordances(self, problem: str) -> list[str]:
        matched_keywords = self._matched_keywords(problem)
        if matched_keywords:
            return [
                f"Use {self.library_focus} concepts to recombine {', '.join(matched_keywords[:2])} into a new mechanism."
            ]
        return [
            f"Use {self.library_focus} as a computational substrate for {self.analysis_focus} exploration."
        ]

    def _problem_tokens(self, problem: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", problem.lower())
        stop_words = {
            "the",
            "and",
            "that",
            "with",
            "this",
            "from",
            "what",
            "when",
            "where",
            "which",
            "into",
            "your",
            "have",
            "will",
            "should",
            "would",
            "there",
            "about",
            "code",
            "system",
            "problem",
        }
        return [token for token in tokens if token not in stop_words]

    def _unique_items(self, items: list[str]) -> list[str]:
        deduplicated: list[str] = []
        for item in items:
            value = str(item).strip()
            if value and value not in deduplicated:
                deduplicated.append(value)
        return deduplicated
