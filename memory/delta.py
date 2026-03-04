import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

from core.kernel import ReasoningResult


@dataclass
class MemoryDelta:
    timestamp: str
    problem_hash: str
    lens_contributions: Dict[str, float]
    reasoning_delta: str
    confidence_delta: float
    confidence_score: float = 0.0
    conclusion_snapshot: str = ""
    top_design_titles: List[str] = dataclasses.field(default_factory=list)
    top_design_primitives: List[str] = dataclasses.field(default_factory=list)
    execution_cycle_summaries: List[str] = dataclasses.field(default_factory=list)
    verified_hypotheses: List[str] = dataclasses.field(default_factory=list)


class DeltaMemory:
    def __init__(self, storage_file: str = "memory_deltas.json"):
        self.storage_file = storage_file
        self.history: List[MemoryDelta] = self._load()

    def _load(self) -> List[MemoryDelta]:
        try:
            with open(self.storage_file, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        history: List[MemoryDelta] = []
        if not isinstance(data, list):
            return history

        for item in data:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized.setdefault("confidence_score", normalized.get("confidence_delta", 0.0))
            normalized.setdefault("conclusion_snapshot", normalized.get("reasoning_delta", ""))
            normalized.setdefault("top_design_titles", [])
            normalized.setdefault("top_design_primitives", [])
            normalized.setdefault("execution_cycle_summaries", [])
            normalized.setdefault("verified_hypotheses", [])
            try:
                history.append(MemoryDelta(**normalized))
            except TypeError:
                continue
        return history

    def _save(self):
        with open(self.storage_file, "w") as f:
            json.dump([dataclasses.asdict(m) for m in self.history], f, indent=2)

    def record(self, result: ReasoningResult, problem: str) -> MemoryDelta:
        problem_hash = hashlib.sha256(problem.encode()).hexdigest()
        past_entries = [m for m in self.history if m.problem_hash == problem_hash]

        last_confidence = past_entries[-1].confidence_score if past_entries else 0.0
        last_conclusion = past_entries[-1].conclusion_snapshot if past_entries else ""
        reasoning_delta = self._build_delta_summary(
            previous_conclusion=last_conclusion,
            current_conclusion=result.conclusion,
            previous_confidence=last_confidence,
            current_confidence=result.epistemic_confidence,
        )

        delta = MemoryDelta(
            timestamp=datetime.now(timezone.utc).isoformat(),
            problem_hash=problem_hash,
            lens_contributions=result.lens_contributions,
            reasoning_delta=reasoning_delta,
            confidence_delta=result.epistemic_confidence - last_confidence,
            confidence_score=result.epistemic_confidence,
            conclusion_snapshot=result.conclusion,
            top_design_titles=[design.title for design in result.generated_designs[:3]],
            top_design_primitives=self._collect_design_primitives(result),
            execution_cycle_summaries=self._collect_execution_summaries(result),
            verified_hypotheses=self._collect_verified_hypotheses(result),
        )

        self.history.append(delta)
        self._save()
        return delta

    def retrieve_relevant(self, problem: str, top_k: int = 3) -> List[MemoryDelta]:
        problem_hash = hashlib.sha256(problem.encode()).hexdigest()
        relevant = [m for m in self.history if m.problem_hash == problem_hash]
        return relevant[-top_k:]

    def get_reasoning_history(self) -> List[MemoryDelta]:
        return self.history

    def retrieve_design_context(self, problem: str, top_k: int = 3) -> List[Dict[str, object]]:
        relevant = self.retrieve_relevant(problem, top_k=top_k)
        context: List[Dict[str, object]] = []
        for delta in reversed(relevant):
            if not delta.top_design_titles and not delta.top_design_primitives:
                continue
            context.append(
                {
                    "titles": list(delta.top_design_titles),
                    "primitives": list(delta.top_design_primitives),
                    "confidence": delta.confidence_score,
                    "timestamp": delta.timestamp,
                }
            )
        return context

    def _build_delta_summary(
        self,
        previous_conclusion: str,
        current_conclusion: str,
        previous_confidence: float,
        current_confidence: float,
    ) -> str:
        if not previous_conclusion:
            return "Initial reasoning snapshot recorded."

        normalized_previous = previous_conclusion.strip()
        normalized_current = current_conclusion.strip()
        if normalized_previous == normalized_current:
            if current_confidence == previous_confidence:
                return "No material change from previous reasoning."
            direction = "up" if current_confidence > previous_confidence else "down"
            return f"Confidence moved {direction} while the conclusion remained stable."

        return "Conclusion changed from the previous reasoning snapshot; review the latest synthesis."

    def _collect_design_primitives(self, result: ReasoningResult) -> List[str]:
        primitives: List[str] = []
        for design in result.generated_designs[:2]:
            for primitive in design.component_primitives[:2]:
                if primitive not in primitives:
                    primitives.append(primitive)
        return primitives

    def _collect_execution_summaries(self, result: ReasoningResult) -> List[str]:
        if result.execution_result is None:
            return []
        return [
            f"cycle {cycle.cycle}: delta {cycle.delta:.2f} :: {'converged' if cycle.converged else 'revised'}"
            for cycle in result.execution_result.history
        ]

    def _collect_verified_hypotheses(self, result: ReasoningResult) -> List[str]:
        if result.execution_result is None or not result.execution_result.converged:
            return []
        deduplicated: List[str] = []
        for cycle in result.execution_result.history:
            if cycle.converged and cycle.hypothesis not in deduplicated:
                deduplicated.append(cycle.hypothesis)
        return deduplicated
