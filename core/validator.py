import dataclasses
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import anthropic

from core.json_utils import clamp_float, ensure_string_list, extract_json_object
from core.kernel import ReasoningResult
from core.runtime_mode import normalize_execution_mode


@dataclass
class ValidationReport:
    is_valid: bool
    attacks: List[str]
    edge_cases: List[str]
    confidence_adjusted: float
    recommendation: str


class AdversarialValidator:
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

    def validate(self, result: ReasoningResult, original_problem: str) -> ValidationReport:
        if result.violated_constraints:
            return ValidationReport(
                is_valid=False,
                attacks=["Hard constraint violation detected by kernel."],
                edge_cases=[],
                confidence_adjusted=0.0,
                recommendation="re-reason",
            )

        local_report = self._validate_locally(result, original_problem)
        if self.execution_mode == "local-only":
            return local_report
        if not self.use_live_model:
            if self.allow_local_fallback:
                return local_report
            raise RuntimeError("AdversarialValidator requires remote mode with a valid ANTHROPIC_API_KEY.")

        system_prompt = """You are the Adversarial Validator for the Derivative AI.
Your job is to relentlessly attack the proposed solution, find edge cases, and evaluate confidence.
Review the original problem and the reasoning result.
Produce a Validation Report in JSON matching this schema:
{
    "is_valid": true,
    "attacks": ["Attack on assumption X"],
    "edge_cases": ["Edge case Y where this fails"],
    "confidence_adjusted": 0.5,
    "recommendation": "accept"
}
Rule: If the reasoning contains critical flaws or confidence < 0.6, is_valid must be false and recommendation "re-reason".
Return one JSON object only."""

        result_dict = dataclasses.asdict(result)

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Problem: {original_problem}\n\nResult:\n{json.dumps(result_dict, indent=2)}",
                    }
                ],
            )
            data = extract_json_object(response.content[0].text)

            confidence_adjusted = clamp_float(
                data.get("confidence_adjusted"),
                default=local_report.confidence_adjusted,
                minimum=0.0,
                maximum=1.0,
            )
            attacks = ensure_string_list(data.get("attacks")) or local_report.attacks
            edge_cases = ensure_string_list(data.get("edge_cases")) or local_report.edge_cases
            is_valid = bool(data.get("is_valid", local_report.is_valid))
            recommendation = str(data.get("recommendation", local_report.recommendation))

            if confidence_adjusted < 0.6 and recommendation == "accept":
                is_valid = False
                recommendation = "re-reason"

            return ValidationReport(
                is_valid=is_valid,
                attacks=attacks,
                edge_cases=edge_cases,
                confidence_adjusted=confidence_adjusted,
                recommendation=recommendation,
            )
        except Exception:
            if self.allow_local_fallback:
                return local_report
            raise

    def _validate_locally(self, result: ReasoningResult, original_problem: str) -> ValidationReport:
        attacks: List[str] = []
        edge_cases: List[str] = []
        normalized_problem = original_problem.lower()

        if not result.reasoning_chain:
            attacks.append("No reasoning chain was produced, so the conclusion is not auditable.")
        if "kernel error" in result.conclusion.lower():
            attacks.append("The conclusion contains an internal synthesis error instead of a decision.")
        if len(result.conclusion.strip()) < 60:
            attacks.append("The conclusion is too terse to support multi-lens validation.")
        if not result.lens_contributions:
            attacks.append("Lens contributions are missing, so provenance is unclear.")

        if len(result.reasoning_chain) < 2:
            edge_cases.append("Short reasoning chains can miss compound interactions between constraints.")
        if "file" in normalized_problem:
            edge_cases.append("Large or malformed file inputs can invalidate the abstraction layer.")
        if any(token in normalized_problem for token in ("api", "auth", "token", "key", "anthropic")):
            edge_cases.append("External authentication and upstream availability remain operational failure points.")
        if any(token in normalized_problem for token in ("distributed", "parallel", "concurrent", "thread")):
            edge_cases.append("Concurrency can surface race conditions, stale reads, or partial failures.")

        confidence_adjusted = clamp_float(
            result.epistemic_confidence - (0.12 * len(attacks)) - (0.03 * len(edge_cases)),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )
        minimum_accept_confidence = 0.6 if attacks else 0.5
        is_valid = not attacks and confidence_adjusted >= minimum_accept_confidence

        if is_valid:
            recommendation = "accept"
        elif confidence_adjusted < 0.35:
            recommendation = "escalate_to_human"
        else:
            recommendation = "re-reason"

        return ValidationReport(
            is_valid=is_valid,
            attacks=attacks,
            edge_cases=edge_cases,
            confidence_adjusted=confidence_adjusted,
            recommendation=recommendation,
        )
