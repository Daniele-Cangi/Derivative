import dataclasses
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import anthropic

from core.json_utils import clamp_float, ensure_string_list, extract_json_object
from core.kernel import ReasoningResult
from core.problem_classifier import ProblemClassification, ProblemClassifier, ProblemType
from core.runtime_mode import normalize_execution_mode


@dataclass
class ValidationReport:
    is_valid: bool
    attacks: List[str]
    edge_cases: List[str]
    confidence_adjusted: float
    recommendation: str


class NumericAnswerCheck:
    """
    If the problem requires a numeric answer, the conclusion must contain
    at least one concrete number that directly addresses the question.
    Fails validation if no numeric answer is present.
    """

    def check(
        self,
        classification: ProblemClassification,
        conclusion: str,
        execution_output: str,
        problem: str = "",
    ) -> tuple[bool, str]:
        payload = self._parse_execution_payload(execution_output)
        result_payload = payload.get("result", {}) if isinstance(payload, dict) else {}
        if (
            isinstance(result_payload, dict)
            and str(result_payload.get("mode", "")).lower() == "infeasible"
            and result_payload.get("is_satisfiable") is False
        ):
            return True, ""

        if not classification.requires_numeric_answer:
            return True, ""

        import re

        conclusion_pattern = re.compile(
            r"\b(?:minimum|maximum|result|answer|requires|rounds?|steps?|hops?|probability|count|threshold|limit)\b[^.:\n]*?\b\d+\.?\d*\b",
            re.IGNORECASE,
        )
        generic_number_pattern = re.compile(r"\b\d+\.?\d*\b")
        indexed_number_pattern = re.compile(r"\b(?:n|k|step|round)\s*(?:=|>=|<=|>|<)?\s*\d+\b", re.IGNORECASE)
        output_pattern = re.compile(
            r'"(?:minimum(?:_rounds)?|maximum|result|answer|count|rounds|steps|hops|satisfiable_count|independent_model_result|correlated_model_result|survival_probability|expected_hours|independent_model_survival|correlated_model_survival|independent_model_expected_hours|correlated_model_expected_hours|threshold_index|threshold_target|result_count|verification_window|qiskit_operations|contradiction_count)"\s*:\s*-?\d+\.?\d*',
            re.IGNORECASE,
        )
        if classification.primary_type in {ProblemType.HYBRID, ProblemType.SYMBOLIC}:
            number_in_conclusion = bool(generic_number_pattern.search(conclusion))
        else:
            number_in_conclusion = bool(conclusion_pattern.search(conclusion))
        number_in_output = bool(output_pattern.search(execution_output))

        if not number_in_conclusion and not number_in_output:
            return (
                False,
                (
                    f"Problem requires a numeric answer (triggered by: {classification.numeric_keywords}) "
                    "but conclusion contains no concrete number. Validation failed regardless of reasoning confidence."
                ),
            )

        if classification.primary_type != ProblemType.HYBRID:
            objectives = self._collect_objectives(classification, problem)
            unresolved = [
                objective
                for objective in objectives
                if not self._objective_is_answered(
                    objective,
                    conclusion,
                    execution_output,
                    generic_number_pattern,
                    indexed_number_pattern,
                )
            ]
            if unresolved:
                return (
                    False,
                    (
                        "Numeric gate detected unresolved objectives. The answer includes numbers but does not map "
                        f"to all explicit requests. Missing objective coverage: {unresolved[:4]}."
                    ),
                )

        return True, ""

    def _collect_objectives(
        self,
        classification: ProblemClassification,
        problem: str,
    ) -> list[str]:
        objectives = list(classification.explicit_objectives)
        if objectives:
            return objectives
        if not problem:
            return []
        return ProblemClassifier().extract_explicit_objectives(problem)

    def _objective_is_answered(
        self,
        objective: str,
        conclusion: str,
        execution_output: str,
        number_pattern,
        indexed_number_pattern,
    ) -> bool:
        import re

        objective_text = objective.lower()
        response_text = f"{conclusion} {execution_output}".lower()
        payload = self._parse_execution_payload(execution_output)
        payload_text = self._payload_to_text(payload).lower()
        combined_text = f"{response_text} {payload_text}"

        if any(token in objective_text for token in ("closed form", "formula", "explicit expression", "recurrence")):
            if "closed_form" in payload_text:
                return True
            if re.search(r"\b[a-zA-Z_][A-Za-z0-9_]*\(\s*n\s*\)\s*=", combined_text):
                return True
            if "characteristic" in combined_text and "root" in combined_text:
                return True

        if any(token in objective_text for token in ("verify", "prove", "show")):
            if re.search(r'"(?:verified|recurrence_verified)"\s*:\s*true', execution_output, re.IGNORECASE):
                return True
            if any(token in combined_text for token in ("verified", "satisfies", "holds", "proof")):
                return True

        if any(token in objective_text for token in ("ratio", "limit")):
            if "ratio_limit" in payload_text:
                return True
            if "limit" in combined_text and number_pattern.search(combined_text):
                return True

        if any(token in objective_text for token in ("threshold", "10^", "10**", "minimum n", "smallest n", "first n")):
            if "threshold_index" in payload_text:
                return True
            if indexed_number_pattern.search(combined_text):
                return True

        objective_tokens = self._objective_tokens(objective_text)
        overlap = sum(1 for token in objective_tokens if token in combined_text)
        token_threshold = min(2, len(objective_tokens))
        if token_threshold > 0 and overlap < token_threshold:
            return False
        return bool(number_pattern.search(combined_text))

    def _objective_tokens(self, objective: str) -> list[str]:
        import re

        stop_tokens = {
            "find",
            "determine",
            "prove",
            "verify",
            "show",
            "compute",
            "calculate",
            "derive",
            "the",
            "and",
            "that",
            "with",
            "under",
            "after",
            "then",
            "such",
            "which",
            "all",
            "for",
            "from",
            "into",
            "does",
            "not",
            "than",
        }
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", objective.lower())
        return [token for token in tokens if token not in stop_tokens and len(token) > 2][:6]

    def _parse_execution_payload(self, execution_output: str) -> dict:
        try:
            payload = json.loads(execution_output) if execution_output else {}
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _payload_to_text(self, payload: dict) -> str:
        parts: List[str] = []

        def walk(value: object) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    parts.append(str(key))
                    walk(nested)
                return
            if isinstance(value, list):
                for item in value:
                    walk(item)
                return
            if value is None:
                return
            parts.append(str(value))

        walk(payload)
        return " ".join(parts)


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

        classification = ProblemClassifier().classify(original_problem)
        numeric_check = NumericAnswerCheck()
        execution_output = result.execution_result.final_output if result.execution_result is not None else ""
        numeric_ok, numeric_failure = numeric_check.check(
            classification,
            result.conclusion,
            execution_output,
            original_problem,
        )
        if not numeric_ok:
            return ValidationReport(
                is_valid=False,
                attacks=[numeric_failure],
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
        model_divergence_penalty = 0.0

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

        execution_output = result.execution_result.final_output if result.execution_result is not None else ""
        try:
            execution_payload = json.loads(execution_output) if execution_output else {}
        except json.JSONDecodeError:
            execution_payload = {}
        execution_result = execution_payload.get("result", {}) if isinstance(execution_payload, dict) else {}
        if isinstance(execution_result, dict):
            independent = execution_result.get("independent_model_result")
            correlated = execution_result.get("correlated_model_result")
            if isinstance(independent, (int, float)) and isinstance(correlated, (int, float)):
                if abs(float(independent) - float(correlated)) > 0:
                    edge_cases.append(
                        "Independent and correlated model results diverged, so confidence was reduced to reflect model sensitivity."
                    )
                    model_divergence_penalty = min(
                        0.18,
                        0.05 + (0.04 * abs(float(independent) - float(correlated))),
                    )

        confidence_adjusted = clamp_float(
            result.epistemic_confidence
            - (0.12 * len(attacks))
            - (0.03 * len(edge_cases))
            - model_divergence_penalty,
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
