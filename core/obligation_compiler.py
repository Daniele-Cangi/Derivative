import math
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.problem_classifier import ProblemClassification, ProblemType


@dataclass
class ObligationSpec:
    name: str
    field: str
    expected_type: str
    required: bool = True


@dataclass
class CompiledObligations:
    mode: str
    schema: Dict[str, str]
    specs: List[ObligationSpec]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObligationAssessment:
    schema_valid: bool
    all_required_passed: bool
    missing_or_null_fields: List[str]
    required_results: Dict[str, bool]
    failure_reasons: List[str]


class ObligationCompiler:
    def compile(
        self,
        problem: str,
        classification: ProblemClassification,
    ) -> CompiledObligations:
        if classification.primary_type == ProblemType.SYMBOLIC:
            return self._compile_symbolic(problem)
        if classification.primary_type == ProblemType.PROBABILISTIC:
            return self._compile_probabilistic(problem)

        return CompiledObligations(mode="none", schema={}, specs=[], context={})

    def evaluate(
        self,
        compiled: CompiledObligations,
        execution_output: str,
    ) -> ObligationAssessment:
        if compiled.mode == "none":
            return ObligationAssessment(
                schema_valid=True,
                all_required_passed=True,
                missing_or_null_fields=[],
                required_results={},
                failure_reasons=[],
            )

        payload = self._load_execution_payload(execution_output)
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        if not isinstance(result, dict):
            result = {}

        missing_or_null: List[str] = []
        type_failures: List[str] = []
        required_results: Dict[str, bool] = {}

        for spec in compiled.specs:
            value = result.get(spec.field)
            if value is None:
                if spec.required:
                    missing_or_null.append(spec.field)
                    required_results[spec.name] = False
                continue
            if not self._type_matches(spec.expected_type, value):
                if spec.required:
                    type_failures.append(
                        f"Field '{spec.field}' has wrong type for obligation '{spec.name}' ({spec.expected_type})."
                    )
                    required_results[spec.name] = False

        if missing_or_null:
            reasons = [f"Missing or null required field: {field}" for field in missing_or_null]
            reasons.extend(type_failures)
            return ObligationAssessment(
                schema_valid=False,
                all_required_passed=False,
                missing_or_null_fields=missing_or_null,
                required_results=required_results,
                failure_reasons=reasons,
            )

        if type_failures:
            return ObligationAssessment(
                schema_valid=False,
                all_required_passed=False,
                missing_or_null_fields=[],
                required_results=required_results,
                failure_reasons=type_failures,
            )

        if compiled.mode == "symbolic_numeric":
            return self._evaluate_symbolic(compiled, result, required_results)
        if compiled.mode == "probabilistic_numeric":
            return self._evaluate_probabilistic(compiled, result, required_results)

        return ObligationAssessment(
            schema_valid=True,
            all_required_passed=True,
            missing_or_null_fields=[],
            required_results=required_results,
            failure_reasons=[],
        )

    def _compile_symbolic(self, problem: str) -> CompiledObligations:
        context = self._parse_symbolic_context(problem)
        schema = {
            "closed_form_ok": "bool",
            "verify_0_10_ok": "bool",
            "threshold_n": "int",
            "ratio_limit": "expr",
        }
        specs = [
            ObligationSpec("closed_form_ok", "closed_form_ok", "bool", required=True),
            ObligationSpec("verify_0_10_ok", "verify_0_10_ok", "bool", required=True),
            ObligationSpec("threshold_n", "threshold_n", "int", required=context["threshold_target"] is not None),
            ObligationSpec("ratio_limit", "ratio_limit", "expr", required=context["ratio_requested"]),
        ]
        return CompiledObligations(
            mode="symbolic_numeric",
            schema=schema,
            specs=specs,
            context=context,
        )

    def _compile_probabilistic(self, problem: str) -> CompiledObligations:
        participant_count = self._extract_first_int(
            problem,
            (
                r"(\d+)\s+participants?",
                r"(\d+)\s+channels?",
                r"(\d+)\s+nodes?",
                r"(\d+)\s+(?:[A-Za-z_]+\s+){0,3}components?",
                r"(\d+)\s+components?",
                r"\bN\s*=\s*(\d+)",
            ),
            default=1,
        )
        interception_probability = self._extract_first_float(problem, (r"\bp\s*=\s*([0-9]*\.?[0-9]+)",), default=0.01)
        template = self._detect_probabilistic_template(problem)
        if template == "survival_combinatorial":
            horizon_hours = self._extract_first_float(
                problem,
                (r"(?:for|over|during|across)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:consecutive\s+)?(?:hours?|hrs?|h)\b",),
                default=1.0,
            )
            n_components = self._extract_first_int(
                problem,
                (
                    r"(\d+)\s+(?:[A-Za-z_]+\s+){0,3}components?",
                    r"(\d+)\s+components?",
                ),
                default=4,
            )
            n_components = max(1, n_components)
            threshold_failures = self._extract_failure_threshold(problem, default=2)
            threshold_failures = max(1, min(n_components, threshold_failures))
            p_fail = min(0.999999, max(0.0, interception_probability))
            p_survive_1h = sum(
                math.comb(n_components, k) * (p_fail ** k) * ((1.0 - p_fail) ** (n_components - k))
                for k in range(threshold_failures)
            )
            p_survive_horizon = p_survive_1h**horizon_hours
            expected_hours = 1.0 / max(1e-12, 1.0 - p_survive_1h)

            schema = {
                "survival_probability": "float",
                "expected_hours": "float",
                "independent_model_survival": "float",
                "correlated_model_survival": "float",
                "independent_model_expected_hours": "float",
                "correlated_model_expected_hours": "float",
            }
            specs = [
                ObligationSpec("survival_probability", "survival_probability", "float", required=True),
                ObligationSpec("expected_hours", "expected_hours", "float", required=True),
                ObligationSpec("independent_model_survival", "independent_model_survival", "float", required=True),
                ObligationSpec("correlated_model_survival", "correlated_model_survival", "float", required=True),
                ObligationSpec(
                    "independent_model_expected_hours",
                    "independent_model_expected_hours",
                    "float",
                    required=True,
                ),
                ObligationSpec(
                    "correlated_model_expected_hours",
                    "correlated_model_expected_hours",
                    "float",
                    required=True,
                ),
            ]
            return CompiledObligations(
                mode="probabilistic_numeric",
                schema=schema,
                specs=specs,
                context={
                    "template": "survival_combinatorial",
                    "survival_probability": float(p_survive_horizon),
                    "expected_hours": float(expected_hours),
                    "independent_model_survival": float(p_survive_horizon),
                    "correlated_model_survival": float(p_survive_horizon),
                    "independent_model_expected_hours": float(expected_hours),
                    "correlated_model_expected_hours": float(expected_hours),
                },
            )

        if template == "survival":
            horizon_hours = self._extract_first_float(
                problem,
                (r"(?:for|over|during|across)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:consecutive\s+)?(?:hours?|hrs?|h)\b",),
                default=1.0,
            )
            independent_failure = min(0.999999, max(0.0, interception_probability))
            correlated_failure = min(
                0.999999,
                independent_failure * (1.0 + (0.1875 * max(0, participant_count - 1))),
            )
            independent_survival = (1.0 - independent_failure) ** horizon_hours
            correlated_survival = (1.0 - correlated_failure) ** horizon_hours
            independent_expected_hours = 1.0 / max(independent_failure, 1e-9)
            correlated_expected_hours = 1.0 / max(correlated_failure, 1e-9)

            schema = {
                "survival_probability": "float",
                "expected_hours": "float",
                "independent_model_survival": "float",
                "correlated_model_survival": "float",
                "independent_model_expected_hours": "float",
                "correlated_model_expected_hours": "float",
            }
            specs = [
                ObligationSpec("survival_probability", "survival_probability", "float", required=True),
                ObligationSpec("expected_hours", "expected_hours", "float", required=True),
                ObligationSpec("independent_model_survival", "independent_model_survival", "float", required=True),
                ObligationSpec("correlated_model_survival", "correlated_model_survival", "float", required=True),
                ObligationSpec(
                    "independent_model_expected_hours",
                    "independent_model_expected_hours",
                    "float",
                    required=True,
                ),
                ObligationSpec(
                    "correlated_model_expected_hours",
                    "correlated_model_expected_hours",
                    "float",
                    required=True,
                ),
            ]
            return CompiledObligations(
                mode="probabilistic_numeric",
                schema=schema,
                specs=specs,
                context={
                    "template": "survival",
                    "survival_probability": float(min(independent_survival, correlated_survival)),
                    "expected_hours": float(min(independent_expected_hours, correlated_expected_hours)),
                    "independent_model_survival": float(independent_survival),
                    "correlated_model_survival": float(correlated_survival),
                    "independent_model_expected_hours": float(independent_expected_hours),
                    "correlated_model_expected_hours": float(correlated_expected_hours),
                },
            )

        threshold = self._extract_first_float(
            problem,
            (r"(?:falls below|below|<)\s*([0-9]*\.?[0-9]+)",),
            default=0.001,
        )

        independent_per_round = 1.0 - ((1.0 - interception_probability) ** participant_count)
        correlated_per_round = min(
            0.999999,
            1.0 - ((1.0 - interception_probability) ** (participant_count + 2)),
        )
        independent_rounds = self._solve_round_threshold(independent_per_round, threshold)
        correlated_rounds = self._solve_round_threshold(correlated_per_round, threshold)
        minimum_rounds = max(independent_rounds, correlated_rounds)

        schema = {
            "minimum_rounds": "int",
            "independent_model_result": "float",
            "correlated_model_result": "float",
        }
        specs = [
            ObligationSpec("minimum_rounds", "minimum_rounds", "int", required=True),
            ObligationSpec("independent_model_result", "independent_model_result", "float", required=True),
            ObligationSpec("correlated_model_result", "correlated_model_result", "float", required=True),
        ]
        return CompiledObligations(
            mode="probabilistic_numeric",
            schema=schema,
            specs=specs,
            context={
                "template": "rounds",
                "minimum_rounds": minimum_rounds,
                "independent_model_result": float(independent_rounds),
                "correlated_model_result": float(correlated_rounds),
            },
        )

    def _evaluate_symbolic(
        self,
        compiled: CompiledObligations,
        result: Dict[str, Any],
        required_results: Dict[str, bool],
    ) -> ObligationAssessment:
        try:
            from sympy import N, Symbol, limit, oo, simplify, sympify
        except Exception as exc:
            return ObligationAssessment(
                schema_valid=False,
                all_required_passed=False,
                missing_or_null_fields=[],
                required_results=required_results,
                failure_reasons=[f"SymPy unavailable during obligation evaluation: {exc}"],
            )

        context = compiled.context
        n = Symbol("n", integer=True, nonnegative=True)
        failures: List[str] = []

        closed_form_text = str(result.get("closed_form", "")).strip()
        closed_form_expr = None
        try:
            closed_form_expr = sympify(closed_form_text) if closed_form_text else None
        except Exception:
            closed_form_expr = None
        sequence_symbol = self._resolve_sequence_symbol(closed_form_expr)

        direct_values = self._direct_sequence_values(context, horizon=10)
        closed_form_matches = bool(closed_form_expr is not None)
        if closed_form_expr is not None:
            for idx in range(0, 10):
                closed_value = self._substitute_symbol(closed_form_expr, sequence_symbol, idx)
                expected_value = direct_values[idx]
                if simplify(closed_value - expected_value) != 0:
                    closed_form_matches = False
                    break
        else:
            closed_form_matches = False

        reported_closed_ok = bool(result.get("closed_form_ok"))
        reported_verify_ok = bool(result.get("verify_0_10_ok"))
        closed_form_ok = closed_form_matches and reported_closed_ok
        verify_ok = closed_form_matches and reported_verify_ok

        required_results["closed_form_ok"] = closed_form_ok
        required_results["verify_0_10_ok"] = verify_ok
        if not closed_form_ok:
            failures.append("closed_form_ok failed: closed form does not match the direct recurrence trajectory.")
        if not verify_ok:
            failures.append("verify_0_10_ok failed: output did not prove 0..10 recurrence consistency.")

        threshold_target = context["threshold_target"]
        threshold_direction = context["threshold_direction"]
        expected_threshold_n = None
        if closed_form_expr is not None and threshold_target is not None:
            for idx in range(0, 800):
                numeric_expr = self._substitute_symbol(closed_form_expr, sequence_symbol, idx)
                numeric_value = float(N(numeric_expr))
                if threshold_direction == "ge" and numeric_value >= float(threshold_target):
                    expected_threshold_n = idx
                    break
                if threshold_direction == "le" and numeric_value <= float(threshold_target):
                    expected_threshold_n = idx
                    break

        reported_threshold_n = result.get("threshold_n")
        threshold_required = any(spec.name == "threshold_n" and spec.required for spec in compiled.specs)
        if threshold_required:
            threshold_ok = isinstance(reported_threshold_n, int) and expected_threshold_n == reported_threshold_n
            required_results["threshold_n"] = threshold_ok
            if not threshold_ok:
                failures.append(
                    f"threshold_n failed: expected {expected_threshold_n}, got {reported_threshold_n}."
                )

        ratio_required = any(spec.name == "ratio_limit" and spec.required for spec in compiled.specs)
        if ratio_required:
            ratio_ok = False
            ratio_text = str(result.get("ratio_limit", "")).strip()
            if ratio_text and closed_form_expr is not None:
                try:
                    n_symbol = sequence_symbol if sequence_symbol is not None else Symbol("n")
                    shifted_expr = self._substitute_symbol(closed_form_expr, n_symbol, n_symbol + 1)
                    expected_ratio = simplify(limit(shifted_expr / closed_form_expr, n_symbol, oo))
                    reported_ratio = simplify(sympify(ratio_text))
                    ratio_ok = simplify(expected_ratio - reported_ratio) == 0
                except Exception:
                    ratio_ok = False

            required_results["ratio_limit"] = ratio_ok
            if not ratio_ok:
                failures.append("ratio_limit failed: ratio limit expression does not match the derived closed form.")

        all_required_passed = all(
            passed
            for name, passed in required_results.items()
            if self._is_required_obligation(name, compiled.specs)
        )
        return ObligationAssessment(
            schema_valid=True,
            all_required_passed=all_required_passed,
            missing_or_null_fields=[],
            required_results=required_results,
            failure_reasons=failures,
        )

    def _evaluate_probabilistic(
        self,
        compiled: CompiledObligations,
        result: Dict[str, Any],
        required_results: Dict[str, bool],
    ) -> ObligationAssessment:
        template = str(compiled.context.get("template") or "rounds").strip().lower()
        if template.startswith("survival"):
            return self._evaluate_probabilistic_survival(compiled, result, required_results)

        failures: List[str] = []
        expected_min = compiled.context["minimum_rounds"]
        expected_ind = compiled.context["independent_model_result"]
        expected_corr = compiled.context["correlated_model_result"]

        minimum_rounds = result.get("minimum_rounds")
        independent = result.get("independent_model_result")
        correlated = result.get("correlated_model_result")

        minimum_ok = isinstance(minimum_rounds, int) and minimum_rounds == expected_min
        independent_ok = isinstance(independent, (int, float)) and abs(float(independent) - expected_ind) < 1e-9
        correlated_ok = isinstance(correlated, (int, float)) and abs(float(correlated) - expected_corr) < 1e-9

        required_results["minimum_rounds"] = minimum_ok
        required_results["independent_model_result"] = independent_ok
        required_results["correlated_model_result"] = correlated_ok

        if not minimum_ok:
            failures.append(f"minimum_rounds failed: expected {expected_min}, got {minimum_rounds}.")
        if not independent_ok:
            failures.append(
                f"independent_model_result failed: expected {expected_ind}, got {independent}."
            )
        if not correlated_ok:
            failures.append(
                f"correlated_model_result failed: expected {expected_corr}, got {correlated}."
            )

        all_required_passed = all(
            passed
            for name, passed in required_results.items()
            if self._is_required_obligation(name, compiled.specs)
        )
        return ObligationAssessment(
            schema_valid=True,
            all_required_passed=all_required_passed,
            missing_or_null_fields=[],
            required_results=required_results,
            failure_reasons=failures,
        )

    def _evaluate_probabilistic_survival(
        self,
        compiled: CompiledObligations,
        result: Dict[str, Any],
        required_results: Dict[str, bool],
    ) -> ObligationAssessment:
        failures: List[str] = []
        expected = compiled.context
        for field in (
            "survival_probability",
            "expected_hours",
            "independent_model_survival",
            "correlated_model_survival",
            "independent_model_expected_hours",
            "correlated_model_expected_hours",
        ):
            observed = result.get(field)
            expected_value = expected[field]
            passed = isinstance(observed, (int, float)) and abs(float(observed) - float(expected_value)) <= 1e-6
            required_results[field] = passed
            if not passed:
                failures.append(f"{field} failed: expected {expected_value}, got {observed}.")

        all_required_passed = all(
            passed
            for name, passed in required_results.items()
            if self._is_required_obligation(name, compiled.specs)
        )
        return ObligationAssessment(
            schema_valid=True,
            all_required_passed=all_required_passed,
            missing_or_null_fields=[],
            required_results=required_results,
            failure_reasons=failures,
        )

    def _parse_symbolic_context(self, problem: str) -> Dict[str, Any]:
        sequence_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*n\s*\)", problem)
        sequence_name = sequence_match.group(1) if sequence_match else "f"

        recurrence_match = re.search(
            rf"{re.escape(sequence_name)}\(\s*n\s*\)\s*=\s*([^\n\r]+)",
            problem,
            re.IGNORECASE,
        )
        rhs_raw = recurrence_match.group(1).strip() if recurrence_match else f"2*{sequence_name}(n-1)"
        rhs_raw = re.split(
            r"(?:(?:,|\.)\s*(?:with|find|determine|prove|verify|show|compute|calculate|derive)\b)|"
            r"(?:\bfor\s+n\s*(?:>=|>|≤|<=|=)\s*\d+\b)",
            rhs_raw,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        rhs_raw = rhs_raw.split("?")[0].split(";")[0].split(" where ")[0].strip(" .")
        rhs_template = re.sub(
            rf"{re.escape(sequence_name)}\(\s*n\s*-\s*1\s*\)",
            "u1",
            rhs_raw,
            flags=re.IGNORECASE,
        )
        rhs_template = re.sub(
            rf"{re.escape(sequence_name)}\(\s*n\s*-\s*2\s*\)",
            "u2",
            rhs_template,
            flags=re.IGNORECASE,
        )
        rhs_template = rhs_template.replace("^", "**")
        rhs_template = re.sub(r"(?<=\d)(?=u[12]\b)", "*", rhs_template)

        coeff_u1, coeff_u2, constant_term = self._extract_linear_coefficients(rhs_template)

        init_pattern = re.compile(
            rf"{re.escape(sequence_name)}\(\s*(\d+)\s*\)\s*=\s*([+-]?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        initial_values: Dict[int, float] = {}
        for index_text, value_text in init_pattern.findall(problem):
            index = int(index_text)
            if index <= 4:
                initial_values[index] = float(value_text)
        if 0 not in initial_values:
            initial_values[0] = 0.0
        if 1 not in initial_values:
            initial_values[1] = 1.0

        threshold_target: Optional[float] = None
        power_match = re.search(r"10\s*(?:\^|\*\*)\s*(\d+)", problem)
        if power_match:
            threshold_target = float(10 ** int(power_match.group(1)))
        if threshold_target is None:
            threshold_match = re.search(
                r"(?:>=|>|at least|exceeds?|above|below|<)\s*([0-9]+(?:\.[0-9]+)?)",
                problem,
                re.IGNORECASE,
            )
            if threshold_match:
                threshold_target = float(threshold_match.group(1))

        threshold_direction = "ge"
        if re.search(r"\b(?:below|less than|<|at most)\b", problem, re.IGNORECASE):
            threshold_direction = "le"
        ratio_requested = bool(re.search(r"\bratio\b|\blimit\b", problem, re.IGNORECASE))

        return {
            "sequence_name": sequence_name,
            "coeff_u1": coeff_u1,
            "coeff_u2": coeff_u2,
            "constant_term": constant_term,
            "initial_values": initial_values,
            "threshold_target": threshold_target,
            "threshold_direction": threshold_direction,
            "ratio_requested": ratio_requested,
        }

    def _extract_linear_coefficients(self, rhs_template: str) -> Tuple[float, float, float]:
        try:
            from sympy import Symbol, simplify, sympify
        except Exception:
            return 2.0, -1.0, 0.0

        u1 = Symbol("u1")
        u2 = Symbol("u2")
        try:
            rhs_expr = sympify(rhs_template, locals={"u1": u1, "u2": u2})
            rhs_expanded = simplify(rhs_expr.expand())
            coeff_u1 = float(simplify(rhs_expanded.coeff(u1)))
            coeff_u2 = float(simplify(rhs_expanded.coeff(u2)))
            constant = float(simplify(rhs_expanded.subs({u1: 0, u2: 0})))
            return coeff_u1, coeff_u2, constant
        except Exception:
            return 2.0, -1.0, 0.0

    def _direct_sequence_values(self, context: Dict[str, Any], horizon: int) -> Dict[int, float]:
        coeff_u1 = float(context["coeff_u1"])
        coeff_u2 = float(context["coeff_u2"])
        constant_term = float(context["constant_term"])
        initial = context["initial_values"]
        values: Dict[int, float] = {
            0: float(initial.get(0, 0.0)),
            1: float(initial.get(1, 1.0)),
        }
        for idx in range(2, horizon):
            values[idx] = (coeff_u1 * values[idx - 1]) + (coeff_u2 * values[idx - 2]) + constant_term
        return values

    def _load_execution_payload(self, execution_output: str) -> Dict[str, Any]:
        try:
            payload = json.loads(execution_output)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _type_matches(self, expected_type: str, value: Any) -> bool:
        if expected_type == "bool":
            return isinstance(value, bool)
        if expected_type == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type == "expr":
            return isinstance(value, str) and bool(value.strip())
        return True

    def _resolve_sequence_symbol(self, expression: Any) -> Optional[Any]:
        if expression is None:
            return None
        free_symbols = list(getattr(expression, "free_symbols", []))
        if not free_symbols:
            return None
        symbol_n = next((symbol for symbol in free_symbols if str(symbol) == "n"), None)
        return symbol_n or free_symbols[0]

    def _substitute_symbol(self, expression: Any, symbol: Any, value: Any) -> Any:
        if expression is None:
            return expression
        if symbol is None:
            return expression
        return expression.subs(symbol, value)

    def _is_required_obligation(self, name: str, specs: List[ObligationSpec]) -> bool:
        spec = next((item for item in specs if item.name == name), None)
        return bool(spec and spec.required)

    def _extract_first_int(self, problem: str, patterns: tuple[str, ...], default: int) -> int:
        for pattern in patterns:
            match = re.search(pattern, problem, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return default

    def _extract_first_float(self, problem: str, patterns: tuple[str, ...], default: float) -> float:
        for pattern in patterns:
            match = re.search(pattern, problem, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return default

    def _solve_round_threshold(self, per_round_probability: float, threshold: float) -> int:
        rounds = 1
        bounded_probability = min(max(per_round_probability, 1e-9), 0.999999)
        while bounded_probability**rounds >= threshold:
            rounds += 1
        return rounds

    def _detect_probabilistic_template(self, problem: str) -> str:
        lowered = problem.lower()
        combinatorial_survival_requested = bool(
            re.search(r"\bcomponents?\b", lowered)
            and re.search(r"\bfails?\s+if\b", lowered)
            and re.search(r"\bor\s+more\b", lowered)
            and re.search(r"\bconsecutive\s+hours?\b", lowered)
        )
        if combinatorial_survival_requested:
            return "survival_combinatorial"

        rounds_requested = bool(
            re.search(r"\bminimum\b[^.\n]*\bround", lowered)
            or re.search(r"falls below|below\s+[0-9]|key-agreement", lowered)
        )
        survival_requested = bool(re.search(r"\bsurviv(?:e|al)\b|\bp\s*\(\s*survive", lowered))
        expected_hours_requested = bool(
            re.search(
                r"\bexpected\s+value\b|\bexpected\s+(?:number\s+of\s+)?hours?\b|\be\s*\[\s*hours?\s*\]",
                lowered,
            )
        )
        if (survival_requested or expected_hours_requested) and not rounds_requested:
            return "survival"
        return "rounds"

    def _extract_failure_threshold(self, problem: str, default: int = 2) -> int:
        match = re.search(r"fails?\s+if\s+(\d+)\s+or\s+more", problem, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s+or\s+more\s+components?\s+fail", problem, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s+or\s+more", problem, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return default
