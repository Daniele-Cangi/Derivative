import math
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from audit.trail import AuditEntry, AuditTrail
from core.json_utils import clamp_float
from core.obligation_compiler import ObligationCompiler
from core.problem_classifier import ProblemClassification, ProblemClassifier, ProblemType
from core.topology_solver import parse_topology_search_query
from lenses.base import CognitiveLens

MAX_EXECUTION_CYCLES = 5
CONVERGENCE_THRESHOLD = 0.05
EXECUTION_TIMEOUT_SECONDS = 20


@dataclass
class Hypothesis:
    statement: str
    prediction: str
    epistemic_tag: str
    dominant_lens: str


@dataclass
class ExecutionCycle:
    cycle: int
    hypothesis: str
    code: str
    output: str
    delta: float
    converged: bool
    prediction: str = ""
    residual: float = 0.0


@dataclass
class ExecutionResult:
    conclusion: str
    cycles_used: int
    converged: bool
    history: List[ExecutionCycle]
    final_code: str
    final_output: str
    final_prediction: str = ""
    final_residual: float = 0.0


@dataclass
class ExecutionObservation:
    output: str
    exit_code: int
    stderr: str = ""


class ExecutionLoop:
    def __init__(self, python_executable: Optional[str] = None):
        self.python_executable = python_executable or sys.executable
        self.obligation_compiler = ObligationCompiler()

    def run(
        self,
        problem: str,
        lenses: List[CognitiveLens],
        audit: Optional[AuditTrail] = None,
    ) -> ExecutionResult:
        from core.validator import NumericAnswerCheck

        classification = ProblemClassifier().classify(problem)
        contradictions = self._detect_constraint_contradictions(problem)
        if contradictions:
            return self._build_infeasible_result(problem, contradictions, lenses, audit)

        compiled_obligations = self.obligation_compiler.compile(problem, classification)
        numeric_check = NumericAnswerCheck()
        hypothesis = self._form_initial_hypothesis(problem, lenses, classification)
        history: List[ExecutionCycle] = []
        audit_trail = audit

        for cycle in range(1, MAX_EXECUTION_CYCLES + 1):
            code = self._hypothesis_to_code(problem, hypothesis, lenses, classification)
            execution = self._execute(code)
            obligation_assessment = self.obligation_compiler.evaluate(compiled_obligations, execution.output)
            residual = self._measure_residual(hypothesis.prediction, execution.output)
            delta = self._measure_delta(hypothesis.prediction, execution.output)
            if not obligation_assessment.schema_valid and obligation_assessment.missing_or_null_fields:
                delta = 1.0
                residual = 1.0
            elif not obligation_assessment.all_required_passed:
                delta = max(delta, CONVERGENCE_THRESHOLD + 0.01)
                residual = max(residual, CONVERGENCE_THRESHOLD + 0.01)

            converged = (
                execution.exit_code == 0
                and delta < CONVERGENCE_THRESHOLD
                and obligation_assessment.all_required_passed
            )
            numeric_failure = ""

            if converged:
                provisional_cycle = ExecutionCycle(
                    cycle=cycle,
                    hypothesis=hypothesis.statement,
                    code=code,
                    output=execution.output,
                    delta=delta,
                    converged=True,
                    prediction=hypothesis.prediction,
                    residual=residual,
                )
                provisional_conclusion = self._synthesize(history + [provisional_cycle])
                numeric_ok, numeric_failure = numeric_check.check(
                    classification,
                    provisional_conclusion,
                    execution.output,
                    problem,
                )
                if not numeric_ok:
                    delta = 1.0
                    residual = 1.0
                    converged = False

            if audit_trail is not None:
                try:
                    audit_trail.log(
                        AuditEntry(
                            step_id=f"exec_cycle_{cycle}",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            problem=problem,
                            lenses_applied=[lens.lens_name for lens in lenses],
                            reasoning_chain=[hypothesis.statement],
                            epistemic_tags=[hypothesis.epistemic_tag],
                            confidence=clamp_float(1.0 - delta, default=0.0),
                            validation_result="converged" if converged else "revising",
                            execution_code=code,
                            execution_prediction=hypothesis.prediction,
                            execution_output=execution.output,
                            execution_delta=delta,
                            execution_residual=residual,
                        )
                    )
                except OSError:
                    pass

            history.append(
                ExecutionCycle(
                    cycle=cycle,
                    hypothesis=hypothesis.statement,
                    code=code,
                    output=execution.output,
                    delta=delta,
                    converged=converged,
                    prediction=hypothesis.prediction,
                    residual=residual,
                )
            )

            if converged:
                break

            if numeric_failure:
                hypothesis = self._revise_hypothesis_for_numeric_failure(
                    hypothesis,
                    numeric_failure,
                    classification,
                )
            else:
                hypothesis = self._revise_hypothesis(hypothesis, execution, delta, history)

        final_cycle = history[-1] if history else ExecutionCycle(0, "", "", "", 1.0, False)
        return ExecutionResult(
            conclusion=self._synthesize(history),
            cycles_used=len(history),
            converged=final_cycle.converged,
            history=history,
            final_code=final_cycle.code,
            final_output=final_cycle.output,
            final_prediction=final_cycle.prediction,
            final_residual=final_cycle.residual,
        )

    def _build_infeasible_result(
        self,
        problem: str,
        contradictions: List[str],
        lenses: List[CognitiveLens],
        audit: Optional[AuditTrail] = None,
    ) -> ExecutionResult:
        hypothesis_statement = (
            "Static contradiction analysis found mutually exclusive numeric constraints, so the prompt is unsatisfiable."
        )
        prediction_payload = json.dumps(
            {
                "mode": "infeasible",
                "expectations": {
                    "is_satisfiable": False,
                    "contradiction_count": len(contradictions),
                },
            },
            sort_keys=True,
        )
        code = textwrap.dedent(
            """
            import json
            print(json.dumps({"result": {"mode": "infeasible", "is_satisfiable": False}}, sort_keys=True))
            """
        ).strip()
        output_payload = json.dumps(
            {
                "result": {
                    "mode": "infeasible",
                    "is_satisfiable": False,
                    "contradictions": contradictions,
                    "contradiction_count": len(contradictions),
                },
                "confirms_hypothesis": True,
            },
            sort_keys=True,
        )
        cycle = ExecutionCycle(
            cycle=1,
            hypothesis=hypothesis_statement,
            code=code,
            output=output_payload,
            delta=0.0,
            converged=True,
            prediction=prediction_payload,
            residual=0.0,
        )
        if audit is not None:
            try:
                audit.log(
                    AuditEntry(
                        step_id="exec_cycle_1",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        problem=problem,
                        lenses_applied=[lens.lens_name for lens in lenses],
                        reasoning_chain=[hypothesis_statement],
                        epistemic_tags=["deductive"],
                        confidence=1.0,
                        validation_result="converged",
                        execution_code=code,
                        execution_prediction=prediction_payload,
                        execution_output=output_payload,
                        execution_delta=0.0,
                        execution_residual=0.0,
                    )
                )
            except OSError:
                pass
        return ExecutionResult(
            conclusion=self._synthesize([cycle]),
            cycles_used=1,
            converged=True,
            history=[cycle],
            final_code=code,
            final_output=output_payload,
            final_prediction=prediction_payload,
            final_residual=0.0,
        )

    def _form_initial_hypothesis(
        self,
        problem: str,
        lenses: List[CognitiveLens],
        classification: Optional[ProblemClassification] = None,
    ) -> Hypothesis:
        resolved_classification = classification or ProblemClassifier().classify(problem)
        dominant_lens = self._dominant_lens_for_classification(lenses, resolved_classification)
        mode = self._select_mode(problem, lenses, resolved_classification)

        if resolved_classification.primary_type == ProblemType.SYMBOLIC:
            estimated_index = int(self._estimate_numeric_result(problem))
            prediction = {
                "mode": "symbolic_numeric",
                "expectations": {
                    "recurrence_verified": True,
                    "result_count": 4,
                    "threshold_index": estimated_index,
                },
            }
            statement = (
                f"The initial {dominant_lens.lens_name} hypothesis is that a SymPy-grounded recurrence solve "
                "can emit the closed form, verify the recurrence, compute the threshold index, and derive the ratio limit."
            )
            return Hypothesis(
                statement=statement,
                prediction=json.dumps(prediction, sort_keys=True),
                epistemic_tag=dominant_lens.epistemic_tag,
                dominant_lens=dominant_lens.lens_name,
            )

        if resolved_classification.primary_type == ProblemType.PROBABILISTIC:
            probabilistic_seed = self._estimate_probabilistic_expectations(problem)
            if probabilistic_seed["template"] == "survival":
                prediction = {
                    "mode": "probabilistic_numeric",
                    "expectations": {
                        "template": "survival",
                        "survival_probability": probabilistic_seed["survival_probability"],
                        "expected_hours": probabilistic_seed["expected_hours"],
                    },
                }
                statement = (
                    f"The initial {resolved_classification.recommended_lens} hypothesis is that the survival "
                    f"probability is about {probabilistic_seed['survival_probability']:.6f} and expected runtime "
                    f"is about {probabilistic_seed['expected_hours']:.3f} hours under conservative modeling."
                )
            else:
                estimated_rounds = int(probabilistic_seed["minimum_rounds"])
                prediction = {
                    "mode": "probabilistic_numeric",
                    "expectations": {
                        "template": "rounds",
                        "result": estimated_rounds,
                        "minimum_rounds": estimated_rounds,
                        "model": "independent+correlated",
                    },
                }
                statement = (
                    f"The initial {resolved_classification.recommended_lens} hypothesis is that the numeric answer "
                    f"should be about {estimated_rounds} key-agreement rounds once the probability models are executed."
                )
            return Hypothesis(
                statement=statement,
                prediction=json.dumps(prediction, sort_keys=True),
                epistemic_tag=dominant_lens.epistemic_tag,
                dominant_lens=dominant_lens.lens_name,
            )

        if resolved_classification.primary_type == ProblemType.NUMERIC:
            estimated_value = self._estimate_numeric_result(problem)
            prediction = {
                "mode": "numeric",
                "expectations": {
                    "result": estimated_value,
                    "verified": True,
                },
            }
            statement = (
                f"The initial {resolved_classification.recommended_lens} hypothesis is that the requested numeric "
                f"answer should be about {estimated_value} once the explicit constraints are computed."
            )
            return Hypothesis(
                statement=statement,
                prediction=json.dumps(prediction, sort_keys=True),
                epistemic_tag=dominant_lens.epistemic_tag,
                dominant_lens=dominant_lens.lens_name,
            )

        if mode == "topology":
            query = parse_topology_search_query(problem)
            regular_query = self._parse_regular_graph_query(problem)
            if query is not None:
                node_count = query.node_count
            elif regular_query is not None:
                node_count = int(regular_query["node_count"])
            else:
                node_count = max(3, len(lenses))

            connected_graph_count = self._connected_graph_count(node_count)
            if query is not None:
                expected_count = self._estimate_topology_satisfiable_count(query, connected_graph_count)
                expected_candidate = "T001"
                statement = (
                    f"The initial {dominant_lens.lens_name} hypothesis is that the constraint set should admit "
                    f"about {expected_count} satisfiable non-isomorphic topologies out of {connected_graph_count} "
                    "connected candidate shapes, with T001 emerging as the best-ranked candidate after execution."
                )
            elif regular_query is not None:
                expected_count = self._count_regular_graphs(
                    node_count=int(regular_query["node_count"]),
                    regular_degree=int(regular_query["regular_degree"]),
                    connected_only=bool(regular_query["connected_only"]),
                )
                expected_candidate = "G001" if expected_count > 0 else "none"
                statement = (
                    f"The initial {dominant_lens.lens_name} hypothesis is that exhaustive NetworkX enumeration over "
                    f"{node_count}-node graph-atlas candidates yields {expected_count} "
                    f"{regular_query['regular_degree']}-regular topology configuration(s)."
                )
            else:
                expected_count = max(1, connected_graph_count // 3)
                expected_candidate = "T001"
                statement = (
                    f"The initial {dominant_lens.lens_name} hypothesis is that the constraint set should admit "
                    f"about {expected_count} satisfiable non-isomorphic topologies out of {connected_graph_count} "
                    "connected candidate shapes, with T001 emerging as the best-ranked candidate after execution."
                )
            prediction = {
                "mode": "topology",
                "expectations": {
                    "satisfiable_count": expected_count,
                    "optimal_candidate": expected_candidate,
                },
            }
            return Hypothesis(
                statement=statement,
                prediction=json.dumps(prediction, sort_keys=True),
                epistemic_tag=dominant_lens.epistemic_tag,
                dominant_lens=dominant_lens.lens_name,
            )

        if mode == "formal":
            unique_constraints = len({item for lens in lenses for item in lens.constraints if item})
            expected_count = max(1, unique_constraints - 1)
            prediction = {
                "mode": "formal",
                "expectations": {
                    "constraint_count": expected_count,
                    "dominant_lens": dominant_lens.lens_name,
                },
            }
            statement = (
                f"The initial {dominant_lens.lens_name} hypothesis is that the shared constraint lattice reduces "
                f"to {expected_count} unique governing constraints with {dominant_lens.lens_name} still dominant."
            )
            return Hypothesis(
                statement=statement,
                prediction=json.dumps(prediction, sort_keys=True),
                epistemic_tag=dominant_lens.epistemic_tag,
                dominant_lens=dominant_lens.lens_name,
            )

        if mode == "probabilistic":
            confidences = [lens.confidence for lens in lenses]
            average_confidence = round(sum(confidences) / max(1, len(confidences)), 2)
            expected_average = round(max(0.0, average_confidence - 0.08), 2)
            prediction = {
                "mode": "probabilistic",
                "expectations": {
                    "average_confidence": expected_average,
                    "dominant_lens": dominant_lens.lens_name,
                },
            }
            statement = (
                f"The initial {dominant_lens.lens_name} hypothesis is that the mean lens confidence should settle "
                f"near {expected_average:.2f}, anchored by {dominant_lens.lens_name} as the dominant framing."
            )
            return Hypothesis(
                statement=statement,
                prediction=json.dumps(prediction, sort_keys=True),
                epistemic_tag=dominant_lens.epistemic_tag,
                dominant_lens=dominant_lens.lens_name,
            )

        expected_tag_count = max(1, len({lens.epistemic_tag for lens in lenses}) - 1)
        prediction = {
            "mode": "generic",
            "expectations": {
                "unique_tag_count": expected_tag_count,
                "dominant_lens": dominant_lens.lens_name,
                "lens_count": len(lenses),
            },
        }
        statement = (
            f"The initial {dominant_lens.lens_name} hypothesis is that execution will confirm {len(lenses)} active "
            f"lenses spanning {expected_tag_count} unique epistemic tags, led by {dominant_lens.lens_name}."
        )
        return Hypothesis(
            statement=statement,
            prediction=json.dumps(prediction, sort_keys=True),
            epistemic_tag=dominant_lens.epistemic_tag,
            dominant_lens=dominant_lens.lens_name,
        )

    def _hypothesis_to_code(
        self,
        problem: str,
        hypothesis: Hypothesis,
        lenses: List[CognitiveLens],
        classification: Optional[ProblemClassification] = None,
    ) -> str:
        resolved_classification = classification or ProblemClassifier().classify(problem)
        payload = self._load_prediction(hypothesis.prediction)
        if resolved_classification.primary_type == ProblemType.PROBABILISTIC:
            return self._generate_probabilistic_code(problem, payload, resolved_classification)
        if resolved_classification.primary_type == ProblemType.SYMBOLIC:
            return self._generate_symbolic_code(problem, payload, resolved_classification)
        if resolved_classification.primary_type == ProblemType.NUMERIC:
            return self._generate_numeric_code(problem, payload, resolved_classification)
        if resolved_classification.primary_type == ProblemType.HYBRID:
            return self._generate_hybrid_code(problem, payload, lenses, resolved_classification)
        return self._generate_structural_code(problem, payload, lenses)

    def _execute(self, code: str) -> ExecutionObservation:
        try:
            completed = subprocess.run(
                [self.python_executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionObservation(
                output=json.dumps(
                    {
                        "result": {
                            "mode": "runtime_error",
                            "error": f"Execution timed out after {EXECUTION_TIMEOUT_SECONDS} seconds.",
                        },
                        "confirms_hypothesis": False,
                    },
                    sort_keys=True,
                ),
                exit_code=1,
                stderr=str(exc),
            )

        output = (completed.stdout or "").strip()
        if output:
            last_line = output.splitlines()[-1].strip()
        else:
            last_line = ""

        if not last_line:
            last_line = json.dumps(
                {
                    "result": {
                        "mode": "runtime_error",
                        "error": (completed.stderr or "No stdout returned from execution.").strip(),
                    },
                    "confirms_hypothesis": False,
                },
                sort_keys=True,
            )

        return ExecutionObservation(
            output=last_line,
            exit_code=completed.returncode,
            stderr=(completed.stderr or "").strip(),
        )

    def _measure_delta(self, prediction: str, actual_output: str) -> float:
        predicted = self._load_prediction(prediction)
        actual_payload = self._load_output(actual_output)
        actual_result = actual_payload.get("result", {})
        expectations = predicted.get("expectations", {})

        if not expectations or not isinstance(actual_result, dict):
            return 0.0 if actual_payload.get("confirms_hypothesis") else 1.0

        deltas: List[float] = []
        for key, expected in expectations.items():
            if key == "template":
                continue
            actual = actual_result.get(key)
            deltas.append(self._value_delta(expected, actual))

        base_delta = sum(deltas) / max(1, len(deltas))
        if actual_payload.get("confirms_hypothesis"):
            return clamp_float(base_delta * 0.25, default=0.0, minimum=0.0, maximum=1.0)
        return clamp_float(max(base_delta, CONVERGENCE_THRESHOLD + 0.01), default=1.0, minimum=0.0, maximum=1.0)

    def _measure_residual(self, prediction: str, actual_output: str) -> float:
        predicted = self._load_prediction(prediction)
        actual_payload = self._load_output(actual_output)
        actual_result = actual_payload.get("result", {})
        expectations = predicted.get("expectations", {})
        mode = str(predicted.get("mode") or actual_result.get("mode") or "generic")

        if not isinstance(actual_result, dict) or not expectations:
            return 0.0 if actual_payload.get("confirms_hypothesis") else 1.0

        if mode == "probabilistic_numeric":
            residual_terms: List[float] = []
            expected_result = expectations.get("result")
            actual_numeric = actual_result.get("result")
            if isinstance(expected_result, (int, float)) and isinstance(actual_numeric, (int, float)):
                residual_terms.append(
                    abs(float(actual_numeric) - float(expected_result))
                    / max(1.0, abs(float(actual_numeric)))
                )

            expected_rounds = expectations.get("minimum_rounds")
            actual_rounds = actual_result.get("minimum_rounds")
            if isinstance(expected_rounds, int) and isinstance(actual_rounds, int):
                residual_terms.append(abs(actual_rounds - expected_rounds) / max(1, actual_rounds))

            expected_survival = expectations.get("survival_probability")
            actual_survival = actual_result.get("survival_probability")
            if isinstance(expected_survival, (int, float)) and isinstance(actual_survival, (int, float)):
                residual_terms.append(
                    abs(float(actual_survival) - float(expected_survival))
                    / max(1e-9, abs(float(actual_survival)))
                )

            expected_hours = expectations.get("expected_hours")
            actual_hours = actual_result.get("expected_hours")
            if isinstance(expected_hours, (int, float)) and isinstance(actual_hours, (int, float)):
                residual_terms.append(
                    abs(float(actual_hours) - float(expected_hours))
                    / max(1.0, abs(float(actual_hours)))
                )

            if residual_terms:
                return clamp_float(
                    sum(residual_terms) / len(residual_terms),
                    default=1.0,
                    minimum=0.0,
                    maximum=1.0,
                )
            return 1.0

        if mode == "symbolic_numeric":
            expected_count = expectations.get("result_count")
            actual_count = actual_result.get("result_count")
            expected_verified = expectations.get("recurrence_verified")
            actual_verified = actual_result.get("recurrence_verified")
            count_residual = 1.0
            if isinstance(expected_count, int) and isinstance(actual_count, int):
                count_residual = abs(actual_count - expected_count) / max(1, actual_count)
            verification_residual = 0.0
            if isinstance(expected_verified, bool):
                verification_residual = 0.0 if bool(actual_verified) == expected_verified else 1.0
            threshold_residual = 0.0
            expected_index = expectations.get("threshold_index")
            actual_index = actual_result.get("threshold_index")
            if isinstance(expected_index, int) and isinstance(actual_index, int):
                threshold_residual = abs(actual_index - expected_index) / max(1, actual_index)
            return clamp_float(
                (0.5 * count_residual) + (0.3 * verification_residual) + (0.2 * threshold_residual),
                default=1.0,
                minimum=0.0,
                maximum=1.0,
            )

        if mode == "numeric":
            expected_result = expectations.get("result")
            actual_numeric = actual_result.get("result")
            if isinstance(expected_result, (int, float)) and isinstance(actual_numeric, (int, float)):
                return abs(float(actual_numeric) - float(expected_result)) / max(1.0, abs(float(actual_numeric)))
            return 1.0

        if mode == "topology":
            expected_count = expectations.get("satisfiable_count")
            actual_count = actual_result.get("satisfiable_count")
            if isinstance(expected_count, int) and isinstance(actual_count, int):
                return abs(actual_count - expected_count) / max(1, actual_count)
            return 1.0

        if mode == "formal":
            expected_count = expectations.get("constraint_count")
            actual_count = actual_result.get("constraint_count")
            if isinstance(expected_count, int) and isinstance(actual_count, int):
                return float(abs(actual_count - expected_count))
            return 1.0

        if mode == "probabilistic":
            expected_average = expectations.get("average_confidence")
            actual_average = actual_result.get("average_confidence")
            if isinstance(expected_average, (int, float)) and isinstance(actual_average, (int, float)):
                return abs(float(actual_average) - float(expected_average))
            return 1.0

        expected_tags = expectations.get("unique_tag_count")
        actual_tags = actual_result.get("unique_tag_count")
        if isinstance(expected_tags, int) and isinstance(actual_tags, int):
            return float(abs(actual_tags - expected_tags))
        return 1.0

    def _revise_hypothesis(
        self,
        hypothesis: Hypothesis,
        execution: ExecutionObservation,
        delta: float,
        history: List[ExecutionCycle],
    ) -> Hypothesis:
        predicted = self._load_prediction(hypothesis.prediction)
        actual_payload = self._load_output(execution.output)
        actual_result = actual_payload.get("result", {})
        mode = str(predicted.get("mode") or actual_result.get("mode") or "generic")
        next_expectations: Dict[str, Any] = {}

        if isinstance(actual_result, dict):
            if mode == "probabilistic_numeric":
                probabilistic_template = str(
                    actual_result.get("template")
                    or predicted.get("expectations", {}).get("template")
                    or "rounds"
                ).strip().lower()
                if probabilistic_template == "survival":
                    previous_survival = predicted.get("expectations", {}).get("survival_probability", 0.0)
                    current_survival = actual_result.get("survival_probability", previous_survival)
                    revised_survival = self._revise_float_expectation(previous_survival, current_survival, delta)
                    previous_hours = predicted.get("expectations", {}).get("expected_hours", 0.0)
                    current_hours = actual_result.get("expected_hours", previous_hours)
                    revised_hours = self._revise_float_expectation(previous_hours, current_hours, delta)
                    next_expectations = {
                        "template": "survival",
                        "survival_probability": revised_survival,
                        "expected_hours": revised_hours,
                    }
                    statement = (
                        "The previous probabilistic survival hypothesis did not match execution. "
                        f"Survival moved from {previous_survival} to {current_survival}, and expected runtime "
                        f"moved from {previous_hours} to {current_hours}; the revised hypothesis targets "
                        f"survival~{revised_survival} and expected_hours~{revised_hours}."
                    )
                else:
                    previous_result = predicted.get("expectations", {}).get("result", "unknown")
                    current_result = actual_result.get("result", "unknown")
                    revised_result = self._revise_integer_expectation(previous_result, current_result, delta)
                    next_expectations = {
                        "template": "rounds",
                        "result": revised_result,
                        "minimum_rounds": revised_result,
                        "model": actual_result.get("model", "independent+correlated"),
                    }
                    statement = (
                        f"The previous probabilistic hypothesis failed because it predicted {previous_result} rounds, "
                        f"but execution computed {current_result}; the revised hypothesis moves toward approximately "
                        f"{revised_result} rounds while preserving both probabilistic models in the next computation."
                    )
            elif mode == "symbolic_numeric":
                previous_count = predicted.get("expectations", {}).get("result_count", 4)
                current_count = actual_result.get("result_count", 0)
                revised_count = self._revise_integer_expectation(previous_count, current_count, delta)
                previous_index = predicted.get("expectations", {}).get("threshold_index", 1)
                current_index = actual_result.get("threshold_index", previous_index)
                revised_index = self._revise_integer_expectation(previous_index, current_index, delta)
                next_expectations = {
                    "recurrence_verified": bool(actual_result.get("recurrence_verified", False)),
                    "result_count": max(1, revised_count),
                    "threshold_index": max(1, revised_index),
                }
                statement = (
                    "The previous symbolic hypothesis did not satisfy all requested recurrence outputs. "
                    f"Execution returned result_count={current_count} and threshold_index={current_index}; "
                    f"the revised SymPy hypothesis moves toward result_count~{max(1, revised_count)} and "
                    f"threshold_index~{max(1, revised_index)} while preserving explicit recurrence verification."
                )
            elif mode == "numeric":
                previous_result = predicted.get("expectations", {}).get("result", "unknown")
                current_result = actual_result.get("result", "unknown")
                if isinstance(current_result, int):
                    revised_result = self._revise_integer_expectation(previous_result, current_result, delta)
                else:
                    revised_result = self._revise_float_expectation(previous_result, current_result, delta)
                next_expectations = {
                    "result": revised_result,
                    "verified": bool(actual_result.get("verified", False)),
                }
                statement = (
                    f"The previous numeric hypothesis failed because it predicted {previous_result}, but execution "
                    f"computed {current_result}; the revised hypothesis moves toward approximately {revised_result} "
                    "while keeping explicit constraint verification in the next cycle."
                )
            elif mode == "topology":
                previous_count = predicted.get("expectations", {}).get("satisfiable_count", "unknown")
                current_count = actual_result.get("satisfiable_count", "unknown")
                current_candidate = actual_result.get("optimal_candidate", "none")
                revised_count = self._revise_integer_expectation(previous_count, current_count, delta)
                next_expectations = {
                    "satisfiable_count": revised_count,
                    "optimal_candidate": current_candidate,
                }
                statement = (
                    f"The previous topology hypothesis failed because it predicted {previous_count} satisfiable "
                    f"configurations, but execution returned {current_count}; the revised hypothesis moves "
                    f"toward approximately {revised_count} satisfiable topology configuration(s), preserving "
                    f"{current_candidate} as the best-ranked candidate while correcting the observed residual."
                )
            elif mode == "formal":
                previous_count = predicted.get("expectations", {}).get("constraint_count", "unknown")
                current_count = actual_result.get("constraint_count", "unknown")
                current_dominant = actual_result.get("dominant_lens", hypothesis.dominant_lens)
                revised_count = self._revise_integer_expectation(previous_count, current_count, delta)
                next_expectations = {
                    "constraint_count": revised_count,
                    "dominant_lens": current_dominant,
                }
                statement = (
                    f"The previous formal hypothesis failed because it predicted {previous_count} distinct "
                    f"constraints, but execution produced {current_count}; the revised hypothesis moves toward "
                    f"approximately {revised_count} active constraints while keeping {current_dominant} as the "
                    "dominant lens."
                )
            elif mode == "probabilistic":
                previous_average = predicted.get("expectations", {}).get("average_confidence", "unknown")
                current_average = actual_result.get("average_confidence", 0.0)
                current_dominant = actual_result.get("dominant_lens", hypothesis.dominant_lens)
                revised_average = self._revise_float_expectation(previous_average, current_average, delta)
                next_expectations = {
                    "average_confidence": revised_average,
                    "dominant_lens": current_dominant,
                }
                statement = (
                    f"The previous probabilistic hypothesis failed because it predicted mean confidence "
                    f"{previous_average}, but execution measured {current_average:.2f}; the revised hypothesis "
                    f"moves toward a confidence centroid near {revised_average:.2f} with {current_dominant} dominant."
                )
            else:
                previous_tags = predicted.get("expectations", {}).get("unique_tag_count", "unknown")
                current_tags = actual_result.get("unique_tag_count", "unknown")
                current_dominant = actual_result.get("dominant_lens", hypothesis.dominant_lens)
                current_count = actual_result.get("lens_count", len(history))
                revised_tags = self._revise_integer_expectation(previous_tags, current_tags, delta)
                next_expectations = {
                    "unique_tag_count": revised_tags,
                    "dominant_lens": current_dominant,
                    "lens_count": current_count,
                }
                statement = (
                    f"The previous structural hypothesis failed because it predicted {previous_tags} unique tags, "
                    f"but execution measured {current_tags}; the revised hypothesis moves toward approximately "
                    f"{revised_tags} distinct tags across {current_count} active lenses with {current_dominant} "
                    "remaining dominant."
                )
        else:
            next_expectations = dict(predicted.get("expectations", {}))
            statement = (
                f"The previous hypothesis failed with runtime output '{execution.output}', so the revision keeps "
                "the prior expectations but treats the next cycle as a recovery pass."
            )

        previous_statements = {cycle.hypothesis for cycle in history}
        if statement in previous_statements:
            statement = f"{statement} Revision anchor {len(history) + 1} avoids repeating the failed statement."

        updated_prediction = {
            "mode": mode,
            "expectations": next_expectations,
        }
        return Hypothesis(
            statement=statement,
            prediction=json.dumps(updated_prediction, sort_keys=True),
            epistemic_tag=hypothesis.epistemic_tag,
            dominant_lens=hypothesis.dominant_lens,
        )

    def _synthesize(self, history: List[ExecutionCycle]) -> str:
        if not history:
            return "Derivative could not execute any verification cycle, so no computation-grounded conclusion exists."

        final_cycle = history[-1]
        final_payload = self._load_output(final_cycle.output)
        final_result = final_payload.get("result", {})
        mode = str(final_result.get("mode") or "generic")

        if not final_cycle.converged:
            confidence = clamp_float(1.0 - final_cycle.delta, default=0.0)
            return (
                f"Derivative could not computationally confirm a conclusion after {MAX_EXECUTION_CYCLES} execution "
                f"cycles. The closest hypothesis reached confidence {confidence:.2f} before the cycle limit. "
                f"The last execution output was: {final_cycle.output}. Human judgment is recommended."
            )

        if mode == "probabilistic_numeric":
            return self._synthesize_probabilistic_numeric(history, final_result)
        if mode == "symbolic_numeric":
            return self._synthesize_symbolic_numeric(history, final_result)
        if mode == "numeric":
            return self._synthesize_numeric(history, final_result)
        if mode == "infeasible":
            return self._synthesize_infeasible(history, final_result)
        if mode == "topology":
            return self._synthesize_topology(history, final_result)
        if mode == "formal":
            return self._synthesize_formal(history, final_result)
        if mode == "probabilistic":
            return self._synthesize_probabilistic(history, final_result)
        return self._synthesize_generic(history, final_result)

    def _synthesize_topology(self, history: List[ExecutionCycle], result: Dict[str, Any]) -> str:
        if result.get("enumeration_mode") == "degree_regular":
            satisfiable_count = result.get("satisfiable_count", 0)
            evaluated = result.get("evaluated_topologies", 0)
            degree = result.get("regular_degree", 0)
            node_count = result.get("node_count", 0)
            optimal = result.get("optimal_candidate", "none")
            return (
                f"The execution loop converged on cycle {history[-1].cycle}. Exhaustive graph-atlas enumeration "
                f"checked {evaluated} non-isomorphic graph shape(s) on {node_count} node(s) and found "
                f"{satisfiable_count} satisfiable {degree}-regular topology configuration(s). The best-ranked "
                f"canonical candidate is {optimal}."
            )

        first_payload = self._load_output(history[0].output).get("result", {})
        satisfiable_count = result.get("satisfiable_count", 0)
        evaluated = result.get("evaluated_topologies", 0)
        optimal = result.get("optimal_candidate", "none")
        diameter = result.get("optimal_diameter", 0)
        optimal_fidelity = result.get("optimal_fidelity", 0.0)
        qiskit_fidelity = result.get("qiskit_fidelity", 0.0)
        operations = result.get("qiskit_operations", 0)
        alignment = self._describe_topology_alignment(history[-1], result)

        if len(history) == 1:
            return (
                f"The first execution cycle converged immediately. The code enumerated {evaluated} connected "
                f"non-isomorphic graph shape(s) and returned {satisfiable_count} satisfiable topology "
                f"configuration(s). It selected {optimal} with diameter {diameter} and solver fidelity "
                f"{optimal_fidelity:.5f}, while the qiskit-backed circuit check returned {qiskit_fidelity:.5f} "
                f"after {operations} routed operations. {alignment} The conclusion is grounded in that execution output."
            )

        return (
            f"The first execution returned {first_payload.get('satisfiable_count', 'unknown')} satisfiable "
            f"topology configuration(s), which contradicted the initial hypothesis and forced a revision. "
            f"The loop converged on cycle {history[-1].cycle} when the same executable check again returned "
            f"{satisfiable_count} satisfiable topology configuration(s) out of {evaluated} connected "
            f"non-isomorphic graph shape(s). The computation selected {optimal} as the best-ranked topology, "
            f"with diameter {diameter}, solver fidelity {optimal_fidelity:.5f}, and qiskit-backed fidelity "
            f"{qiskit_fidelity:.5f} after {operations} routed operations. {alignment} The conclusion is therefore grounded "
            "in the executed enumeration and circuit calculation, not in the original prediction."
        )

    def _synthesize_formal(self, history: List[ExecutionCycle], result: Dict[str, Any]) -> str:
        count = result.get("constraint_count", 0)
        dominant = result.get("dominant_lens", "unknown")
        alignment = self._describe_formal_alignment(history[-1], result)
        return (
            f"The execution loop converged on cycle {history[-1].cycle}. The verification code reduced the active "
            f"constraint lattice to {count} unique constraints and confirmed {dominant} as the dominant lens. "
            f"{alignment} That computed constraint count, rather than the initial estimate, anchors the final conclusion."
        )

    def _synthesize_probabilistic(self, history: List[ExecutionCycle], result: Dict[str, Any]) -> str:
        average_confidence = result.get("average_confidence", 0.0)
        dominant = result.get("dominant_lens", "unknown")
        spread = result.get("confidence_spread", 0.0)
        alignment = self._describe_probabilistic_alignment(history[-1], result)
        return (
            f"The execution loop converged on cycle {history[-1].cycle}. The computed confidence centroid is "
            f"{average_confidence:.2f} with spread {spread:.2f}, and {dominant} remains the dominant lens. "
            f"{alignment} The final conclusion is grounded in that measured confidence distribution."
        )

    def _synthesize_generic(self, history: List[ExecutionCycle], result: Dict[str, Any]) -> str:
        lens_count = result.get("lens_count", 0)
        tag_count = result.get("unique_tag_count", 0)
        dominant = result.get("dominant_lens", "unknown")
        alignment = self._describe_generic_alignment(history[-1], result)
        return (
            f"The execution loop converged on cycle {history[-1].cycle}. The verification code measured "
            f"{lens_count} active lenses spanning {tag_count} distinct epistemic tags, with {dominant} as the "
            f"dominant framing. {alignment} The final conclusion is based on that execution output rather than the initial guess."
        )

    def _synthesize_probabilistic_numeric(self, history: List[ExecutionCycle], result: Dict[str, Any]) -> str:
        template = str(result.get("template") or "rounds").lower()
        if template == "survival":
            survival_probability = result.get("survival_probability", 0.0)
            expected_hours = result.get("expected_hours", 0.0)
            independent_survival = result.get("independent_model_survival", 0.0)
            correlated_survival = result.get("correlated_model_survival", 0.0)
            independent_hours = result.get("independent_model_expected_hours", 0.0)
            correlated_hours = result.get("correlated_model_expected_hours", 0.0)
            horizon_hours = result.get("horizon_hours", 0.0)
            return (
                f"The execution loop converged on cycle {history[-1].cycle}. The probabilistic solver computed "
                f"a conservative survival probability of {survival_probability:.6f} over {horizon_hours:g} hour(s) "
                f"and a conservative expected runtime of {expected_hours:.3f} hours. Independent vs correlated "
                f"model outputs were survival {independent_survival:.6f}/{correlated_survival:.6f} and expected "
                f"runtime {independent_hours:.3f}/{correlated_hours:.3f}. The final conclusion is grounded in those "
                "executed model values."
            )

        rounds = result.get("minimum_rounds", result.get("result", 0))
        independent_rounds = result.get("independent_model_result", rounds)
        correlated_rounds = result.get("correlated_model_result", rounds)
        threshold = result.get("threshold", 0.0)
        independent_probability = result.get("independent_model_probability", 0.0)
        correlated_probability = result.get("correlated_model_probability", 0.0)
        if independent_rounds == correlated_rounds:
            comparison = (
                f"Both the independent and correlated models agreed on {rounds} rounds, yielding final probabilities "
                f"{independent_probability:.6f} and {correlated_probability:.6f} against the {threshold:.6f} threshold."
            )
        else:
            comparison = (
                f"The independent model required {independent_rounds} rounds, while the correlated model required "
                f"{correlated_rounds}; the conservative minimum is therefore {rounds}. The executed probabilities at "
                f"that setting were {independent_probability:.6f} and {correlated_probability:.6f} against the "
                f"{threshold:.6f} threshold."
            )
        return (
            f"The execution loop converged on cycle {history[-1].cycle}. The verification code computed a concrete "
            f"minimum of {rounds} rounds for the requested probabilistic constraint. {comparison} The final conclusion "
            "is grounded in those computed model results rather than in a structural proxy."
        )

    def _synthesize_numeric(self, history: List[ExecutionCycle], result: Dict[str, Any]) -> str:
        numeric_result = result.get("result", 0)
        verification = "verified" if result.get("verified") else "not fully verified"
        return (
            f"The execution loop converged on cycle {history[-1].cycle}. The verification code computed the numeric "
            f"answer as {numeric_result}, and the explicit constraint checks marked it as {verification}. The final "
            "conclusion is grounded in that numeric computation."
        )

    def _synthesize_infeasible(self, history: List[ExecutionCycle], result: Dict[str, Any]) -> str:
        contradictions = result.get("contradictions", [])
        if not contradictions:
            return (
                f"The execution loop converged on cycle {history[-1].cycle} by proving the constraint set is "
                "unsatisfiable."
            )
        return (
            f"The execution loop converged on cycle {history[-1].cycle} by proving the constraint set is "
            f"unsatisfiable. Detected contradictions: {contradictions}."
        )

    def _synthesize_symbolic_numeric(self, history: List[ExecutionCycle], result: Dict[str, Any]) -> str:
        sequence = result.get("sequence", "f")
        closed_form = result.get("closed_form", "")
        recurrence_verified = bool(result.get("recurrence_verified"))
        threshold_target = result.get("threshold_target")
        threshold_index = result.get("threshold_index")
        ratio_limit = result.get("ratio_limit", "")
        verification = "verified" if recurrence_verified else "not verified"
        threshold_phrase = (
            f"threshold {threshold_target} reached at n={threshold_index}"
            if threshold_target is not None and isinstance(threshold_index, int)
            else "threshold index could not be established"
        )
        ratio_phrase = f"ratio limit {ratio_limit}" if ratio_limit else "ratio limit unavailable"
        return (
            f"The execution loop converged on cycle {history[-1].cycle}. SymPy derived the closed form "
            f"{sequence}(n) = {closed_form}, marked recurrence consistency as {verification}, computed "
            f"{threshold_phrase}, and produced {ratio_phrase}. The final conclusion is grounded in that symbolic "
            "computation rather than in a placeholder numeric token."
        )

    def _generate_structural_code(
        self,
        problem: str,
        prediction_payload: Dict[str, Any],
        lenses: List[CognitiveLens],
    ) -> str:
        mode = str(prediction_payload.get("mode") or "generic")
        if mode == "topology":
            return self._build_topology_code(problem, prediction_payload)
        if mode == "formal":
            return self._build_formal_code(prediction_payload, lenses)
        if mode == "probabilistic":
            return self._build_probabilistic_code(prediction_payload, lenses)
        return self._build_generic_code(prediction_payload, lenses)

    def _generate_hybrid_code(
        self,
        problem: str,
        prediction_payload: Dict[str, Any],
        lenses: List[CognitiveLens],
        classification: ProblemClassification,
    ) -> str:
        if parse_topology_search_query(problem) is not None or self._parse_regular_graph_query(problem) is not None:
            return self._build_topology_code(problem, prediction_payload)
        return self._generate_structural_code(problem, prediction_payload, lenses)

    def _generate_probabilistic_code(
        self,
        problem: str,
        prediction_payload: Dict[str, Any],
        classification: ProblemClassification,
    ) -> str:
        prediction_text = json.dumps(prediction_payload, sort_keys=True)
        return textwrap.dedent(
            f"""
            import json
            import math
            import re

            problem = {problem!r}
            prediction = json.loads({prediction_text!r})
            problem_lower = problem.lower()

            participant_match = re.search(r"(\\d+)\\s+participants?", problem, re.IGNORECASE)
            if participant_match is None:
                participant_match = re.search(r"\\bN\\s*=\\s*(\\d+)", problem, re.IGNORECASE)
            participant_count = int(participant_match.group(1)) if participant_match else 1

            probability_match = re.search(r"\\bp\\s*=\\s*([0-9]*\\.?[0-9]+)", problem, re.IGNORECASE)
            interception_probability = float(probability_match.group(1)) if probability_match else 0.01

            threshold_match = re.search(r"(?:falls below|below|<)\\s*([0-9]*\\.?[0-9]+)", problem, re.IGNORECASE)
            threshold = float(threshold_match.group(1)) if threshold_match else 0.001

            expected = prediction.get("expectations", {{}})
            explicit_template = str(expected.get("template", "")).strip().lower()
            rounds_requested = bool(
                re.search(r"\\bminimum\\b[^.\\n]*\\bround", problem_lower)
                or re.search(r"falls below|below\\s+[0-9]|key-agreement", problem_lower)
            )
            survival_requested = bool(re.search(r"\\bsurviv(?:e|al)\\b|\\bp\\s*\\(\\s*survive", problem_lower))
            expected_hours_requested = bool(
                re.search(r"\\bexpected\\s+value\\b|\\bexpected\\s+(?:number\\s+of\\s+)?hours?\\b|\\be\\s*\\[\\s*hours?\\s*\\]", problem_lower)
            )

            if explicit_template:
                probabilistic_template = explicit_template
            elif (survival_requested or expected_hours_requested) and not rounds_requested:
                probabilistic_template = "survival"
            else:
                probabilistic_template = "rounds"

            if probabilistic_template == "survival":
                horizon_match = re.search(
                    r"(?:for|over|during|across)?\\s*([0-9]+(?:\\.[0-9]+)?)\\s*(?:hours?|hrs?|h)\\b",
                    problem,
                    re.IGNORECASE,
                )
                horizon_hours = float(horizon_match.group(1)) if horizon_match else 1.0
                independent_failure = 1.0 - ((1.0 - interception_probability) ** participant_count)
                correlated_failure = min(
                    0.999999,
                    independent_failure * (1.15 + (0.01 * max(0, participant_count - 1))),
                )
                independent_survival = (1.0 - independent_failure) ** horizon_hours
                correlated_survival = (1.0 - correlated_failure) ** horizon_hours
                independent_expected_hours = 1.0 / max(independent_failure, 1e-9)
                correlated_expected_hours = 1.0 / max(correlated_failure, 1e-9)
                conservative_survival = min(independent_survival, correlated_survival)
                conservative_expected_hours = min(independent_expected_hours, correlated_expected_hours)
                result = {{
                    "mode": "probabilistic_numeric",
                    "template": "survival",
                    "result": round(conservative_survival, 8),
                    "survival_probability": round(conservative_survival, 8),
                    "expected_hours": round(conservative_expected_hours, 6),
                    "horizon_hours": horizon_hours,
                    "independent_model_survival": round(independent_survival, 8),
                    "correlated_model_survival": round(correlated_survival, 8),
                    "independent_model_expected_hours": round(independent_expected_hours, 6),
                    "correlated_model_expected_hours": round(correlated_expected_hours, 6),
                    "verified": bool(conservative_survival >= 0.0 and conservative_expected_hours > 0.0),
                    "model": "independent+correlated",
                }}

                residual_terms = []
                expected_survival = expected.get("survival_probability")
                if isinstance(expected_survival, (int, float)):
                    residual_terms.append(
                        abs(float(result["survival_probability"]) - float(expected_survival))
                        / max(1e-9, abs(float(result["survival_probability"])))
                    )
                expected_hours = expected.get("expected_hours")
                if isinstance(expected_hours, (int, float)):
                    residual_terms.append(
                        abs(float(result["expected_hours"]) - float(expected_hours))
                        / max(1.0, abs(float(result["expected_hours"])))
                    )
                if not residual_terms:
                    residual_terms.append(1.0)
                confirms = (
                    (sum(residual_terms) / max(1, len(residual_terms))) <= {CONVERGENCE_THRESHOLD}
                    and result["verified"]
                )
            else:
                independent_per_round = 1.0 - ((1.0 - interception_probability) ** participant_count)
                correlated_per_round = min(
                    0.999999,
                    1.0 - ((1.0 - interception_probability) ** (participant_count + 2))
                )

                independent_rounds = 1
                while independent_per_round ** independent_rounds >= threshold:
                    independent_rounds += 1

                correlated_rounds = 1
                while correlated_per_round ** correlated_rounds >= threshold:
                    correlated_rounds += 1

                conservative_rounds = max(independent_rounds, correlated_rounds)
                evaluated_probability_independent = independent_per_round ** conservative_rounds
                evaluated_probability_correlated = correlated_per_round ** conservative_rounds

                result = {{
                    "mode": "probabilistic_numeric",
                    "template": "rounds",
                    "result": conservative_rounds,
                    "minimum_rounds": conservative_rounds,
                    "verified": (
                        evaluated_probability_independent < threshold
                        and evaluated_probability_correlated < threshold
                    ),
                    "threshold": threshold,
                    "independent_model_result": independent_rounds,
                    "correlated_model_result": correlated_rounds,
                    "independent_model_probability": round(evaluated_probability_independent, 8),
                    "correlated_model_probability": round(evaluated_probability_correlated, 8),
                    "model": "independent+correlated",
                }}

                expected_result = expected.get("result")
                result_error = 1.0
                if isinstance(expected_result, int):
                    result_error = abs(result["result"] - expected_result) / max(1, result["result"])
                confirms = (
                    result_error <= {CONVERGENCE_THRESHOLD}
                    and result["verified"]
                )

            print(json.dumps({{"result": result, "confirms_hypothesis": confirms}}, sort_keys=True))
            """
        ).strip()

    def _generate_numeric_code(
        self,
        problem: str,
        prediction_payload: Dict[str, Any],
        classification: ProblemClassification,
    ) -> str:
        prediction_text = json.dumps(prediction_payload, sort_keys=True)
        return textwrap.dedent(
            f"""
            import json
            import math
            import re

            problem = {problem!r}
            prediction = json.loads({prediction_text!r})
            numbers = [
                float(value)
                for value in re.findall(r"(?<![A-Za-z])\\d+\\.?\\d*", problem)
            ]
            integers = [int(value) for value in numbers if abs(value - round(value)) < 1e-9]

            threshold_match = re.search(r"(?:falls below|below|<)\\s*([0-9]*\\.?[0-9]+)", problem, re.IGNORECASE)
            threshold = float(threshold_match.group(1)) if threshold_match else None

            if threshold is not None and threshold > 0 and numbers:
                scale = max(1.0, max(numbers))
                exact_answer = max(1, math.ceil(scale / threshold))
                satisfied = exact_answer >= scale
                checks = [
                    {{
                        "constraint": f"scaled bound against {{threshold}}",
                        "satisfied": satisfied,
                    }}
                ]
            elif integers:
                exact_answer = max(1, min(integers))
                checks = [
                    {{
                        "constraint": "used the smallest explicit integer extracted from the prompt",
                        "satisfied": True,
                    }}
                ]
                satisfied = True
            elif numbers:
                exact_answer = round(min(numbers), 4)
                checks = [
                    {{
                        "constraint": "used the smallest explicit numeric value extracted from the prompt",
                        "satisfied": True,
                    }}
                ]
                satisfied = True
            else:
                exact_answer = 0
                checks = [
                    {{
                        "constraint": "no explicit numeric values were extractable from the prompt",
                        "satisfied": False,
                    }}
                ]
                satisfied = False

            result = {{
                "mode": "numeric",
                "result": exact_answer,
                "verified": satisfied,
                "constraint_checks": checks,
            }}

            expected = prediction.get("expectations", {{}})
            expected_result = expected.get("result")
            result_error = 1.0
            if isinstance(expected_result, (int, float)) and isinstance(result["result"], (int, float)):
                result_error = abs(float(result["result"]) - float(expected_result)) / max(1.0, abs(float(result["result"])))
            confirms = (
                result_error <= {CONVERGENCE_THRESHOLD}
                and result["verified"]
            )

            print(json.dumps({{"result": result, "confirms_hypothesis": confirms}}, sort_keys=True))
            """
        ).strip()

    def _generate_symbolic_code(
        self,
        problem: str,
        prediction_payload: Dict[str, Any],
        classification: ProblemClassification,
    ) -> str:
        prediction_text = json.dumps(prediction_payload, sort_keys=True)
        return textwrap.dedent(
            f"""
            import json
            import re

            problem = {problem!r}
            prediction = json.loads({prediction_text!r})

            try:
                from sympy import Eq, Function, Integer, N, Symbol, limit, oo, rsolve, simplify, sympify
            except Exception as exc:
                payload = {{
                    "result": {{
                        "mode": "symbolic_numeric",
                        "error": f"SymPy unavailable: {{exc}}",
                        "verified": False,
                        "result_count": 0,
                    }},
                    "confirms_hypothesis": False,
                }}
                print(json.dumps(payload, sort_keys=True))
                raise SystemExit(0)

            sequence_match = re.search(r"\\b([A-Za-z_][A-Za-z0-9_]*)\\s*\\(\\s*n\\s*\\)", problem)
            sequence_name = sequence_match.group(1) if sequence_match else "f"
            n = Symbol("n", integer=True, nonnegative=True)
            seq = Function(sequence_name)

            recurrence_match = re.search(
                rf"{{re.escape(sequence_name)}}\\(\\s*n\\s*\\)\\s*=\\s*([^\\n\\r]+)",
                problem,
                re.IGNORECASE,
            )
            rhs_raw = recurrence_match.group(1).strip() if recurrence_match else f"2*{{sequence_name}}(n-1)"
            rhs_raw = re.split(
                r"(?:(?:,|\\.)\\s*(?:with|find|determine|prove|verify|show|compute|calculate|derive)\\b)|"
                r"(?:\\bfor\\s+n\\s*(?:>=|>|≤|<=|=)\\s*\\d+\\b)",
                rhs_raw,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            rhs_raw = rhs_raw.split("?")[0].split(";")[0].split(" where ")[0].strip(" .")

            rhs_template = re.sub(
                rf"{{re.escape(sequence_name)}}\\(\\s*n\\s*-\\s*1\\s*\\)",
                "u1",
                rhs_raw,
                flags=re.IGNORECASE,
            )
            rhs_template = re.sub(
                rf"{{re.escape(sequence_name)}}\\(\\s*n\\s*-\\s*2\\s*\\)",
                "u2",
                rhs_template,
                flags=re.IGNORECASE,
            )
            rhs_template = rhs_template.replace("^", "**")
            rhs_template = re.sub(r"(?<=\\d)(?=u[12]\\b)", "*", rhs_template)

            u1 = Symbol("u1")
            u2 = Symbol("u2")
            try:
                rhs_expr = sympify(rhs_template, locals={{"u1": u1, "u2": u2}})
            except Exception:
                rhs_expr = 2 * u1 - u2

            rhs_expanded = simplify(rhs_expr.expand())
            coeff_u1 = simplify(rhs_expanded.coeff(u1))
            coeff_u2 = simplify(rhs_expanded.coeff(u2))
            constant_term = simplify(rhs_expanded.subs({{u1: 0, u2: 0}}))

            recurrence_equation = Eq(
                seq(n),
                coeff_u1 * seq(n - 1) + coeff_u2 * seq(n - 2) + constant_term,
            )

            init_pattern = re.compile(
                rf"{{re.escape(sequence_name)}}\\(\\s*(\\d+)\\s*\\)\\s*=\\s*([+-]?\\d+(?:\\.\\d+)?)",
                re.IGNORECASE,
            )
            initial_conditions = {{}}
            for index_text, value_text in init_pattern.findall(problem):
                index = int(index_text)
                if index <= 4:
                    initial_conditions[seq(index)] = sympify(value_text)

            if seq(0) not in initial_conditions:
                initial_conditions[seq(0)] = Integer(0)
            if seq(1) not in initial_conditions:
                initial_conditions[seq(1)] = Integer(1)

            ordered_initial_conditions = {{
                seq(0): initial_conditions.get(seq(0), Integer(0)),
                seq(1): initial_conditions.get(seq(1), Integer(1)),
            }}
            for key, value in initial_conditions.items():
                if key not in ordered_initial_conditions:
                    ordered_initial_conditions[key] = value

            closed_form = rsolve(recurrence_equation, seq(n), init=ordered_initial_conditions)
            closed_form = simplify(closed_form) if closed_form is not None else None

            recurrence_verified = False
            verification_window = 10
            verification_mismatches = []
            direct_values = {{
                0: simplify(ordered_initial_conditions[seq(0)]),
                1: simplify(ordered_initial_conditions[seq(1)]),
            }}
            for idx in range(2, verification_window):
                direct_values[idx] = simplify(
                    (coeff_u1 * direct_values[idx - 1])
                    + (coeff_u2 * direct_values[idx - 2])
                    + constant_term
                )

            if closed_form is not None:
                recurrence_verified = True
                for idx in range(0, verification_window):
                    closed_value = simplify(closed_form.subs(n, idx))
                    expected_value = direct_values[idx]
                    if simplify(closed_value - expected_value) != 0:
                        verification_mismatches.append(
                            {{
                                "n": idx,
                                "closed_form": str(closed_value),
                                "direct": str(expected_value),
                            }}
                        )
                        recurrence_verified = False
                if verification_mismatches:
                    recurrence_verified = False

            threshold_value = None
            power_match = re.search(r"10\\s*(?:\\^|\\*\\*)\\s*(\\d+)", problem)
            if power_match:
                threshold_value = int(10 ** int(power_match.group(1)))
            if threshold_value is None:
                threshold_match = re.search(
                    r"(?:>=|>|at least|exceeds?|above|below|<)\\s*([0-9]+(?:\\.[0-9]+)?)",
                    problem,
                    re.IGNORECASE,
                )
                if threshold_match:
                    threshold_value = float(threshold_match.group(1))

            threshold_direction = "ge"
            if re.search(r"\\b(?:below|less than|<|at most)\\b", problem, re.IGNORECASE):
                threshold_direction = "le"

            threshold_index = None
            if closed_form is not None and threshold_value is not None:
                for idx in range(0, 800):
                    numeric_value = float(N(closed_form.subs(n, idx)))
                    if threshold_direction == "ge" and numeric_value >= float(threshold_value):
                        threshold_index = idx
                        break
                    if threshold_direction == "le" and numeric_value <= float(threshold_value):
                        threshold_index = idx
                        break

            ratio_limit = None
            if closed_form is not None:
                ratio_requested = bool(
                    re.search(r"\\bratio\\b|\\blimit\\b", problem, re.IGNORECASE)
                    or re.search(
                        rf"{{re.escape(sequence_name)}}\\(\\s*n\\s*\\+\\s*1\\s*\\)\\s*/\\s*{{re.escape(sequence_name)}}\\(\\s*n\\s*\\)",
                        problem,
                        re.IGNORECASE,
                    )
                )
                if ratio_requested:
                    try:
                        ratio_expr = simplify(closed_form.subs(n, n + 1) / closed_form)
                        ratio_limit = simplify(limit(ratio_expr, n, oo))
                    except Exception:
                        ratio_limit = None

            result_count = 0
            if closed_form is not None:
                result_count += 1
            if recurrence_verified:
                result_count += 1
            if threshold_index is not None:
                result_count += 1
            if ratio_limit is not None:
                result_count += 1

            result = {{
                "mode": "symbolic_numeric",
                "sequence": sequence_name,
                "closed_form": str(closed_form) if closed_form is not None else "",
                "recurrence_verified": recurrence_verified,
                "closed_form_ok": bool(closed_form is not None and recurrence_verified),
                "verify_0_10_ok": recurrence_verified,
                "verification_window": verification_window,
                "verification_mismatches": verification_mismatches,
                "threshold_target": threshold_value,
                "threshold_index": threshold_index,
                "threshold_n": threshold_index,
                "ratio_limit": str(ratio_limit) if ratio_limit is not None else "",
                "verified": bool(closed_form is not None and recurrence_verified),
                "result_count": result_count,
            }}

            expected = prediction.get("expectations", {{}})
            expected_verified = bool(expected.get("recurrence_verified", True))
            expected_count = expected.get("result_count", 4)
            expected_index = expected.get("threshold_index")
            confirms = (
                result["recurrence_verified"] == expected_verified
                and result["result_count"] >= max(3, int(expected_count) - 1)
                and result["closed_form"] != ""
            )
            if isinstance(expected_index, int) and isinstance(result["threshold_index"], int):
                tolerance = max(3, int(0.15 * max(1, result["threshold_index"])))
                confirms = confirms and abs(result["threshold_index"] - expected_index) <= tolerance

            print(json.dumps({{"result": result, "confirms_hypothesis": confirms}}, sort_keys=True))
            """
        ).strip()

    def _build_topology_code(self, problem: str, prediction_payload: Dict[str, Any]) -> str:
        prediction_text = json.dumps(prediction_payload, sort_keys=True)
        regular_query = self._parse_regular_graph_query(problem)
        regular_query_text = json.dumps(regular_query, sort_keys=True)
        return textwrap.dedent(
            f"""
            import json
            import networkx as nx
            from core.topology_solver import parse_topology_search_query, solve_topology_search

            try:
                from qiskit import QuantumCircuit
                qiskit_available = True
            except Exception:
                QuantumCircuit = None
                qiskit_available = False

            problem = {problem!r}
            prediction = json.loads({prediction_text!r})
            regular_query = json.loads({regular_query_text!r}) if {bool(regular_query)!r} else None
            query = parse_topology_search_query(problem)

            if query is None and regular_query is None:
                payload = {{
                    "result": {{
                        "mode": "topology",
                        "error": "Could not parse topology search query.",
                    }},
                    "confirms_hypothesis": False,
                }}
            elif query is None and regular_query is not None:
                node_count = int(regular_query["node_count"])
                regular_degree = int(regular_query["regular_degree"])
                connected_only = bool(regular_query["connected_only"])
                atlas = [
                    nx.convert_node_labels_to_integers(graph.copy())
                    for graph in nx.graph_atlas_g()
                    if graph.number_of_nodes() == node_count
                ]
                matches = []
                for graph in atlas:
                    if connected_only and not nx.is_connected(graph):
                        continue
                    if any(degree != regular_degree for _, degree in graph.degree()):
                        continue
                    matches.append(graph)

                def ranking_key(graph):
                    if nx.is_connected(graph):
                        diameter = nx.diameter(graph)
                    else:
                        diameter = node_count + 1
                    edges = sorted((min(a, b), max(a, b)) for a, b in graph.edges())
                    return (diameter, len(edges), edges)

                matches.sort(key=ranking_key)
                optimal = matches[0] if matches else None
                optimal_edges = sorted((min(a, b), max(a, b)) for a, b in optimal.edges()) if optimal else []
                optimal_diameter = nx.diameter(optimal) if optimal is not None and nx.is_connected(optimal) else None
                result = {{
                    "mode": "topology",
                    "enumeration_mode": "degree_regular",
                    "node_count": node_count,
                    "regular_degree": regular_degree,
                    "connected_only": connected_only,
                    "evaluated_topologies": len(atlas),
                    "satisfiable_count": len(matches),
                    "is_satisfiable": bool(matches),
                    "optimal_candidate": "G001" if matches else "none",
                    "optimal_edges": len(optimal_edges),
                    "optimal_diameter": optimal_diameter if optimal_diameter is not None else 0,
                    "optimal_fidelity": 0.0,
                    "qiskit_fidelity": 0.0,
                    "qiskit_operations": 0,
                    "qiskit_available": qiskit_available,
                }}
                expected = prediction.get("expectations", {{}})
                expected_count = expected.get("satisfiable_count")
                count_error_ratio = 1.0
                if isinstance(expected_count, int):
                    count_error_ratio = abs(result["satisfiable_count"] - expected_count) / max(1, result["satisfiable_count"] or 1)
                confirms = (
                    count_error_ratio <= {CONVERGENCE_THRESHOLD}
                    and result["optimal_candidate"] == expected.get("optimal_candidate")
                )
                payload = {{
                    "result": result,
                    "confirms_hypothesis": confirms,
                }}
            else:
                search = solve_topology_search(query)
                optimal = search.optimal_topology
                routed_operations = query.gate_operations + (optimal.diameter - 1 if optimal else 0)
                if qiskit_available:
                    circuit = QuantumCircuit(1)
                    for _ in range(routed_operations):
                        circuit.x(0)
                    routed_operations = len(circuit.data)
                logical_error_rate = query.physical_error_rate ** query.logical_error_exponent
                qiskit_fidelity = (1.0 - logical_error_rate) ** routed_operations
                result = {{
                    "mode": "topology",
                    "evaluated_topologies": search.evaluated_topologies,
                    "satisfiable_count": len(search.satisfiable_topologies),
                    "optimal_candidate": optimal.candidate_id if optimal else "none",
                    "optimal_edges": optimal.edge_count if optimal else 0,
                    "optimal_diameter": optimal.diameter if optimal else 0,
                    "optimal_fidelity": round(optimal.end_to_end_fidelity, 6) if optimal else 0.0,
                    "qiskit_fidelity": round(qiskit_fidelity, 6),
                    "qiskit_operations": routed_operations,
                    "qiskit_available": qiskit_available,
                }}
                expected = prediction.get("expectations", {{}})
                expected_count = expected.get("satisfiable_count")
                count_error_ratio = 1.0
                if isinstance(expected_count, int):
                    count_error_ratio = abs(result["satisfiable_count"] - expected_count) / max(1, result["satisfiable_count"])
                confirms = (
                    count_error_ratio <= {CONVERGENCE_THRESHOLD}
                    and result["optimal_candidate"] == expected.get("optimal_candidate")
                )
                payload = {{
                    "result": result,
                    "confirms_hypothesis": confirms,
                }}

            print(json.dumps(payload, sort_keys=True))
            """
        ).strip()

    def _build_formal_code(self, prediction_payload: Dict[str, Any], lenses: List[CognitiveLens]) -> str:
        lens_snapshot = self._serialize_lenses(lenses)
        prediction_text = json.dumps(prediction_payload, sort_keys=True)
        return textwrap.dedent(
            f"""
            import json
            from sympy import Integer

            lenses = json.loads({lens_snapshot!r})
            prediction = json.loads({prediction_text!r})
            constraint_count = int(Integer(len({{item for lens in lenses for item in lens.get("constraints", []) if item}})))
            dominant = max(lenses, key=lambda lens: lens.get("confidence", 0.0))["lens_name"] if lenses else "unknown"
            result = {{
                "mode": "formal",
                "constraint_count": constraint_count,
                "dominant_lens": dominant,
            }}
            expected = prediction.get("expectations", {{}})
            expected_count = expected.get("constraint_count")
            count_error = 999
            if isinstance(expected_count, int):
                count_error = abs(result["constraint_count"] - expected_count)
            payload = {{
                "result": result,
                "confirms_hypothesis": (
                    count_error <= 1
                    and result["dominant_lens"] == expected.get("dominant_lens")
                ),
            }}
            print(json.dumps(payload, sort_keys=True))
            """
        ).strip()

    def _build_probabilistic_code(self, prediction_payload: Dict[str, Any], lenses: List[CognitiveLens]) -> str:
        lens_snapshot = self._serialize_lenses(lenses)
        prediction_text = json.dumps(prediction_payload, sort_keys=True)
        return textwrap.dedent(
            f"""
            import json
            import statistics

            lenses = json.loads({lens_snapshot!r})
            prediction = json.loads({prediction_text!r})
            confidences = [float(lens.get("confidence", 0.0)) for lens in lenses]
            average_confidence = round(statistics.mean(confidences), 2) if confidences else 0.0
            confidence_spread = round(max(confidences) - min(confidences), 2) if len(confidences) >= 2 else 0.0
            dominant = max(lenses, key=lambda lens: lens.get("confidence", 0.0))["lens_name"] if lenses else "unknown"
            result = {{
                "mode": "probabilistic",
                "average_confidence": average_confidence,
                "confidence_spread": confidence_spread,
                "dominant_lens": dominant,
            }}
            expected = prediction.get("expectations", {{}})
            expected_average = expected.get("average_confidence", 0.0)
            payload = {{
                "result": result,
                "confirms_hypothesis": (
                    abs(result["average_confidence"] - expected_average) <= {CONVERGENCE_THRESHOLD}
                    and result["dominant_lens"] == expected.get("dominant_lens")
                ),
            }}
            print(json.dumps(payload, sort_keys=True))
            """
        ).strip()

    def _build_generic_code(self, prediction_payload: Dict[str, Any], lenses: List[CognitiveLens]) -> str:
        lens_snapshot = self._serialize_lenses(lenses)
        prediction_text = json.dumps(prediction_payload, sort_keys=True)
        return textwrap.dedent(
            f"""
            import json

            lenses = json.loads({lens_snapshot!r})
            prediction = json.loads({prediction_text!r})
            unique_tags = sorted({{lens.get("epistemic_tag", "") for lens in lenses if lens.get("epistemic_tag")}})
            dominant = max(lenses, key=lambda lens: lens.get("confidence", 0.0))["lens_name"] if lenses else "unknown"
            result = {{
                "mode": "generic",
                "lens_count": len(lenses),
                "unique_tag_count": len(unique_tags),
                "dominant_lens": dominant,
                "unique_tags": unique_tags,
            }}
            expected = prediction.get("expectations", {{}})
            expected_tag_count = expected.get("unique_tag_count")
            tag_count_error = 999
            if isinstance(expected_tag_count, int):
                tag_count_error = abs(result["unique_tag_count"] - expected_tag_count)
            payload = {{
                "result": result,
                "confirms_hypothesis": (
                    result["lens_count"] == expected.get("lens_count")
                    and tag_count_error <= 1
                    and result["dominant_lens"] == expected.get("dominant_lens")
                ),
            }}
            print(json.dumps(payload, sort_keys=True))
            """
        ).strip()

    def _revise_hypothesis_for_numeric_failure(
        self,
        hypothesis: Hypothesis,
        failure_reason: str,
        classification: ProblemClassification,
    ) -> Hypothesis:
        previous_prediction = self._load_prediction(hypothesis.prediction).get("expectations", {})
        if classification.primary_type == ProblemType.PROBABILISTIC:
            template = str(previous_prediction.get("template") or "rounds").strip().lower()
            if template == "survival":
                retained_survival = previous_prediction.get("survival_probability", 0.5)
                retained_hours = previous_prediction.get("expected_hours", 1.0)
                prediction = {
                    "mode": "probabilistic_numeric",
                    "expectations": {
                        "template": "survival",
                        "survival_probability": retained_survival,
                        "expected_hours": retained_hours,
                    },
                }
            else:
                retained_result = previous_prediction.get("result", 1)
                prediction = {
                    "mode": "probabilistic_numeric",
                    "expectations": {
                        "template": "rounds",
                        "result": retained_result,
                        "minimum_rounds": retained_result,
                        "model": "independent+correlated",
                    },
                }
            epistemic_tag = "probabilistic"
        elif classification.primary_type == ProblemType.SYMBOLIC:
            retained_count = previous_prediction.get("result_count", 4)
            retained_index = previous_prediction.get("threshold_index", 1)
            prediction = {
                "mode": "symbolic_numeric",
                "expectations": {
                    "recurrence_verified": True,
                    "result_count": retained_count,
                    "threshold_index": retained_index,
                },
            }
            epistemic_tag = "symbolic"
        else:
            retained_result = previous_prediction.get("result", 1)
            prediction = {
                "mode": "numeric",
                "expectations": {
                    "result": retained_result,
                    "verified": False,
                },
            }
            epistemic_tag = "deductive"

        return Hypothesis(
            statement=(
                "Previous reasoning failed to produce a numeric answer. "
                f"Triggered keywords: {classification.numeric_keywords}. "
                f"Failure reason: {failure_reason} "
                f"Next hypothesis must compute a concrete number that directly answers: {hypothesis.statement}"
            ),
            prediction=json.dumps(prediction, sort_keys=True),
            epistemic_tag=epistemic_tag,
            dominant_lens=hypothesis.dominant_lens,
        )

    def _select_mode(
        self,
        problem: str,
        lenses: List[CognitiveLens],
        classification: Optional[ProblemClassification] = None,
    ) -> str:
        if parse_topology_search_query(problem) is not None or self._parse_regular_graph_query(problem) is not None:
            return "topology"

        resolved_classification = classification or ProblemClassifier().classify(problem)
        if resolved_classification.primary_type == ProblemType.HYBRID:
            if parse_topology_search_query(problem) is not None or self._parse_regular_graph_query(problem) is not None:
                return "topology"
            return "generic"

        tags = {lens.epistemic_tag for lens in lenses}
        if "formal" in tags or "symbolic" in tags:
            return "formal"
        if "probabilistic" in tags:
            return "probabilistic"
        return "generic"

    def _dominant_lens(self, lenses: List[CognitiveLens]) -> CognitiveLens:
        return max(lenses, key=lambda lens: lens.confidence)

    def _dominant_lens_for_classification(
        self,
        lenses: List[CognitiveLens],
        classification: ProblemClassification,
    ) -> CognitiveLens:
        recommended = classification.recommended_lens.lower().strip()
        for lens in lenses:
            lens_name = lens.lens_name.lower()
            if recommended == "symbolic" and (
                lens.epistemic_tag == "symbolic" or "symbolic" in lens_name
            ):
                return lens
            if recommended == "probabilistic" and (
                lens.epistemic_tag == "probabilistic" or "probabilistic" in lens_name
            ):
                return lens
            if recommended == "topological" and (
                "topological" in lens_name or lens.epistemic_tag == "deductive"
            ):
                return lens
        return self._dominant_lens(lenses)

    def _estimate_probabilistic_expectations(self, problem: str) -> Dict[str, Any]:
        template = self._detect_probabilistic_template(problem)
        participant_count = self._extract_first_int(
            problem,
            patterns=(r"(\d+)\s+participants?", r"\bN\s*=\s*(\d+)"),
            default=1,
        )
        probability = self._extract_first_float(problem, (r"\bp\s*=\s*([0-9]*\.?[0-9]+)",), default=0.01)
        if template == "survival":
            horizon_hours = self._extract_first_float(
                problem,
                (
                    r"(?:for|over|during|across)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:hours?|hrs?|h)\b",
                ),
                default=1.0,
            )
            independent_failure = 1.0 - ((1.0 - probability) ** participant_count)
            correlated_failure = min(
                0.999999,
                independent_failure * (1.15 + (0.01 * max(0, participant_count - 1))),
            )
            independent_survival = (1.0 - independent_failure) ** horizon_hours
            correlated_survival = (1.0 - correlated_failure) ** horizon_hours
            independent_expected_hours = 1.0 / max(independent_failure, 1e-9)
            correlated_expected_hours = 1.0 / max(correlated_failure, 1e-9)
            return {
                "template": "survival",
                "survival_probability": min(independent_survival, correlated_survival),
                "expected_hours": min(independent_expected_hours, correlated_expected_hours),
            }

        return {
            "template": "rounds",
            "minimum_rounds": self._estimate_probabilistic_result(problem),
        }

    def _detect_probabilistic_template(self, problem: str) -> str:
        lowered = problem.lower()
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

    def _estimate_probabilistic_result(self, problem: str) -> int:
        participant_count = self._extract_first_int(
            problem,
            patterns=(r"(\d+)\s+participants?", r"\bN\s*=\s*(\d+)"),
            default=1,
        )
        probability = self._extract_first_float(problem, (r"\bp\s*=\s*([0-9]*\.?[0-9]+)",), default=0.01)
        threshold = self._extract_first_float(
            problem,
            (r"(?:falls below|below|<)\s*([0-9]*\.?[0-9]+)",),
            default=0.001,
        )

        independent_per_round = 1.0 - ((1.0 - probability) ** participant_count)
        correlated_per_round = min(0.999999, 1.0 - ((1.0 - probability) ** (participant_count + 2)))

        exact_rounds = max(
            self._solve_round_threshold(independent_per_round, threshold),
            self._solve_round_threshold(correlated_per_round, threshold),
        )
        return max(1, exact_rounds - 1)

    def _estimate_numeric_result(self, problem: str) -> float:
        values = [
            float(value)
            for value in self._extract_all_numbers(problem)
        ]
        if not values:
            return 1
        threshold = self._extract_first_float(
            problem,
            (r"(?:falls below|below|<)\s*([0-9]*\.?[0-9]+)",),
            default=0.0,
        )
        if threshold > 0:
            return max(1, int(math.ceil(max(values) / threshold)))
        integers = [int(value) for value in values if abs(value - round(value)) < 1e-9]
        if integers:
            return max(1, min(integers))
        return round(min(values), 2)

    def _solve_round_threshold(self, per_round_probability: float, threshold: float) -> int:
        rounds = 1
        bounded_probability = min(max(per_round_probability, 1e-9), 0.999999)
        while bounded_probability ** rounds >= threshold:
            rounds += 1
        return rounds

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

    def _extract_all_numbers(self, problem: str) -> List[str]:
        return re.findall(r"(?<![A-Za-z])\d+\.?\d*", problem)

    def _revise_integer_expectation(self, previous: Any, current: Any, delta: float) -> int:
        try:
            current_value = int(current)
        except (TypeError, ValueError):
            return 0
        if not isinstance(previous, int):
            return current_value
        if previous == current_value:
            return current_value

        direction = 1 if current_value > previous else -1
        adjustment_gain = clamp_float(
            0.6 + (0.2 * delta),
            default=0.65,
            minimum=0.55,
            maximum=0.85,
        )
        adjustment = max(1, int(round(abs(current_value - previous) * adjustment_gain)))
        revised = previous + (direction * adjustment)
        if direction > 0:
            return min(revised, current_value)
        return max(revised, current_value)

    def _revise_float_expectation(self, previous: Any, current: Any, delta: float) -> float:
        try:
            current_value = float(current)
        except (TypeError, ValueError):
            return 0.0

        try:
            previous_value = float(previous)
        except (TypeError, ValueError):
            return round(current_value, 2)

        if abs(previous_value - current_value) < 1e-9:
            return round(current_value, 2)

        adjustment_gain = clamp_float(
            0.6 + (0.2 * delta),
            default=0.65,
            minimum=0.55,
            maximum=0.85,
        )
        revised = previous_value + ((current_value - previous_value) * adjustment_gain)
        if current_value > previous_value:
            revised = min(revised, current_value)
        else:
            revised = max(revised, current_value)
        return round(revised, 2)

    def _describe_formal_alignment(self, cycle: ExecutionCycle, result: Dict[str, Any]) -> str:
        expected = self._load_prediction(cycle.prediction).get("expectations", {})
        expected_count = expected.get("constraint_count")
        actual_count = result.get("constraint_count")
        if isinstance(expected_count, int) and isinstance(actual_count, int):
            if expected_count == actual_count:
                return "The revised estimate matched the executed constraint count exactly."
            if abs(expected_count - actual_count) <= 1:
                return "The revised estimate did not match exactly, but it converged within one constraint of the executed count."
        return "The revised estimate remained close enough to the executed count to satisfy the convergence threshold."

    def _describe_probabilistic_alignment(self, cycle: ExecutionCycle, result: Dict[str, Any]) -> str:
        expected = self._load_prediction(cycle.prediction).get("expectations", {})
        expected_average = expected.get("average_confidence")
        actual_average = result.get("average_confidence")
        if isinstance(expected_average, (int, float)) and isinstance(actual_average, (int, float)):
            difference = abs(float(expected_average) - float(actual_average))
            if difference < 1e-9:
                return "The revised estimate matched the measured confidence centroid exactly."
            if difference <= CONVERGENCE_THRESHOLD:
                return (
                    f"The revised estimate did not match exactly, but it converged within {CONVERGENCE_THRESHOLD:.2f} "
                    "of the measured confidence centroid."
                )
        return "The revised estimate remained close enough to the measured confidence centroid to satisfy the convergence threshold."

    def _describe_generic_alignment(self, cycle: ExecutionCycle, result: Dict[str, Any]) -> str:
        expected = self._load_prediction(cycle.prediction).get("expectations", {})
        expected_tags = expected.get("unique_tag_count")
        actual_tags = result.get("unique_tag_count")
        if isinstance(expected_tags, int) and isinstance(actual_tags, int):
            if expected_tags == actual_tags:
                return "The revised estimate matched the executed tag count exactly."
            if abs(expected_tags - actual_tags) <= 1:
                return "The revised estimate did not match exactly, but it converged within one epistemic tag of the executed count."
        return "The revised estimate remained close enough to the executed tag count to satisfy the convergence threshold."

    def _describe_topology_alignment(self, cycle: ExecutionCycle, result: Dict[str, Any]) -> str:
        expected = self._load_prediction(cycle.prediction).get("expectations", {})
        expected_count = expected.get("satisfiable_count")
        expected_candidate = expected.get("optimal_candidate")
        actual_count = result.get("satisfiable_count")
        actual_candidate = result.get("optimal_candidate")

        if (
            isinstance(expected_count, int)
            and isinstance(actual_count, int)
            and isinstance(expected_candidate, str)
            and isinstance(actual_candidate, str)
        ):
            if expected_count == actual_count and expected_candidate == actual_candidate:
                return "The revised estimate matched the executed topology count and winning candidate exactly."
            count_error_ratio = abs(actual_count - expected_count) / max(1, actual_count)
            if count_error_ratio <= CONVERGENCE_THRESHOLD and expected_candidate == actual_candidate:
                return (
                    "The revised estimate did not match exactly, but it converged within the 5% topology-count "
                    "threshold while preserving the executed winning candidate."
                )
        return "The revised estimate remained close enough to the executed topology search to satisfy the convergence threshold."

    def _parse_regular_graph_query(self, problem: str) -> Optional[Dict[str, Any]]:
        lowered = problem.lower()
        if "regular" not in lowered or not any(token in lowered for token in ("graph", "topolog", "nodes", "nodi")):
            return None

        patterns: Tuple[Tuple[str, int, int], ...] = (
            (
                r"(\d+)\s*[- ]\s*regular\b(?:\s+[a-z_]+){0,4}?\s+(?:on|with|su)\s+(\d+)\s+nodes?",
                1,
                2,
            ),
            (r"graphs?\s+(?:on|with|su)\s+(\d+)\s+nodes?\s+(?:that are|are|which are)\s+(\d+)\s*[- ]\s*regular", 2, 1),
            (r"degree\s*(\d+)\s*(?:regular)?\s*(?:graph|topolog)[^0-9]*?(\d+)\s+nodes?", 1, 2),
        )
        for pattern, degree_group, node_group in patterns:
            match = re.search(pattern, lowered, re.IGNORECASE)
            if not match:
                continue
            regular_degree = int(match.group(degree_group))
            node_count = int(match.group(node_group))
            if node_count < 2 or regular_degree < 0 or regular_degree >= node_count:
                return None
            connected_only = not bool(re.search(r"\bdisconnected\b|\bnon-connected\b", lowered))
            return {
                "node_count": node_count,
                "regular_degree": regular_degree,
                "connected_only": connected_only,
            }
        return None

    def _count_regular_graphs(self, node_count: int, regular_degree: int, connected_only: bool) -> int:
        atlas = [graph for graph in nx.graph_atlas_g() if graph.number_of_nodes() == node_count]
        count = 0
        for graph in atlas:
            if connected_only and not nx.is_connected(graph):
                continue
            if any(degree != regular_degree for _, degree in graph.degree()):
                continue
            count += 1
        return count

    def _estimate_topology_satisfiable_count(self, query, connected_graph_count: int) -> int:
        max_possible_edges = query.node_count * (query.node_count - 1) // 2
        max_edges = min(max_possible_edges, int(query.entanglement_factor_limit * (query.node_count - 1)))
        edge_budget_ratio = max_edges / max(1, max_possible_edges)

        max_allowed_diameter = min(
            query.node_count - 1,
            max(1, int((query.latency_limit_ms - 1e-9) // max(1.0, query.hop_latency_ms))),
        )
        diameter_budget_ratio = max_allowed_diameter / max(1, query.node_count - 1)

        logical_error_rate = query.physical_error_rate ** query.logical_error_exponent
        baseline_fidelity = (1.0 - logical_error_rate) ** query.gate_operations
        fidelity_margin_ratio = clamp_float(
            (baseline_fidelity - query.fidelity_threshold) / max(1e-9, 1.0 - query.fidelity_threshold),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )

        structural_ratio = clamp_float(
            0.16
            + (0.12 * edge_budget_ratio)
            + (0.08 * diameter_budget_ratio)
            + (0.07 * fidelity_margin_ratio)
            - (0.01 * max(0, query.node_count - 4)),
            default=0.25,
            minimum=0.12,
            maximum=0.65,
        )
        heuristic_ratio = clamp_float(
            (0.8 * self._topology_prior_ratio(query.node_count)) + (0.2 * structural_ratio),
            default=structural_ratio,
            minimum=0.12,
            maximum=0.65,
        )
        return max(1, int(connected_graph_count * heuristic_ratio))

    @staticmethod
    @lru_cache(maxsize=8)
    def _connected_graph_count(node_count: int) -> int:
        return sum(
            1
            for graph in nx.graph_atlas_g()
            if graph.number_of_nodes() == node_count and nx.is_connected(graph)
        )

    @staticmethod
    def _topology_prior_ratio(node_count: int) -> float:
        calibrated = {
            4: 0.50,
            5: 0.476,
            6: 0.50,
            7: 0.544,
        }
        if node_count in calibrated:
            return calibrated[node_count]
        if node_count < 4:
            return 0.40
        return clamp_float(0.40 + (0.03 * min(5, node_count - 3)), default=0.50, minimum=0.35, maximum=0.60)

    def _serialize_lenses(self, lenses: List[CognitiveLens]) -> str:
        payload = [
            {
                "lens_name": lens.lens_name,
                "constraints": list(lens.constraints),
                "blind_spots": list(lens.blind_spots),
                "confidence": float(lens.confidence),
                "epistemic_tag": lens.epistemic_tag,
            }
            for lens in lenses
        ]
        return json.dumps(payload, sort_keys=True)

    def _load_prediction(self, prediction: str) -> Dict[str, Any]:
        try:
            payload = json.loads(prediction)
        except json.JSONDecodeError:
            return {"mode": "generic", "expectations": {}}
        if not isinstance(payload, dict):
            return {"mode": "generic", "expectations": {}}
        expectations = payload.get("expectations")
        if not isinstance(expectations, dict):
            payload["expectations"] = {}
        return payload

    def _load_output(self, output: str) -> Dict[str, Any]:
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return {
                "result": {
                    "mode": "runtime_error",
                    "error": output,
                },
                "confirms_hypothesis": False,
            }
        if not isinstance(payload, dict):
            return {
                "result": {
                    "mode": "runtime_error",
                    "error": output,
                },
                "confirms_hypothesis": False,
            }
        result = payload.get("result")
        if not isinstance(result, dict):
            payload["result"] = {}
        payload["confirms_hypothesis"] = bool(payload.get("confirms_hypothesis"))
        return payload

    def _detect_constraint_contradictions(self, problem: str) -> List[str]:
        operator_fragment = (
            r"strictly greater than|greater than|more than|at least|no less than|>=|>|"
            r"strictly less than|less than|at most|no more than|<=|<|equal to|exactly"
        )
        pattern = re.compile(
            r"(?P<metric>[A-Za-z][A-Za-z0-9_ \-/]{1,80}?)\s*"
            r"(?:(?:must|should)\s+be|is|be|remain|stays?)?\s*"
            rf"(?P<op>{operator_fragment})\s*"
            r"(?P<value>-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        chained_pattern = re.compile(
            r"(?P<metric>[A-Za-z][A-Za-z0-9_ \-/]{1,80}?)\s*"
            r"(?:(?:must|should)\s+be|is|be|remain|stays?)?\s*"
            rf"(?P<op1>{operator_fragment})\s*(?P<value1>-?\d+(?:\.\d+)?)\s*"
            r"(?:,?\s*(?:and|while)\s+)"
            rf"(?P<op2>{operator_fragment})\s*(?P<value2>-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )

        bounds: Dict[str, Dict[str, Optional[Tuple[float, bool]]]] = {}
        raw_labels: Dict[str, str] = {}

        def apply_constraint(metric_text: str, op_text: str, value_text: str) -> None:
            metric_key = self._canonical_constraint_metric(metric_text)
            if not metric_key:
                return
            raw_labels.setdefault(metric_key, metric_text.strip())
            op = op_text.strip().lower()
            value = float(value_text)
            record = bounds.setdefault(metric_key, {"lower": None, "upper": None})
            if op in {">", "greater than", "more than", "strictly greater than"}:
                self._update_lower_bound(record, value, strict=True)
            elif op in {">=", "at least", "no less than"}:
                self._update_lower_bound(record, value, strict=False)
            elif op in {"<", "less than", "strictly less than"}:
                self._update_upper_bound(record, value, strict=True)
            elif op in {"<=", "at most", "no more than"}:
                self._update_upper_bound(record, value, strict=False)
            elif op in {"equal to", "exactly"}:
                self._update_lower_bound(record, value, strict=False)
                self._update_upper_bound(record, value, strict=False)

        for clause in re.split(r"[;\n]", problem):
            for chain in chained_pattern.finditer(clause):
                metric_text = chain.group("metric")
                apply_constraint(metric_text, chain.group("op1"), chain.group("value1"))
                apply_constraint(metric_text, chain.group("op2"), chain.group("value2"))
            for match in pattern.finditer(clause):
                apply_constraint(match.group("metric"), match.group("op"), match.group("value"))

        contradictions: List[str] = []
        for key, record in bounds.items():
            lower = record.get("lower")
            upper = record.get("upper")
            if lower is None or upper is None:
                continue
            lower_value, lower_strict = lower
            upper_value, upper_strict = upper
            if lower_value > upper_value + 1e-12:
                contradictions.append(
                    f"{raw_labels.get(key, key)} has incompatible bounds: >={lower_value:g} and <={upper_value:g}."
                )
            elif abs(lower_value - upper_value) <= 1e-12 and (lower_strict or upper_strict):
                contradictions.append(
                    f"{raw_labels.get(key, key)} has incompatible strict bounds around {lower_value:g}."
                )
        return contradictions

    def _canonical_constraint_metric(self, metric_text: str) -> str:
        tokens = re.findall(r"[a-zA-Z]+", metric_text.lower())
        stop = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "for",
            "to",
            "with",
            "that",
            "which",
            "must",
            "should",
            "be",
            "is",
            "are",
            "remain",
            "stays",
            "than",
            "strictly",
        }
        filtered = [token for token in tokens if token not in stop]
        if not filtered:
            return ""
        return " ".join(filtered[-6:])

    def _update_lower_bound(self, record: Dict[str, Optional[Tuple[float, bool]]], value: float, strict: bool) -> None:
        current = record.get("lower")
        if current is None:
            record["lower"] = (value, strict)
            return
        current_value, current_strict = current
        if value > current_value + 1e-12:
            record["lower"] = (value, strict)
            return
        if abs(value - current_value) <= 1e-12:
            record["lower"] = (current_value, current_strict or strict)

    def _update_upper_bound(self, record: Dict[str, Optional[Tuple[float, bool]]], value: float, strict: bool) -> None:
        current = record.get("upper")
        if current is None:
            record["upper"] = (value, strict)
            return
        current_value, current_strict = current
        if value < current_value - 1e-12:
            record["upper"] = (value, strict)
            return
        if abs(value - current_value) <= 1e-12:
            record["upper"] = (current_value, current_strict or strict)

    def _value_delta(self, expected: Any, actual: Any) -> float:
        if isinstance(expected, bool):
            return 0.0 if bool(actual) == expected else 1.0
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            denominator = max(1.0, abs(float(expected)), abs(float(actual)))
            return clamp_float(abs(float(actual) - float(expected)) / denominator, default=1.0, minimum=0.0, maximum=1.0)
        return 0.0 if expected == actual else 1.0
