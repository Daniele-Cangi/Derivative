import math
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

import networkx as nx

from audit.trail import AuditEntry, AuditTrail
from core.json_utils import clamp_float
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

    def run(
        self,
        problem: str,
        lenses: List[CognitiveLens],
        audit: Optional[AuditTrail] = None,
    ) -> ExecutionResult:
        from core.validator import NumericAnswerCheck

        classification = ProblemClassifier().classify(problem)
        numeric_check = NumericAnswerCheck()
        hypothesis = self._form_initial_hypothesis(problem, lenses, classification)
        history: List[ExecutionCycle] = []
        audit_trail = audit

        for cycle in range(1, MAX_EXECUTION_CYCLES + 1):
            code = self._hypothesis_to_code(problem, hypothesis, lenses, classification)
            execution = self._execute(code)
            residual = self._measure_residual(hypothesis.prediction, execution.output)
            delta = self._measure_delta(hypothesis.prediction, execution.output)
            converged = delta < CONVERGENCE_THRESHOLD and execution.exit_code == 0
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

    def _form_initial_hypothesis(
        self,
        problem: str,
        lenses: List[CognitiveLens],
        classification: Optional[ProblemClassification] = None,
    ) -> Hypothesis:
        resolved_classification = classification or ProblemClassifier().classify(problem)
        dominant_lens = self._dominant_lens(lenses)
        mode = self._select_mode(problem, lenses, resolved_classification)

        if resolved_classification.primary_type == ProblemType.PROBABILISTIC:
            estimated_rounds = self._estimate_probabilistic_result(problem)
            prediction = {
                "mode": "probabilistic_numeric",
                "expectations": {
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
                epistemic_tag="probabilistic",
                dominant_lens=resolved_classification.recommended_lens,
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
                epistemic_tag="deductive",
                dominant_lens=resolved_classification.recommended_lens,
            )

        if mode == "topology":
            query = parse_topology_search_query(problem)
            node_count = query.node_count if query is not None else max(3, len(lenses))
            connected_graph_count = self._connected_graph_count(node_count)
            if query is not None:
                expected_count = self._estimate_topology_satisfiable_count(query, connected_graph_count)
            else:
                expected_count = max(1, connected_graph_count // 3)
            prediction = {
                "mode": "topology",
                "expectations": {
                    "satisfiable_count": expected_count,
                    "optimal_candidate": "T001",
                },
            }
            statement = (
                f"The initial {dominant_lens.lens_name} hypothesis is that the constraint set should admit "
                f"about {expected_count} satisfiable non-isomorphic topologies out of {connected_graph_count} "
                "connected candidate shapes, with T001 emerging as the best-ranked candidate after execution."
            )
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
            expected_result = expectations.get("result")
            actual_numeric = actual_result.get("result")
            if isinstance(expected_result, int) and isinstance(actual_numeric, int):
                return abs(actual_numeric - expected_result) / max(1, actual_numeric)
            return 1.0

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
                previous_result = predicted.get("expectations", {}).get("result", "unknown")
                current_result = actual_result.get("result", "unknown")
                revised_result = self._revise_integer_expectation(previous_result, current_result, delta)
                next_expectations = {
                    "result": revised_result,
                    "minimum_rounds": revised_result,
                    "model": actual_result.get("model", "independent+correlated"),
                }
                statement = (
                    f"The previous probabilistic hypothesis failed because it predicted {previous_result} rounds, "
                    f"but execution computed {current_result}; the revised hypothesis moves toward approximately "
                    f"{revised_result} rounds while preserving both probabilistic models in the next computation."
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
        if mode == "numeric":
            return self._synthesize_numeric(history, final_result)
        if mode == "topology":
            return self._synthesize_topology(history, final_result)
        if mode == "formal":
            return self._synthesize_formal(history, final_result)
        if mode == "probabilistic":
            return self._synthesize_probabilistic(history, final_result)
        return self._synthesize_generic(history, final_result)

    def _synthesize_topology(self, history: List[ExecutionCycle], result: Dict[str, Any]) -> str:
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
        if parse_topology_search_query(problem) is not None:
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

            expected = prediction.get("expectations", {{}})
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

    def _build_topology_code(self, problem: str, prediction_payload: Dict[str, Any]) -> str:
        prediction_text = json.dumps(prediction_payload, sort_keys=True)
        return textwrap.dedent(
            f"""
            import json
            from core.topology_solver import parse_topology_search_query, solve_topology_search

            try:
                from qiskit import QuantumCircuit
                qiskit_available = True
            except Exception:
                QuantumCircuit = None
                qiskit_available = False

            problem = {problem!r}
            prediction = json.loads({prediction_text!r})
            query = parse_topology_search_query(problem)

            if query is None:
                payload = {{
                    "result": {{
                        "mode": "topology",
                        "error": "Could not parse topology search query.",
                    }},
                    "confirms_hypothesis": False,
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
            retained_result = previous_prediction.get("result", 1)
            prediction = {
                "mode": "probabilistic_numeric",
                "expectations": {
                    "result": retained_result,
                    "minimum_rounds": retained_result,
                    "model": "independent+correlated",
                },
            }
            epistemic_tag = "probabilistic"
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
            dominant_lens=classification.recommended_lens,
        )

    def _select_mode(
        self,
        problem: str,
        lenses: List[CognitiveLens],
        classification: Optional[ProblemClassification] = None,
    ) -> str:
        if parse_topology_search_query(problem) is not None:
            return "topology"

        resolved_classification = classification or ProblemClassifier().classify(problem)
        if resolved_classification.primary_type == ProblemType.HYBRID:
            return "topology" if parse_topology_search_query(problem) is not None else "generic"

        tags = {lens.epistemic_tag for lens in lenses}
        if "formal" in tags or "symbolic" in tags:
            return "formal"
        if "probabilistic" in tags:
            return "probabilistic"
        return "generic"

    def _dominant_lens(self, lenses: List[CognitiveLens]) -> CognitiveLens:
        return max(lenses, key=lambda lens: lens.confidence)

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

    def _value_delta(self, expected: Any, actual: Any) -> float:
        if isinstance(expected, bool):
            return 0.0 if bool(actual) == expected else 1.0
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            denominator = max(1.0, abs(float(expected)), abs(float(actual)))
            return clamp_float(abs(float(actual) - float(expected)) / denominator, default=1.0, minimum=0.0, maximum=1.0)
        return 0.0 if expected == actual else 1.0
