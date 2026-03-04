import json

from audit.trail import AuditTrail
from core.execution_loop import (
    CONVERGENCE_THRESHOLD,
    ExecutionCycle,
    Hypothesis,
    ExecutionLoop,
    ExecutionObservation,
)
from lenses.base import CognitiveLens


def build_exact_query(node_count: int) -> str:
    return f"""
Given a quantum error-corrected distributed system with N={node_count} nodes,
where each node maintains a logical qubit with physical error rate p=0.001,
find all satisfiable topological configurations such that:

1. End-to-end fidelity > 0.95 after 50 gate operations
2. No single node failure causes system-wide decoherence
3. Classical control latency between any two nodes < 50ms
4. Total entanglement overhead does not exceed 3x the minimum spanning tree

Which configurations exist? Which is optimal under simultaneous
novelty + feasibility scoring? Emit a verifiable circuit.
""".strip()


EXACT_QUERY = build_exact_query(4)
CRYPTO_QUERY = (
    "A cryptographic key exchange protocol uses 5 participants in a ring topology. "
    "Each participant i can only communicate directly with participant i+1 mod 5. "
    "Each communication channel has a probability of interception p=0.03. "
    "Find the minimum number of independent key-agreement rounds required so that the "
    "probability of at least one undetected interception across the full ring falls below 0.001. "
    "Then verify that your answer satisfies the constraint under both independent and correlated "
    "channel failure models."
)


def test_execution_loop_revises_and_logs_topology_cycles(tmp_path):
    loop = ExecutionLoop()
    audit = AuditTrail(log_file=str(tmp_path / "audit.json"))
    lenses = [
        CognitiveLens("Topological", "framing", ["constraint1"], ["spot1"], 0.9, "deductive"),
        CognitiveLens("Physical", "framing", ["constraint2"], ["spot2"], 0.8, "physical"),
        CognitiveLens("Quantum", "framing", ["constraint3"], ["spot3"], 0.85, "quantum"),
    ]

    result = loop.run(EXACT_QUERY, lenses, audit=audit)

    assert result.converged is True
    assert 2 <= result.cycles_used <= 5
    assert len(result.history) == result.cycles_used
    assert result.history[0].delta > CONVERGENCE_THRESHOLD
    assert result.history[0].hypothesis != result.history[1].hypothesis
    assert result.final_code
    assert result.final_output.startswith("{")
    assert result.final_prediction
    assert result.final_residual <= CONVERGENCE_THRESHOLD
    assert "qiskit-backed" in result.conclusion
    assert "|" not in result.conclusion

    trace = audit.get_full_trace()
    assert [entry.step_id for entry in trace] == [f"exec_cycle_{index}" for index in range(1, result.cycles_used + 1)]
    assert all(entry.execution_code for entry in trace)
    assert all(entry.execution_prediction for entry in trace)
    assert all(entry.execution_output for entry in trace)
    assert all(entry.execution_residual >= 0.0 for entry in trace)
    assert trace[0].execution_delta > CONVERGENCE_THRESHOLD


def test_execution_loop_computes_numeric_probabilistic_answer():
    loop = ExecutionLoop()
    lenses = [
        CognitiveLens("Probabilistic", "framing", ["constraint1"], ["spot1"], 0.9, "probabilistic"),
        CognitiveLens("Topological", "framing", ["constraint2"], ["spot2"], 0.7, "deductive"),
    ]

    result = loop.run(CRYPTO_QUERY, lenses)
    output = json.loads(result.final_output)
    computed = output["result"]

    assert result.converged is True
    assert result.cycles_used >= 2
    assert computed["mode"] == "probabilistic_numeric"
    assert computed["minimum_rounds"] == 5
    assert computed["independent_model_result"] == 4
    assert computed["correlated_model_result"] == 5
    assert "minimum of 5 rounds" in result.conclusion


def test_execution_loop_continues_after_numeric_check_failure(tmp_path):
    class StubExecutionLoop(ExecutionLoop):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def _form_initial_hypothesis(self, problem, lenses, classification=None) -> Hypothesis:
            return Hypothesis(
                statement="generic structural placeholder",
                prediction=json.dumps(
                    {
                        "mode": "generic",
                        "expectations": {
                            "lens_count": 2,
                            "unique_tag_count": 1,
                            "dominant_lens": "Probabilistic",
                        },
                    },
                    sort_keys=True,
                ),
                epistemic_tag="probabilistic",
                dominant_lens="Probabilistic",
            )

        def _execute(self, code: str) -> ExecutionObservation:
            self.calls += 1
            if self.calls == 1:
                return ExecutionObservation(
                    output=json.dumps(
                        {
                            "result": {
                                "mode": "generic",
                                "lens_count": 2,
                                "unique_tag_count": 1,
                                "dominant_lens": "Probabilistic",
                            },
                            "confirms_hypothesis": True,
                        },
                        sort_keys=True,
                    ),
                    exit_code=0,
                )
            return ExecutionObservation(
                output=json.dumps(
                    {
                        "result": {
                            "mode": "probabilistic_numeric",
                            "result": 1,
                            "minimum_rounds": 1,
                            "verified": True,
                            "threshold": 0.001,
                            "independent_model_result": 1,
                            "correlated_model_result": 1,
                            "independent_model_probability": 0.0008,
                            "correlated_model_probability": 0.0009,
                            "model": "independent+correlated",
                        },
                        "confirms_hypothesis": True,
                    },
                    sort_keys=True,
                ),
                exit_code=0,
            )

    loop = StubExecutionLoop()
    audit = AuditTrail(log_file=str(tmp_path / "audit.json"))
    lenses = [
        CognitiveLens("Probabilistic", "framing", ["constraint1"], ["spot1"], 0.9, "probabilistic"),
        CognitiveLens("Topological", "framing", ["constraint2"], ["spot2"], 0.7, "deductive"),
    ]

    result = loop.run(CRYPTO_QUERY, lenses, audit=audit)
    trace = audit.get_full_trace()

    assert result.converged is True
    assert result.cycles_used == 2
    assert result.history[0].delta == 1.0
    assert result.history[0].residual == 1.0
    assert "Previous reasoning failed to produce a numeric answer" in result.history[1].hypothesis
    assert "minimum of 1 rounds" in result.conclusion
    assert '"minimum_rounds": 1' in result.final_output
    assert [entry.step_id for entry in trace] == ["exec_cycle_1", "exec_cycle_2"]
    assert trace[0].execution_delta == 1.0
    assert trace[0].execution_residual == 1.0
    assert trace[1].execution_delta < CONVERGENCE_THRESHOLD
    assert trace[1].execution_prediction


def test_execution_loop_allows_hybrid_topology_convergence_with_numeric_gate(tmp_path):
    class HybridStubExecutionLoop(ExecutionLoop):
        def _form_initial_hypothesis(self, problem, lenses, classification=None) -> Hypothesis:
            return Hypothesis(
                statement="topology hypothesis",
                prediction=json.dumps(
                    {
                        "mode": "topology",
                        "expectations": {
                            "satisfiable_count": 2,
                            "optimal_candidate": "T001",
                        },
                    },
                    sort_keys=True,
                ),
                epistemic_tag="deductive",
                dominant_lens="Topological",
            )

        def _execute(self, code: str) -> ExecutionObservation:
            return ExecutionObservation(
                output=json.dumps(
                    {
                        "result": {
                            "mode": "topology",
                            "satisfiable_count": 2,
                            "evaluated_topologies": 4,
                            "optimal_candidate": "T001",
                            "optimal_diameter": 2,
                            "optimal_fidelity": 0.99995,
                            "qiskit_fidelity": 0.99995,
                            "qiskit_operations": 12,
                        },
                        "confirms_hypothesis": True,
                    },
                    sort_keys=True,
                ),
                exit_code=0,
            )

    loop = HybridStubExecutionLoop()
    audit = AuditTrail(log_file=str(tmp_path / "audit.json"))
    lenses = [
        CognitiveLens("Topological", "framing", ["constraint1"], ["spot1"], 0.9, "deductive"),
        CognitiveLens("Physical", "framing", ["constraint2"], ["spot2"], 0.8, "physical"),
        CognitiveLens("Quantum", "framing", ["constraint3"], ["spot3"], 0.85, "quantum"),
    ]

    result = loop.run(build_exact_query(4), lenses, audit=audit)
    trace = audit.get_full_trace()

    assert result.converged is True
    assert result.cycles_used == 1
    assert result.history[0].delta == 0.0
    assert result.history[0].residual == 0.0
    assert '"satisfiable_count": 2' in result.final_output
    assert "returned 2 satisfiable topology configuration(s)" in result.conclusion
    assert [entry.step_id for entry in trace] == ["exec_cycle_1"]
    assert trace[0].execution_delta == 0.0
    assert trace[0].validation_result == "converged"


def test_execution_loop_uses_informed_topology_seed():
    loop = ExecutionLoop()
    lenses = [
        CognitiveLens("Topological", "framing", ["constraint1"], ["spot1"], 0.9, "deductive"),
        CognitiveLens("Physical", "framing", ["constraint2"], ["spot2"], 0.8, "physical"),
        CognitiveLens("Quantum", "framing", ["constraint3"], ["spot3"], 0.85, "quantum"),
    ]

    hypothesis = loop._form_initial_hypothesis(EXACT_QUERY, lenses)
    prediction = json.loads(hypothesis.prediction)

    assert "out of" in hypothesis.statement
    assert 1 < prediction["expectations"]["satisfiable_count"] < 4


def test_execution_loop_calibrates_topology_seeds_across_n4_to_n7():
    loop = ExecutionLoop()
    lenses = [
        CognitiveLens("Topological", "framing", ["constraint1"], ["spot1"], 0.9, "deductive"),
        CognitiveLens("Physical", "framing", ["constraint2"], ["spot2"], 0.8, "physical"),
        CognitiveLens("Quantum", "framing", ["constraint3"], ["spot3"], 0.85, "quantum"),
    ]

    estimates = {}
    for node_count in range(4, 8):
        hypothesis = loop._form_initial_hypothesis(build_exact_query(node_count), lenses)
        prediction = json.loads(hypothesis.prediction)
        estimates[node_count] = prediction["expectations"]["satisfiable_count"]

    assert estimates[4] == 2
    assert estimates[5] == 9
    assert estimates[6] == 53
    assert estimates[7] == 431


def test_execution_loop_revises_topology_hypothesis_gradually():
    loop = ExecutionLoop()
    lenses = [
        CognitiveLens("Topological", "framing", ["constraint1"], ["spot1"], 0.9, "deductive"),
        CognitiveLens("Physical", "framing", ["constraint2"], ["spot2"], 0.8, "physical"),
        CognitiveLens("Quantum", "framing", ["constraint3"], ["spot3"], 0.85, "quantum"),
    ]

    initial = loop._form_initial_hypothesis(build_exact_query(7), lenses)
    initial_prediction = json.loads(initial.prediction)
    execution = ExecutionObservation(
        output=json.dumps(
            {
                "result": {
                    "mode": "topology",
                    "satisfiable_count": 464,
                    "optimal_candidate": "T001",
                },
                "confirms_hypothesis": False,
            },
            sort_keys=True,
        ),
        exit_code=0,
    )
    history = [
        ExecutionCycle(
            cycle=1,
            hypothesis=initial.statement,
            code="",
            output=execution.output,
            delta=0.06,
            converged=False,
        )
    ]

    revised = loop._revise_hypothesis(initial, execution, 0.06, history)
    revised_prediction = json.loads(revised.prediction)

    assert "approximately" in revised.statement
    assert initial_prediction["expectations"]["satisfiable_count"] < revised_prediction["expectations"]["satisfiable_count"] < 464


def test_execution_loop_accepts_close_topology_prediction_as_confirmed():
    loop = ExecutionLoop()
    payload = {
        "mode": "topology",
        "expectations": {
            "satisfiable_count": 451,
            "optimal_candidate": "T001",
        },
    }

    code = loop._build_topology_code(build_exact_query(7), payload)
    observation = loop._execute(code)
    result = json.loads(observation.output)
    summary = loop._synthesize_topology(
        [
            ExecutionCycle(
                cycle=2,
                hypothesis="topology revision",
                code=code,
                output=observation.output,
                delta=0.0035,
                converged=True,
                prediction=json.dumps(payload, sort_keys=True),
            )
        ],
        result["result"],
    )

    assert observation.exit_code == 0
    assert result["confirms_hypothesis"] is True
    assert result["result"]["satisfiable_count"] == 464
    assert "within the 5% topology-count threshold" in summary


def test_execution_loop_updates_formal_branch_gradually_and_accepts_close_predictions():
    loop = ExecutionLoop()
    lenses = [
        CognitiveLens("Formal", "framing", ["c1", "c2", "c3", "c4", "c5", "c6"], ["spot1"], 0.9, "formal"),
        CognitiveLens("Support", "framing", ["c6"], ["spot2"], 0.7, "deductive"),
    ]
    hypothesis = Hypothesis(
        statement="formal seed",
        prediction=json.dumps(
            {
                "mode": "formal",
                "expectations": {
                    "constraint_count": 2,
                    "dominant_lens": "Formal",
                },
            },
            sort_keys=True,
        ),
        epistemic_tag="formal",
        dominant_lens="Formal",
    )
    execution = ExecutionObservation(
        output=json.dumps(
            {
                "result": {
                    "mode": "formal",
                    "constraint_count": 6,
                    "dominant_lens": "Formal",
                },
                "confirms_hypothesis": False,
            },
            sort_keys=True,
        ),
        exit_code=0,
    )

    revised = loop._revise_hypothesis(hypothesis, execution, 0.2, [])
    revised_prediction = json.loads(revised.prediction)
    formal_payload = {
        "mode": "formal",
        "expectations": {
            "constraint_count": 5,
            "dominant_lens": "Formal",
        },
    }
    code = loop._build_formal_code(formal_payload, lenses)
    observation = loop._execute(code)
    result = json.loads(observation.output)
    summary = loop._synthesize_formal(
        [
            ExecutionCycle(
                cycle=2,
                hypothesis=revised.statement,
                code=code,
                output=observation.output,
                delta=0.03,
                converged=True,
                prediction=json.dumps(formal_payload, sort_keys=True),
            )
        ],
        result["result"],
    )

    assert 2 < revised_prediction["expectations"]["constraint_count"] < 6
    assert "approximately" in revised.statement
    assert observation.exit_code == 0
    assert result["confirms_hypothesis"] is True
    assert "within one constraint" in summary


def test_execution_loop_updates_probabilistic_branch_gradually_and_accepts_close_predictions():
    loop = ExecutionLoop()
    lenses = [
        CognitiveLens("ProbA", "framing", ["c1"], ["spot1"], 0.8, "probabilistic"),
        CognitiveLens("ProbB", "framing", ["c2"], ["spot2"], 0.74, "causal"),
    ]
    hypothesis = Hypothesis(
        statement="prob seed",
        prediction=json.dumps(
            {
                "mode": "probabilistic",
                "expectations": {
                    "average_confidence": 0.5,
                    "dominant_lens": "ProbA",
                },
            },
            sort_keys=True,
        ),
        epistemic_tag="probabilistic",
        dominant_lens="ProbA",
    )
    execution = ExecutionObservation(
        output=json.dumps(
            {
                "result": {
                    "mode": "probabilistic",
                    "average_confidence": 0.77,
                    "dominant_lens": "ProbA",
                },
                "confirms_hypothesis": False,
            },
            sort_keys=True,
        ),
        exit_code=0,
    )

    revised = loop._revise_hypothesis(hypothesis, execution, 0.2, [])
    revised_prediction = json.loads(revised.prediction)
    probabilistic_payload = {
        "mode": "probabilistic",
        "expectations": {
            "average_confidence": 0.73,
            "dominant_lens": "ProbA",
        },
    }
    code = loop._build_probabilistic_code(probabilistic_payload, lenses)
    observation = loop._execute(code)
    result = json.loads(observation.output)
    summary = loop._synthesize_probabilistic(
        [
            ExecutionCycle(
                cycle=2,
                hypothesis=revised.statement,
                code=code,
                output=observation.output,
                delta=0.02,
                converged=True,
                prediction=json.dumps(probabilistic_payload, sort_keys=True),
            )
        ],
        result["result"],
    )

    assert 0.5 < revised_prediction["expectations"]["average_confidence"] < 0.77
    assert "moves toward" in revised.statement
    assert observation.exit_code == 0
    assert result["confirms_hypothesis"] is True
    assert "within 0.05" in summary


def test_execution_loop_updates_generic_branch_gradually_and_accepts_close_predictions():
    loop = ExecutionLoop()
    lenses = [
        CognitiveLens("LensA", "framing", ["c1"], ["spot1"], 0.88, "deductive"),
        CognitiveLens("LensB", "framing", ["c2"], ["spot2"], 0.7, "physical"),
        CognitiveLens("LensC", "framing", ["c3"], ["spot3"], 0.6, "causal"),
    ]
    hypothesis = Hypothesis(
        statement="generic seed",
        prediction=json.dumps(
            {
                "mode": "generic",
                "expectations": {
                    "unique_tag_count": 1,
                    "dominant_lens": "LensA",
                    "lens_count": 3,
                },
            },
            sort_keys=True,
        ),
        epistemic_tag="deductive",
        dominant_lens="LensA",
    )
    execution = ExecutionObservation(
        output=json.dumps(
            {
                "result": {
                    "mode": "generic",
                    "unique_tag_count": 3,
                    "dominant_lens": "LensA",
                    "lens_count": 3,
                },
                "confirms_hypothesis": False,
            },
            sort_keys=True,
        ),
        exit_code=0,
    )

    revised = loop._revise_hypothesis(hypothesis, execution, 0.2, [])
    revised_prediction = json.loads(revised.prediction)
    generic_payload = {
        "mode": "generic",
        "expectations": {
            "unique_tag_count": 2,
            "dominant_lens": "LensA",
            "lens_count": 3,
        },
    }
    code = loop._build_generic_code(generic_payload, lenses)
    observation = loop._execute(code)
    result = json.loads(observation.output)
    summary = loop._synthesize_generic(
        [
            ExecutionCycle(
                cycle=2,
                hypothesis=revised.statement,
                code=code,
                output=observation.output,
                delta=0.03,
                converged=True,
                prediction=json.dumps(generic_payload, sort_keys=True),
            )
        ],
        result["result"],
    )

    assert 1 < revised_prediction["expectations"]["unique_tag_count"] < 3
    assert "approximately" in revised.statement
    assert observation.exit_code == 0
    assert result["confirms_hypothesis"] is True
    assert "within one epistemic tag" in summary
