from core.execution_loop import ExecutionCycle, ExecutionResult
from core.kernel import ReasoningResult, ReasoningStep
from memory.gene_pool import DesignGenePool


def test_gene_pool_records_converged_hypotheses(tmp_path):
    pool = DesignGenePool(storage_file=str(tmp_path / "gene_pool.json"))
    result = ReasoningResult(
        conclusion="Execution-grounded conclusion",
        reasoning_chain=[ReasoningStep("1", "step")],
        violated_constraints=[],
        epistemic_confidence=0.9,
        lens_contributions={"LensA": 1.0},
        generated_designs=[],
        execution_result=ExecutionResult(
            conclusion="Converged",
            cycles_used=2,
            converged=True,
            history=[
                ExecutionCycle(
                    cycle=1,
                    hypothesis="Initial hypothesis",
                    code="print('x')",
                    output='{"result": {}, "confirms_hypothesis": false}',
                    delta=0.2,
                    converged=False,
                ),
                ExecutionCycle(
                    cycle=2,
                    hypothesis="Revised hypothesis",
                    code="print('y')",
                    output='{"result": {}, "confirms_hypothesis": true}',
                    delta=0.0,
                    converged=True,
                ),
            ],
            final_code="print('y')",
            final_output='{"result": {}, "confirms_hypothesis": true}',
        ),
    )

    recorded = pool.record_execution(result, "same problem")

    assert len(recorded) == 1
    assert recorded[0].hypothesis == "Revised hypothesis"
    assert recorded[0].was_verified is True
