from core.artifacts import DesignArtifact
from core.designs import EmergentDesign
from core.executor import ControlledExecutor


def test_executor_validates_generated_artifacts():
    executor = ControlledExecutor()
    design = EmergentDesign(
        design_id="ED-1",
        title="Quantum Branch Forge",
        premise="A branching architecture.",
        composition_tags=["quantum", "formal"],
        component_primitives=["Spawn parallel design states."],
        implementation_outline=["Frame the invention around branching."],
        governing_constraints=["Keep hard invariants satisfiable."],
        novelty_score=0.9,
        feasibility_score=0.8,
        composite_score=0.86,
        artifacts=[
            DesignArtifact(
                artifact_id="A1",
                artifact_type="blueprint",
                filename="quantum_branch_forge.yaml",
                language="yaml",
                content="title: Quantum Branch Forge\nmutation_strategy: branching\ncomposite_score: 0.86",
                execution_note="Load manifest.",
            ),
            DesignArtifact(
                artifact_id="A2",
                artifact_type="quantum_circuit",
                filename="quantum_branch_forge_circuit.py",
                language="python",
                content="from qiskit import QuantumCircuit\n\ndef build_design_circuit():\n    return QuantumCircuit(2)\n",
                execution_note="Build the circuit.",
            ),
            DesignArtifact(
                artifact_id="A3",
                artifact_type="topology_enumeration",
                filename="quantum_branch_forge_enumeration.json",
                language="json",
                content='{"search_type": "exact_topology_enumeration", "candidates": [{"candidate_id": "T001"}]}',
                execution_note="Persist the exact enumeration.",
            ),
        ],
        lineage_titles=[],
        mutation_strategy="late-collapse branching",
    )

    sessions = executor.evaluate_designs([design])

    assert len(sessions) == 1
    assert sessions[0].status == "validated"
    assert sessions[0].aggregate_score > 0.7
    assert len(sessions[0].reports) == 3
