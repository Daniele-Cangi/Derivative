from core.artifacts import DesignArtifact
from core.designs import EmergentDesign
from core.launcher import GeneratedArtifactLauncher
from core.workspace import ArtifactWorkspace


def test_launcher_runs_generated_python_artifact(tmp_path):
    workspace = ArtifactWorkspace(base_dir=str(tmp_path))
    launcher = GeneratedArtifactLauncher(timeout_seconds=5)

    design = EmergentDesign(
        design_id="ED-1",
        title="Constraint Lattice Compiler",
        premise="A constraint-first design.",
        composition_tags=["formal"],
        component_primitives=["Compile invariants into a lattice."],
        implementation_outline=["Frame the lattice."],
        governing_constraints=["Keep hard requirements satisfiable."],
        novelty_score=0.8,
        feasibility_score=0.8,
        composite_score=0.8,
        artifacts=[
            DesignArtifact(
                artifact_id="A1",
                artifact_type="constraint_solver",
                filename="constraint_lattice_compiler_solver.py",
                language="python",
                content=(
                    "def build_solver():\n"
                    "    class Dummy:\n"
                    "        def assertions(self):\n"
                    "            return [1, 2, 3]\n"
                    "    return Dummy()\n"
                ),
                execution_note="Build the solver.",
            )
        ],
        lineage_titles=[],
        mutation_strategy="constraint-compilation",
    )

    export = workspace.export(
        step_id="launch1",
        problem="launch problem",
        designs=[design],
        execution_sessions=[],
    )
    sessions = launcher.launch_designs([design], export.artifact_paths)

    assert len(sessions) == 1
    assert sessions[0].status == "launched"
    assert sessions[0].successful_count == 1
    assert "build_solver" in sessions[0].reports[0].result_summary
