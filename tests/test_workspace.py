import json

from core.artifacts import DesignArtifact
from core.designs import EmergentDesign
from core.executor import ExecutionSession
from core.workspace import ArtifactWorkspace


def test_workspace_exports_artifacts_and_manifest(tmp_path):
    workspace = ArtifactWorkspace(base_dir=str(tmp_path))
    design = EmergentDesign(
        design_id="ED-1",
        title="Constraint Lattice Compiler",
        premise="A constraint-first design.",
        composition_tags=["formal", "symbolic"],
        component_primitives=["Compile invariants into a lattice."],
        implementation_outline=["Frame the lattice."],
        governing_constraints=["Keep hard requirements satisfiable."],
        novelty_score=0.9,
        feasibility_score=0.8,
        composite_score=0.86,
        artifacts=[
            DesignArtifact(
                artifact_id="A1",
                artifact_type="blueprint",
                filename="constraint_lattice_compiler.yaml",
                language="yaml",
                content="title: Constraint Lattice Compiler\nmutation_strategy: constraint-compilation",
                execution_note="Load manifest.",
            )
        ],
        lineage_titles=[],
        mutation_strategy="constraint-compilation",
    )
    sessions = [
        ExecutionSession(
            session_id="XS-1",
            design_id="ED-1",
            design_title="Constraint Lattice Compiler",
            status="validated",
            aggregate_score=0.84,
            reports=[],
        )
    ]

    export = workspace.export(
        step_id="abc123",
        problem="test problem",
        designs=[design],
        execution_sessions=sessions,
    )

    assert export.exported_files
    assert export.artifact_paths["A1"] in export.exported_files
    assert export.root_path.startswith(str(tmp_path))
    with open(export.manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["problem"] == "test problem"
    assert manifest["designs"][0]["execution_status"] == "validated"
