import json
import ast
from dataclasses import dataclass, field
from typing import List

from core.artifacts import DesignArtifact
from core.designs import EmergentDesign
from core.json_utils import clamp_float


@dataclass
class ArtifactExecutionReport:
    artifact_id: str
    filename: str
    artifact_type: str
    status: str
    validation_score: float
    observations: List[str] = field(default_factory=list)


@dataclass
class ExecutionSession:
    session_id: str
    design_id: str
    design_title: str
    status: str
    aggregate_score: float
    reports: List[ArtifactExecutionReport] = field(default_factory=list)


class ControlledExecutor:
    def evaluate_designs(self, designs: List[EmergentDesign]) -> List[ExecutionSession]:
        sessions: List[ExecutionSession] = []
        for index, design in enumerate(designs, start=1):
            reports = [self._evaluate_artifact(artifact) for artifact in design.artifacts]
            aggregate_score = clamp_float(
                sum(report.validation_score for report in reports) / max(1, len(reports)),
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            )
            failure_count = sum(1 for report in reports if report.status == "failed")
            warning_count = sum(1 for report in reports if report.status == "warning")

            if failure_count:
                status = "failed"
            elif warning_count:
                status = "warning"
            else:
                status = "validated"

            sessions.append(
                ExecutionSession(
                    session_id=f"XS-{index}",
                    design_id=design.design_id,
                    design_title=design.title,
                    status=status,
                    aggregate_score=aggregate_score,
                    reports=reports,
                )
            )

        return sessions

    def _evaluate_artifact(self, artifact: DesignArtifact) -> ArtifactExecutionReport:
        handlers = {
            "python": self._validate_python,
            "yaml": self._validate_yaml,
            "json": self._validate_json,
            "markdown": self._validate_markdown,
        }
        validator = handlers.get(artifact.language, self._validate_generic)
        status, score, observations = validator(artifact)
        return ArtifactExecutionReport(
            artifact_id=artifact.artifact_id,
            filename=artifact.filename,
            artifact_type=artifact.artifact_type,
            status=status,
            validation_score=score,
            observations=observations,
        )

    def _validate_python(self, artifact: DesignArtifact) -> tuple[str, float, List[str]]:
        observations: List[str] = []
        try:
            ast.parse(artifact.content)
        except SyntaxError as exc:
            observations.append(f"Python syntax failed: line {exc.lineno}")
            return "failed", 0.0, observations

        score = 0.65
        observations.append("Python syntax parsed successfully.")
        if "def " in artifact.content:
            score += 0.1
            observations.append("Found at least one callable entry point.")
        if artifact.artifact_type == "quantum_circuit" and "QuantumCircuit" in artifact.content:
            score += 0.1
            observations.append("Quantum circuit scaffold detected.")
        if artifact.artifact_type == "constraint_solver" and "Solver" in artifact.content:
            score += 0.1
            observations.append("Constraint solver scaffold detected.")
        if artifact.artifact_type == "topology_graph" and "networkx" in artifact.content:
            score += 0.1
            observations.append("Topology graph scaffold detected.")
        return "validated", clamp_float(score), observations

    def _validate_yaml(self, artifact: DesignArtifact) -> tuple[str, float, List[str]]:
        lines = [line for line in artifact.content.splitlines() if line.strip()]
        observations: List[str] = []
        if not lines:
            return "failed", 0.0, ["YAML artifact is empty."]

        colon_lines = sum(1 for line in lines if ":" in line)
        if colon_lines < 3:
            return "warning", 0.45, ["YAML-like structure is too sparse for reliable orchestration."]

        score = 0.7 + min(0.2, 0.03 * colon_lines)
        observations.append("Structured YAML manifest detected.")
        if any(line.startswith("title:") for line in lines):
            score += 0.05
            observations.append("Manifest declares a title.")
        return "validated", clamp_float(score), observations

    def _validate_markdown(self, artifact: DesignArtifact) -> tuple[str, float, List[str]]:
        lines = [line for line in artifact.content.splitlines() if line.strip()]
        if not lines:
            return "failed", 0.0, ["Markdown artifact is empty."]

        observations = ["Markdown artifact contains readable lineage notes."]
        score = 0.6
        if any(line.startswith("#") for line in lines):
            score += 0.1
            observations.append("Section headers detected.")
        if any(line.startswith("-") for line in lines):
            score += 0.1
            observations.append("Mutation bullet structure detected.")
        return "validated", clamp_float(score), observations

    def _validate_json(self, artifact: DesignArtifact) -> tuple[str, float, List[str]]:
        try:
            payload = json.loads(artifact.content)
        except json.JSONDecodeError as exc:
            return "failed", 0.0, [f"JSON parsing failed: line {exc.lineno} column {exc.colno}."]

        observations = ["JSON artifact parsed successfully."]
        score = 0.7
        if isinstance(payload, dict):
            score += 0.05
            observations.append("Top-level JSON object detected.")
        if artifact.artifact_type == "topology_enumeration":
            candidates = payload.get("candidates") if isinstance(payload, dict) else None
            if isinstance(candidates, list):
                score += 0.15
                observations.append(f"Exact topology enumeration payload contains {len(candidates)} candidate(s).")
            else:
                return "warning", 0.55, ["Topology enumeration artifact is missing its candidates list."]
        return "validated", clamp_float(score), observations

    def _validate_generic(self, artifact: DesignArtifact) -> tuple[str, float, List[str]]:
        if artifact.content.strip():
            return "warning", 0.5, ["Artifact content is present, but no dedicated validator exists."]
        return "failed", 0.0, ["Artifact content is empty."]
