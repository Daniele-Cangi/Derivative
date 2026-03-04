import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from core.designs import EmergentDesign
from core.executor import ExecutionSession


@dataclass
class WorkspaceExport:
    workspace_id: str
    root_path: str
    manifest_path: str
    exported_files: List[str]
    artifact_paths: dict[str, str]


class ArtifactWorkspace:
    def __init__(self, base_dir: str = "generated_artifacts"):
        self.base_dir = base_dir

    def export(
        self,
        step_id: str,
        problem: str,
        designs: List[EmergentDesign],
        execution_sessions: Optional[List[ExecutionSession]] = None,
    ) -> WorkspaceExport:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        workspace_id = f"{timestamp}_{step_id}"
        root_path = os.path.abspath(os.path.join(self.base_dir, workspace_id))
        os.makedirs(root_path, exist_ok=True)

        exported_files: List[str] = []
        artifact_paths: dict[str, str] = {}
        for design in designs:
            design_dir = os.path.join(root_path, self._slugify(f"{design.design_id}_{design.title}"))
            os.makedirs(design_dir, exist_ok=True)
            for artifact in design.artifacts:
                artifact_path = os.path.join(design_dir, artifact.filename)
                with open(artifact_path, "w", encoding="utf-8") as handle:
                    handle.write(artifact.content)
                absolute_artifact_path = os.path.abspath(artifact_path)
                exported_files.append(absolute_artifact_path)
                artifact_paths[artifact.artifact_id] = absolute_artifact_path

        manifest_path = os.path.join(root_path, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                self._build_manifest(problem, designs, execution_sessions or [], exported_files),
                handle,
                indent=2,
            )

        return WorkspaceExport(
            workspace_id=workspace_id,
            root_path=root_path,
            manifest_path=os.path.abspath(manifest_path),
            exported_files=exported_files,
            artifact_paths=artifact_paths,
        )

    def _build_manifest(
        self,
        problem: str,
        designs: List[EmergentDesign],
        execution_sessions: List[ExecutionSession],
        exported_files: List[str],
    ) -> dict:
        sessions_by_design = {session.design_id: session for session in execution_sessions}
        return {
            "problem": problem,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "exported_file_count": len(exported_files),
            "designs": [
                {
                    "design_id": design.design_id,
                    "title": design.title,
                    "mutation_strategy": design.mutation_strategy,
                    "composite_score": design.composite_score,
                    "artifact_files": [artifact.filename for artifact in design.artifacts],
                    "execution_status": sessions_by_design.get(design.design_id).status
                    if design.design_id in sessions_by_design
                    else "not_evaluated",
                }
                for design in designs
            ],
        }

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
        return slug or "design"
