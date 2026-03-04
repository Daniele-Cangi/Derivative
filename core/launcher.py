import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List

from core.artifacts import DesignArtifact
from core.designs import EmergentDesign


@dataclass
class ArtifactLaunchReport:
    artifact_id: str
    filename: str
    status: str
    exit_code: int
    result_summary: str
    stdout: str = ""
    stderr: str = ""


@dataclass
class LaunchSession:
    session_id: str
    design_id: str
    design_title: str
    status: str
    launched_count: int
    successful_count: int
    reports: List[ArtifactLaunchReport] = field(default_factory=list)


class GeneratedArtifactLauncher:
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds

    def launch_designs(
        self,
        designs: List[EmergentDesign],
        artifact_paths: Dict[str, str],
    ) -> List[LaunchSession]:
        sessions: List[LaunchSession] = []
        for index, design in enumerate(designs, start=1):
            reports: List[ArtifactLaunchReport] = []
            for artifact in design.artifacts:
                if artifact.language != "python":
                    continue
                reports.append(self._launch_python_artifact(artifact, artifact_paths))

            successful_count = sum(1 for report in reports if report.status == "launched")
            launched_count = len(reports)
            if launched_count == 0:
                status = "skipped"
            elif successful_count == launched_count:
                status = "launched"
            elif successful_count > 0:
                status = "partial"
            else:
                status = "failed"

            sessions.append(
                LaunchSession(
                    session_id=f"LS-{index}",
                    design_id=design.design_id,
                    design_title=design.title,
                    status=status,
                    launched_count=launched_count,
                    successful_count=successful_count,
                    reports=reports,
                )
            )

        return sessions

    def _launch_python_artifact(
        self,
        artifact: DesignArtifact,
        artifact_paths: Dict[str, str],
    ) -> ArtifactLaunchReport:
        artifact_path = artifact_paths.get(artifact.artifact_id)
        if not artifact_path:
            return ArtifactLaunchReport(
                artifact_id=artifact.artifact_id,
                filename=artifact.filename,
                status="failed",
                exit_code=1,
                result_summary="Artifact path not found in workspace export.",
            )

        script = "\n".join(
            [
                "import importlib.util",
                "import json",
                "import sys",
                "",
                "path = sys.argv[1]",
                "spec = importlib.util.spec_from_file_location('derivative_generated_module', path)",
                "module = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(module)",
                "candidates = [name for name in dir(module) if name.startswith('build_') and callable(getattr(module, name))]",
                "if not candidates:",
                "    print(json.dumps({'status': 'no_entrypoint'}))",
                "    raise SystemExit(2)",
                "entrypoint = candidates[0]",
                "result = getattr(module, entrypoint)()",
                "payload = {'status': 'ok', 'entrypoint': entrypoint, 'result_type': type(result).__name__}",
                "if hasattr(result, 'num_qubits'):",
                "    payload['num_qubits'] = int(result.num_qubits)",
                "if hasattr(result, 'number_of_nodes') and callable(result.number_of_nodes):",
                "    payload['node_count'] = int(result.number_of_nodes())",
                "if hasattr(result, 'number_of_edges') and callable(result.number_of_edges):",
                "    payload['edge_count'] = int(result.number_of_edges())",
                "if hasattr(result, 'assertions') and callable(result.assertions):",
                "    payload['assertion_count'] = len(result.assertions())",
                "print(json.dumps(payload))",
            ]
        )

        try:
            completed = subprocess.run(
                [sys.executable, "-c", script, artifact_path],
                capture_output=True,
                text=True,
                timeout=self._resolve_timeout(artifact),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ArtifactLaunchReport(
                artifact_id=artifact.artifact_id,
                filename=artifact.filename,
                status="failed",
                exit_code=124,
                result_summary=f"Execution exceeded {self._resolve_timeout(artifact)}s timeout.",
            )

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        summary = self._summarize_process_result(stdout, stderr, completed.returncode)
        status = "launched" if completed.returncode == 0 else "failed"

        return ArtifactLaunchReport(
            artifact_id=artifact.artifact_id,
            filename=artifact.filename,
            status=status,
            exit_code=completed.returncode,
            result_summary=summary,
            stdout=stdout,
            stderr=stderr,
        )

    def _summarize_process_result(self, stdout: str, stderr: str, returncode: int) -> str:
        if stdout:
            last_line = stdout.splitlines()[-1]
            try:
                payload = json.loads(last_line)
            except json.JSONDecodeError:
                payload = None

            if isinstance(payload, dict):
                if payload.get("status") == "ok":
                    summary_parts = [
                        f"{payload.get('entrypoint', 'unknown')} -> {payload.get('result_type', 'unknown')}"
                    ]
                    if "num_qubits" in payload:
                        summary_parts.append(f"qubits={payload['num_qubits']}")
                    if "node_count" in payload:
                        summary_parts.append(f"nodes={payload['node_count']}")
                    if "edge_count" in payload:
                        summary_parts.append(f"edges={payload['edge_count']}")
                    if "assertion_count" in payload:
                        summary_parts.append(f"assertions={payload['assertion_count']}")
                    return ", ".join(summary_parts)
                if payload.get("status") == "no_entrypoint":
                    return "No build_* entrypoint found in generated module."

            return last_line[:200]

        if stderr:
            return stderr.splitlines()[-1][:200]
        return f"Process exited with code {returncode}."

    def _resolve_timeout(self, artifact: DesignArtifact) -> int:
        if artifact.artifact_type == "quantum_circuit":
            return max(self.timeout_seconds, 30)
        return self.timeout_seconds
