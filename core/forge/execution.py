import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Protocol


LOCAL_BACKEND = "local"
DOCKER_BACKEND = "docker"
SUPPORTED_EXECUTION_BACKENDS = frozenset({LOCAL_BACKEND, DOCKER_BACKEND})
DEFAULT_SANDBOX_IMAGE = "derivative-forge-sandbox:py311"
_LOCAL_ENV_ALLOWLIST = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
}


@dataclass(frozen=True)
class ExecutionPolicy:
    backend: str
    image: str = DEFAULT_SANDBOX_IMAGE
    network_mode: str = "none"
    read_only_root: bool = True
    workspace_access: str = "read_write"
    timeout_seconds: int = 120
    memory_mb: int = 512
    cpu_count: float = 1.0
    pids_limit: int = 64
    tmpfs_mb: int = 64
    max_output_chars: int = 12000

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_EXECUTION_BACKENDS:
            raise ValueError(f"Unsupported execution backend: {self.backend}")
        if self.timeout_seconds < 1:
            raise ValueError("Execution timeout_seconds must be positive.")
        if self.memory_mb < 64 or self.cpu_count <= 0 or self.pids_limit < 1:
            raise ValueError("Execution resource limits must be positive and bounded.")
        if self.max_output_chars < 1000:
            raise ValueError("Execution max_output_chars must be at least 1000.")
        if self.backend == DOCKER_BACKEND:
            if self.network_mode != "none":
                raise ValueError("Docker sandbox network_mode must be 'none'.")
            if not self.read_only_root:
                raise ValueError("Docker sandbox requires a read-only root filesystem.")
            if self.workspace_access != "read_write":
                raise ValueError("Docker sandbox workspace_access must be 'read_write'.")

    @property
    def isolated(self) -> bool:
        return self.backend == DOCKER_BACKEND

    def evidence(self) -> Dict[str, object]:
        return {**asdict(self), "isolated": self.isolated}


@dataclass(frozen=True)
class SandboxProcessRequest:
    command: List[str]
    workspace: Path
    working_directory: str = "."
    environment: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class SandboxProcessResult:
    returncode: int
    stdout: str
    stderr: str
    backend: str
    execution_time_seconds: float
    timed_out: bool = False
    launch_error: str | None = None
    isolation: Dict[str, object] = field(default_factory=dict)


class ProcessExecutor(Protocol):
    policy: ExecutionPolicy

    def run(self, request: SandboxProcessRequest) -> SandboxProcessResult: ...


class LocalProcessExecutor:
    """Trusted-test backend. Production orchestration must require isolation."""

    def __init__(
        self,
        policy: ExecutionPolicy | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.policy = policy or ExecutionPolicy(backend=LOCAL_BACKEND)
        if self.policy.backend != LOCAL_BACKEND:
            raise ValueError("LocalProcessExecutor requires backend='local'.")
        self.python_executable = python_executable or sys.executable

    def run(self, request: SandboxProcessRequest) -> SandboxProcessResult:
        started = time.perf_counter()
        workspace, cwd = _resolve_workspace(request)
        command = list(request.command)
        if command and command[0] == "python":
            command[0] = self.python_executable
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in _LOCAL_ENV_ALLOWLIST
        }
        environment.update({str(key): str(value) for key, value in request.environment.items()})
        timeout = request.timeout_seconds or self.policy.timeout_seconds
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=environment,
            )
            return _result_from_completed(
                completed,
                self.policy,
                time.perf_counter() - started,
            )
        except subprocess.TimeoutExpired as exc:
            return _timeout_result(exc, self.policy, time.perf_counter() - started, timeout)
        except OSError as exc:
            return _launch_error_result(self.policy, time.perf_counter() - started, exc)


class DockerSandboxExecutor:
    def __init__(self, policy: ExecutionPolicy | None = None) -> None:
        self.policy = policy or ExecutionPolicy(backend=DOCKER_BACKEND)
        if self.policy.backend != DOCKER_BACKEND:
            raise ValueError("DockerSandboxExecutor requires backend='docker'.")

    def run(self, request: SandboxProcessRequest) -> SandboxProcessResult:
        started = time.perf_counter()
        workspace, cwd = _resolve_workspace(request)
        relative_cwd = cwd.relative_to(workspace).as_posix()
        container_cwd = "/workspace" if relative_cwd == "." else f"/workspace/{relative_cwd}"
        timeout = request.timeout_seconds or self.policy.timeout_seconds
        container_name = f"forge-sandbox-{uuid.uuid4().hex[:16]}"
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            self.policy.network_mode,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            f"{self.policy.memory_mb}m",
            "--cpus",
            str(self.policy.cpu_count),
            "--pids-limit",
            str(self.policy.pids_limit),
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={self.policy.tmpfs_mb}m",
            "--mount",
            f"type=bind,source={workspace},target=/workspace",
            "--workdir",
            container_cwd,
        ]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        command.extend(["--env", "HOME=/tmp", "--env", "TMPDIR=/tmp"])
        for key, value in sorted(request.environment.items()):
            command.extend(["--env", f"{key}={value}"])
        command.extend([self.policy.image, *request.command])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=_docker_client_environment(),
            )
            result = _result_from_completed(
                completed,
                self.policy,
                time.perf_counter() - started,
            )
            if completed.returncode in {125, 126, 127}:
                return SandboxProcessResult(
                    **{
                        **result.__dict__,
                        "launch_error": (completed.stderr or completed.stdout).strip()
                        or "Docker sandbox failed to launch.",
                    }
                )
            return result
        except subprocess.TimeoutExpired as exc:
            _force_remove_container(container_name)
            return _timeout_result(exc, self.policy, time.perf_counter() - started, timeout)
        except OSError as exc:
            return _launch_error_result(self.policy, time.perf_counter() - started, exc)


def create_process_executor(
    backend: str,
    *,
    timeout_seconds: int = 120,
    image: str = DEFAULT_SANDBOX_IMAGE,
    python_executable: str | None = None,
) -> ProcessExecutor:
    normalized = backend.strip().lower()
    policy = ExecutionPolicy(
        backend=normalized,
        image=image,
        timeout_seconds=timeout_seconds,
    )
    if normalized == DOCKER_BACKEND:
        return DockerSandboxExecutor(policy)
    return LocalProcessExecutor(policy, python_executable=python_executable)


def _resolve_workspace(request: SandboxProcessRequest) -> tuple[Path, Path]:
    workspace = Path(request.workspace).resolve()
    if not workspace.is_dir():
        raise ValueError(f"Execution workspace does not exist: {workspace}")
    cwd = (workspace / request.working_directory).resolve()
    if cwd != workspace and not cwd.is_relative_to(workspace):
        raise ValueError("Execution working directory escapes the workspace.")
    if not cwd.is_dir():
        raise ValueError(f"Execution working directory does not exist: {cwd}")
    if not request.command or not all(isinstance(part, str) and part for part in request.command):
        raise ValueError("Execution command must contain non-empty string arguments.")
    return workspace, cwd


def _result_from_completed(
    completed: subprocess.CompletedProcess[str],
    policy: ExecutionPolicy,
    elapsed: float,
) -> SandboxProcessResult:
    return SandboxProcessResult(
        returncode=completed.returncode,
        stdout=_bounded(completed.stdout or "", policy.max_output_chars),
        stderr=_bounded(completed.stderr or "", policy.max_output_chars),
        backend=policy.backend,
        execution_time_seconds=elapsed,
        isolation=policy.evidence(),
    )


def _timeout_result(
    exc: subprocess.TimeoutExpired,
    policy: ExecutionPolicy,
    elapsed: float,
    timeout: int,
) -> SandboxProcessResult:
    return SandboxProcessResult(
        returncode=124,
        stdout=_bounded(_coerce_output(exc.stdout), policy.max_output_chars),
        stderr=_bounded(
            _coerce_output(exc.stderr) or f"Execution timed out after {timeout} seconds.",
            policy.max_output_chars,
        ),
        backend=policy.backend,
        execution_time_seconds=elapsed,
        timed_out=True,
        isolation=policy.evidence(),
    )


def _launch_error_result(
    policy: ExecutionPolicy,
    elapsed: float,
    exc: OSError,
) -> SandboxProcessResult:
    message = f"{type(exc).__name__}: {exc}"
    return SandboxProcessResult(
        returncode=125,
        stdout="",
        stderr=message,
        backend=policy.backend,
        execution_time_seconds=elapsed,
        launch_error=message,
        isolation=policy.evidence(),
    )


def _force_remove_container(container_name: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "--force", container_name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_docker_client_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def _docker_client_environment() -> Dict[str, str]:
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "HOME", "USERPROFILE", "DOCKER_HOST"}
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed
    }


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
