import json
import os
import subprocess
from pathlib import Path

import pytest

import core.forge.execution as execution_module
from core.forge.execution import (
    DOCKER_BACKEND,
    DockerSandboxExecutor,
    ExecutionPolicy,
    LocalProcessExecutor,
    SandboxProcessRequest,
)
from core.forge.validator_stage import ValidatorStage
from core.forge.coder_stage import CoderStage
from core.forge.contracts import FeasiblePlan
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler
from forge import run_forge


DOCKER_TESTS_ENABLED = os.environ.get("FORGE_RUN_DOCKER_TESTS") == "1"


def test_local_executor_uses_allowlisted_environment_and_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-generated-code")
    executor = LocalProcessExecutor()

    result = executor.run(
        SandboxProcessRequest(
            command=[
                "python",
                "-c",
                "import json, os; from pathlib import Path; "
                "print(json.dumps({'cwd': Path.cwd().name, 'secret': 'OPENAI_API_KEY' in os.environ}))",
            ],
            workspace=tmp_path,
        )
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload == {"cwd": tmp_path.name, "secret": False}
    assert result.isolation["isolated"] is False


def test_local_executor_normalizes_timeout_without_raising(tmp_path):
    executor = LocalProcessExecutor()

    result = executor.run(
        SandboxProcessRequest(
            command=["python", "-c", "import time; time.sleep(2)"],
            workspace=tmp_path,
            timeout_seconds=1,
        )
    )

    assert result.returncode == 124
    assert result.timed_out is True
    assert "timed out" in result.stderr.lower()


def test_executor_rejects_working_directory_escape(tmp_path):
    executor = LocalProcessExecutor()

    with pytest.raises(ValueError, match="escapes the workspace"):
        executor.run(
            SandboxProcessRequest(
                command=["python", "-c", "print('unsafe')"],
                workspace=tmp_path,
                working_directory="..",
            )
        )


def test_docker_executor_applies_network_filesystem_and_resource_controls(
    tmp_path,
    monkeypatch,
):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="isolated\n", stderr="")

    monkeypatch.setattr(execution_module.subprocess, "run", fake_run)
    policy = ExecutionPolicy(
        backend=DOCKER_BACKEND,
        image="forge-test-image:locked",
        timeout_seconds=20,
        memory_mb=256,
        cpu_count=0.5,
        pids_limit=32,
        tmpfs_mb=16,
    )
    result = DockerSandboxExecutor(policy).run(
        SandboxProcessRequest(
            command=["python", "-c", "print('isolated')"],
            workspace=tmp_path,
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
        )
    )

    command = calls[0]
    assert result.returncode == 0
    assert result.isolation["isolated"] is True
    assert command[:2] == ["docker", "run"]
    assert _option_value(command, "--network") == "none"
    assert "--read-only" in command
    assert _option_value(command, "--cap-drop") == "ALL"
    assert _option_value(command, "--security-opt") == "no-new-privileges"
    assert _option_value(command, "--memory") == "256m"
    assert _option_value(command, "--cpus") == "0.5"
    assert _option_value(command, "--pids-limit") == "32"
    assert "nosuid" in _option_value(command, "--tmpfs")
    assert "nodev" in _option_value(command, "--tmpfs")
    assert "noexec" in _option_value(command, "--tmpfs")
    assert _option_value(command, "--workdir") == "/workspace"
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        assert _option_value(command, "--user") == f"{os.getuid()}:{os.getgid()}"
    assert "HOME=/tmp" in command
    assert "TMPDIR=/tmp" in command
    mount = _option_value(command, "--mount")
    assert f"source={Path(tmp_path).resolve()}" in mount
    assert "target=/workspace" in mount
    assert "forge-test-image:locked" in command


def test_docker_launch_failure_is_fail_closed(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            125,
            stdout="",
            stderr="Cannot connect to the Docker daemon",
        )

    monkeypatch.setattr(execution_module.subprocess, "run", fake_run)
    result = DockerSandboxExecutor().run(
        SandboxProcessRequest(
            command=["python", "-c", "print('never')"],
            workspace=tmp_path,
        )
    )

    assert result.returncode == 125
    assert result.launch_error is not None
    assert "Docker daemon" in result.launch_error
    assert result.isolation["isolated"] is True


def test_validator_refuses_local_backend_when_isolation_is_required():
    validator = ValidatorStage(
        executor=LocalProcessExecutor(),
        require_isolation=True,
    )

    result = validator.validate(None, None, None)  # type: ignore[arg-type]

    assert result.passed is False
    assert result.failure_signatures == ["sandbox_policy_violation"]
    assert result.evidence["execution_policy"]["isolated"] is False
    assert result.evidence["executed_tests"]["ran"] is False
    assert result.metrics["passed_layers"] == {
        "layer1": False,
        "layer2": False,
        "layer3": False,
    }


def test_forge_orchestrator_fails_closed_when_local_backend_is_requested(tmp_path):
    result = run_forge(
        requirement=(
            "Build a Python CLI that reads a CSV of contracts, writes a summary CSV, "
            "and includes tests."
        ),
        execution_backend="local",
        output_root=str(tmp_path / "runs"),
        packaging_output_root=str(tmp_path / "packages"),
        max_coder_attempts=1,
    )

    assert result.terminal_status == "validation_failed"
    assert result.validation is not None
    assert result.validation.failure_signatures == ["sandbox_policy_violation"]
    assert result.packaged_artifact is None
    assert not (tmp_path / "packages").exists()


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


@pytest.mark.skipif(not DOCKER_TESTS_ENABLED, reason="real Docker sandbox tests are CI-gated")
def test_real_docker_sandbox_enforces_runtime_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret-must-not-cross-boundary")
    executor = DockerSandboxExecutor()
    script = '''import json
import os
import socket
from pathlib import Path

root_blocked = False
try:
    Path('/forge-root-write').write_text('forbidden', encoding='utf-8')
except OSError:
    root_blocked = True

network_blocked = False
try:
    socket.create_connection(('1.1.1.1', 53), timeout=0.5)
except OSError:
    network_blocked = True

Path('workspace-write.txt').write_text('allowed', encoding='utf-8')
cgroup = Path('/sys/fs/cgroup')
payload = {
    'root_blocked': root_blocked,
    'network_blocked': network_blocked,
    'workspace_write': Path('workspace-write.txt').read_text(encoding='utf-8'),
    'host_secret_present': 'OPENAI_API_KEY' in os.environ,
    'memory_max': (cgroup / 'memory.max').read_text().strip(),
    'pids_max': (cgroup / 'pids.max').read_text().strip(),
    'cpu_max': (cgroup / 'cpu.max').read_text().strip(),
}
print(json.dumps(payload, sort_keys=True))
'''

    result = executor.run(
        SandboxProcessRequest(
            command=["python", "-c", script],
            workspace=tmp_path,
            timeout_seconds=20,
        )
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["root_blocked"] is True
    assert payload["network_blocked"] is True
    assert payload["workspace_write"] == "allowed"
    assert payload["host_secret_present"] is False
    assert int(payload["memory_max"]) <= 512 * 1024 * 1024
    assert int(payload["pids_max"]) <= 64
    quota, period = (int(value) for value in payload["cpu_max"].split())
    assert quota / period <= 1.0
    assert result.isolation["isolated"] is True


@pytest.mark.skipif(not DOCKER_TESTS_ENABLED, reason="real Docker sandbox tests are CI-gated")
def test_real_validator_executes_generated_artifact_inside_docker(tmp_path):
    requirement = (
        "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
        "flags contracts expiring in less than 90 days, writes a summary CSV, and includes tests."
    )
    spec = RequirementCompiler().compile(requirement)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)
    validator = ValidatorStage(
        executor=DockerSandboxExecutor(),
        require_isolation=True,
    )

    validation = validator.validate(artifact, plan, spec)

    assert validation.passed is True, validation.failures
    assert validation.evidence["execution_policy"]["backend"] == "docker"
    assert validation.evidence["execution_policy"]["isolated"] is True
    assert validation.evidence["executed_tests"]["backend"] == "docker"
    assert validation.evidence["executed_tests"]["isolation"]["network_mode"] == "none"
