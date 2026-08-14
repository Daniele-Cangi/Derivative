import json
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


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]
