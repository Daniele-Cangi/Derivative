import ast
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from core.forge.contracts import BuildSpec, CodeArtifact, FeasiblePlan, ValidationLayerResult
from core.forge.execution import (
    LocalProcessExecutor,
    ProcessExecutor,
    SandboxProcessRequest,
    SandboxProcessResult,
)
from core.forge.validation.common import ValidationLayerBase


class RuntimeValidationLayer(ValidationLayerBase):
    def __init__(self, executor: ProcessExecutor | str, timeout_seconds: int):
        self.executor = (
            LocalProcessExecutor(python_executable=executor)
            if isinstance(executor, str)
            else executor
        )
        self.timeout_seconds = timeout_seconds

    def validate(
        self,
        code_artifact: CodeArtifact,
        plan: FeasiblePlan,
        build_spec: BuildSpec,
        materialized: Dict[str, Path],
        workspace: Path,
    ) -> ValidationLayerResult:
        started = time.perf_counter()
        failures: List[str] = []
        signatures: List[str] = []
        evidence: Dict[str, object] = {
            "parse_errors": [],
            "import_results": {},
            "entrypoint_results": {},
        }

        expected_paths = {generated.path for generated in code_artifact.files}
        missing_written = sorted(path for path in expected_paths if path not in materialized)
        if missing_written:
            failures.append(f"Workspace materialization failed for files: {missing_written}.")
            self._append_unique(signatures, "missing_required_file")

        for path, target in materialized.items():
            if not path.endswith(".py"):
                continue
            try:
                ast.parse(target.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                failures.append(f"Syntax error in {path}: line {exc.lineno} column {exc.offset}.")
                self._append_unique(signatures, "syntax_error")
                evidence["parse_errors"].append(
                    {"path": path, "line": exc.lineno, "column": exc.offset, "message": str(exc)}
                )

        src_modules = sorted(
            path for path in materialized.keys() if path.startswith("src/") and path.endswith(".py")
        )
        import_ok, import_payload = self._import_modules(workspace, src_modules)
        evidence["import_results"] = import_payload
        if not import_ok:
            failures.append("Module import checks failed for one or more src modules.")
            self._append_unique(
                signatures,
                "sandbox_unavailable" if import_payload.get("launch_error") else "import_failure",
            )

        entrypoint_evidence: Dict[str, object] = {}
        declared_entrypoint_interfaces = [
            interface
            for interface in plan.interfaces
            if interface.interface_type in {"cli_entrypoint", "entrypoint"}
        ]
        if declared_entrypoint_interfaces and not code_artifact.runnable_entrypoints:
            failures.append("Entrypoint interface declared but no runnable_entrypoints were provided.")
            self._append_unique(signatures, "missing_entrypoint")

        entrypoint_names = [interface.name for interface in declared_entrypoint_interfaces]
        for entrypoint in code_artifact.runnable_entrypoints:
            result = self._execute_entrypoint(
                workspace,
                materialized,
                entrypoint,
                build_spec,
                entrypoint_names,
            )
            entrypoint_evidence[entrypoint] = result
            if not result.get("exists", False):
                failures.append(f"Declared runnable entrypoint is missing: {entrypoint}.")
                self._append_unique(signatures, "missing_entrypoint")
            elif not result.get("function_present", False):
                failures.append(f"Entrypoint function was not found in {entrypoint}.")
                self._append_unique(signatures, "missing_entrypoint")
            elif not result.get("executed", False):
                failures.append(f"Entrypoint execution failed for {entrypoint}.")
                self._append_unique(
                    signatures,
                    "sandbox_unavailable" if result.get("launch_error") else "import_failure",
                )
        evidence["entrypoint_results"] = entrypoint_evidence
        evidence["failure_signatures"] = signatures

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        passed = len(failures) == 0
        metrics = {
            "duration_ms": elapsed_ms,
            "checked_python_files": len([path for path in materialized if path.endswith(".py")]),
            "imported_modules": len(src_modules),
            "entrypoint_count": len(code_artifact.runnable_entrypoints),
        }
        return ValidationLayerResult(
            layer_name="layer1_syntax_import_run",
            passed=passed,
            failures=failures,
            evidence=evidence,
            metrics=metrics,
        )

    def _import_modules(self, workspace: Path, module_paths: List[str]) -> Tuple[bool, Dict[str, object]]:
        if not module_paths:
            return True, {"modules": {}, "returncode": 0}
        modules = [self._module_name_for_src_path(path) for path in module_paths]
        script = (
            "import importlib\n"
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            "workspace = Path.cwd()\n"
            "sys.path.insert(0, str(workspace / 'src'))\n"
            f"modules = {modules!r}\n"
            "results = {}\n"
            "ok = True\n"
            "for module in modules:\n"
            "    try:\n"
            "        importlib.import_module(module)\n"
            "        results[module] = {'ok': True}\n"
            "    except Exception as exc:\n"
            "        ok = False\n"
            "        results[module] = {'ok': False, 'error': str(exc)}\n"
            "print(json.dumps({'ok': ok, 'modules': results}, sort_keys=True))\n"
        )
        completed = self._run_subprocess(script, cwd=workspace)
        stdout = completed.stdout.strip()
        payload = {"ok": False, "modules": {}, "returncode": completed.returncode}
        if stdout:
            try:
                payload.update(json.loads(stdout.splitlines()[-1]))
            except json.JSONDecodeError:
                payload["raw_stdout"] = stdout
        payload["stderr"] = completed.stderr.strip()
        payload["backend"] = completed.backend
        payload["timed_out"] = completed.timed_out
        payload["launch_error"] = completed.launch_error
        return bool(payload.get("ok", False)) and completed.returncode == 0, payload

    def _execute_entrypoint(
        self,
        workspace: Path,
        materialized: Dict[str, Path],
        entrypoint: str,
        build_spec: BuildSpec,
        declared_entrypoint_names: List[str],
    ) -> Dict[str, object]:
        result = {
            "exists": False,
            "function_present": False,
            "executed": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
        target = materialized.get(entrypoint)
        if target is None or not target.exists():
            return result
        result["exists"] = True

        try:
            tree = ast.parse(target.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            result["stderr"] = str(exc)
            return result

        function_nodes = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        function_names = set(function_nodes)
        candidate = next(
            (name for name in declared_entrypoint_names if name in function_names),
            "",
        )
        if not candidate:
            candidate = "main" if "main" in function_names else ("run" if "run" in function_names else "")
        if not candidate:
            return result
        result["function_present"] = True

        module_name = self._module_name_for_src_path(entrypoint)
        is_jsonl_input = self._is_jsonl_input(build_spec)
        is_jsonl_pipeline = self._is_jsonl_pipeline(build_spec)
        is_json_merge_cli = self._is_json_merge_cli(build_spec)
        input_format = self._input_format(build_spec)
        output_format = self._output_format(build_spec)
        input_path = workspace / f"validator_input.{input_format}"
        output_path = workspace / f"validator_output.{output_format}"
        input_path.write_text(self._sample_input_content(build_spec), encoding="utf-8")
        invocation_args: list[object] = []
        if candidate == "main":
            if is_jsonl_pipeline:
                quarantine_jsonl = workspace / "validator_quarantine.jsonl"
                invocation_args = [[
                    self._workspace_path(input_path, workspace),
                    self._workspace_path(quarantine_jsonl, workspace),
                    self._workspace_path(output_path, workspace),
                ]]
            elif is_json_merge_cli:
                base_json = workspace / "validator_base.json"
                override_json = workspace / "validator_override.json"
                output_json = workspace / "validator_output.json"
                base_json.write_text(
                    '{"service":{"host":"localhost","ports":[80]},"enabled":true}',
                    encoding="utf-8",
                )
                override_json.write_text(
                    '{"service":{"ports":[443]},"enabled":false}',
                    encoding="utf-8",
                )
                invocation_args = [[
                    self._workspace_path(base_json, workspace),
                    self._workspace_path(override_json, workspace),
                    self._workspace_path(output_json, workspace),
                ]]
            elif entrypoint.lower().endswith(("src/cli.py", "src/main.py")):
                invocation_args = [[
                    self._workspace_path(input_path, workspace),
                    self._workspace_path(output_path, workspace),
                ]]
        elif candidate == "run":
            invocation_args = self._run_invocation_arguments(
                function_nodes[candidate],
                workspace,
                input_path,
                output_path,
            )
        result["smoke_contract"] = {
            "input_format": input_format,
            "output_format": output_format,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "argument_count": len(invocation_args),
        }
        script = (
            "import importlib\n"
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            "workspace = Path.cwd()\n"
            "sys.path.insert(0, str(workspace / 'src'))\n"
            f"module = importlib.import_module({module_name!r})\n"
            f"fn = getattr(module, {candidate!r})\n"
            f"invoke_args = {invocation_args!r}\n"
            "result = fn(*invoke_args)\n"
            "print(json.dumps({'result': result if isinstance(result, (int, str, bool, float)) else str(result)}))\n"
        )
        completed = self._run_subprocess(script, cwd=workspace)
        result["returncode"] = completed.returncode
        result["stdout"] = completed.stdout.strip()
        result["stderr"] = completed.stderr.strip()
        result["executed"] = completed.returncode == 0
        result["backend"] = completed.backend
        result["timed_out"] = completed.timed_out
        result["launch_error"] = completed.launch_error
        return result

    def _sample_input_content(self, build_spec: BuildSpec) -> str:
        if self._is_jsonl_input(build_spec):
            atom_text = " ".join(atom.text.lower() for atom in build_spec.requirement_atoms)
            if "application log" in atom_text or "counts_by_level" in atom_text:
                return '{"level":"INFO","message":"validator"}\n'
            if "sensor_id" in atom_text and "numeric value" in atom_text:
                return '{"sensor_id":"sensor-1","value":21.5}\n'
            return '{"device_id":"device-1","timestamp":"2026-01-15T12:00:00Z","temperature_c":21.5}\n'
        atom_text = " ".join(atom.text.lower() for atom in build_spec.requirement_atoms)
        if self._is_json_array_input(build_spec):
            return '[{"id":"validator-a","score":2},{"id":"validator-b","score":1}]\n'
        if "invoice" in atom_text or "due_date" in atom_text:
            return (
                "invoice_id,due_date,amount,customer_name\n"
                "INV-1,2026-01-15,100.00,Acme\n"
            )
        return "contract_id,expiration_date\nA,2026-01-15\n"

    def _sample_input_csv_content(self, build_spec: BuildSpec) -> str:
        return self._sample_input_content(build_spec)

    def _is_jsonl_pipeline(self, build_spec: BuildSpec) -> bool:
        evidence_terms = {
            term
            for atom in build_spec.requirement_atoms
            for term in atom.evidence_terms
        }
        return bool({"input_jsonl", "jsonl"} & evidence_terms) and "quarantine" in evidence_terms

    def _is_jsonl_input(self, build_spec: BuildSpec) -> bool:
        return any(
            bool({"input_jsonl", "jsonl"} & set(atom.evidence_terms))
            for atom in build_spec.requirement_atoms
        )

    def _is_json_merge_cli(self, build_spec: BuildSpec) -> bool:
        evidence_terms = {
            term
            for atom in build_spec.requirement_atoms
            for term in atom.evidence_terms
        }
        return "recursive_json_merge" in evidence_terms

    def _input_format(self, build_spec: BuildSpec) -> str:
        if self._is_jsonl_input(build_spec):
            return "jsonl"
        if self._is_json_array_input(build_spec) or self._is_json_merge_cli(build_spec):
            return "json"
        return "csv"

    def _output_format(self, build_spec: BuildSpec) -> str:
        atom_text = " ".join(atom.text.lower() for atom in build_spec.requirement_atoms)
        if "summary csv" in atom_text or "writes a csv" in atom_text:
            return "csv"
        if re.search(r"\bwrites?\b.{0,120}\bjson\b", atom_text):
            return "json"
        if "summary_json" in atom_text or "summary json" in atom_text:
            return "json"
        return "csv"

    @staticmethod
    def _is_json_array_input(build_spec: BuildSpec) -> bool:
        atom_text = " ".join(atom.text.lower() for atom in build_spec.requirement_atoms)
        return "reads a json array" in atom_text or "reads json array" in atom_text

    @staticmethod
    def _run_invocation_arguments(
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
        workspace: Path,
        input_path: Path,
        output_path: Path,
    ) -> list[object]:
        positional = [
            argument.arg
            for argument in (*function_node.args.posonlyargs, *function_node.args.args)
        ]
        required_count = len(positional) - len(function_node.args.defaults)
        mapped: list[object] = []
        for index, name in enumerate(positional):
            normalized = name.lower()
            value: object | None = None
            if normalized in {"input_path", "source_path", "input_file"}:
                value = RuntimeValidationLayer._workspace_path(input_path, workspace)
            elif normalized in {
                "output_path",
                "summary_path",
                "summary_json_path",
                "summary_csv_path",
                "output_file",
            }:
                value = RuntimeValidationLayer._workspace_path(output_path, workspace)
            elif normalized == "quarantine_path":
                value = "validator_quarantine.jsonl"
            elif normalized in {"watch_dir", "input_dir", "input_directory"}:
                directory = workspace / "validator_input"
                directory.mkdir(exist_ok=True)
                value = RuntimeValidationLayer._workspace_path(directory, workspace)
            elif normalized == "quarantine_dir":
                directory = workspace / "validator_quarantine"
                directory.mkdir(exist_ok=True)
                value = RuntimeValidationLayer._workspace_path(directory, workspace)
            elif normalized == "db_path":
                value = "validator.sqlite3"
            elif normalized == "poll_once":
                value = True
            if value is None:
                if index < required_count:
                    return []
                break
            mapped.append(value)
        return mapped

    def _run_subprocess(self, script: str, cwd: Path) -> SandboxProcessResult:
        return self.executor.run(
            SandboxProcessRequest(
                command=["python", "-c", script],
                workspace=cwd,
                environment={"PYTHONDONTWRITEBYTECODE": "1"},
                timeout_seconds=self.timeout_seconds,
            )
        )

    @staticmethod
    def _workspace_path(path: Path, workspace: Path) -> str:
        return path.relative_to(workspace).as_posix()

    @staticmethod
    def _module_name_for_src_path(path: str) -> str:
        normalized = Path(path.replace("\\", "/")).with_suffix("")
        parts = list(normalized.parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
