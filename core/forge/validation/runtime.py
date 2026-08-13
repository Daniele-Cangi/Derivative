import ast
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

from core.forge.contracts import BuildSpec, CodeArtifact, FeasiblePlan, ValidationLayerResult
from core.forge.validation.common import ValidationLayerBase


class RuntimeValidationLayer(ValidationLayerBase):
    def __init__(self, python_executable: str, timeout_seconds: int):
        self.python_executable = python_executable
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
            self._append_unique(signatures, "import_failure")

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
                self._append_unique(signatures, "import_failure")
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
        modules = [Path(path).stem for path in module_paths]
        script = (
            "import importlib\n"
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            f"workspace = Path({str(workspace)!r})\n"
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

        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        candidate = next(
            (name for name in declared_entrypoint_names if name in function_names),
            "",
        )
        if not candidate:
            candidate = "main" if "main" in function_names else ("run" if "run" in function_names else "")
        if not candidate:
            return result
        result["function_present"] = True

        module_name = target.stem
        is_jsonl_pipeline = self._is_jsonl_pipeline(build_spec)
        is_json_merge_cli = self._is_json_merge_cli(build_spec)
        input_csv = workspace / ("validator_input.jsonl" if is_jsonl_pipeline else "validator_input.csv")
        output_csv = workspace / "validator_output.csv"
        input_csv.write_text(self._sample_input_content(build_spec), encoding="utf-8")
        call_args = ""
        if candidate == "main":
            if is_jsonl_pipeline:
                quarantine_jsonl = workspace / "validator_quarantine.jsonl"
                call_args = (
                    f"[{str(input_csv)!r}, {str(quarantine_jsonl)!r}, "
                    f"{str(output_csv)!r}]"
                )
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
                call_args = (
                    f"[{str(base_json)!r}, {str(override_json)!r}, "
                    f"{str(output_json)!r}]"
                )
            elif entrypoint.lower().endswith(("src/cli.py", "src/main.py")):
                call_args = f"[{str(input_csv)!r}, {str(output_csv)!r}]"
        script = (
            "import importlib\n"
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            f"workspace = Path({str(workspace)!r})\n"
            "sys.path.insert(0, str(workspace / 'src'))\n"
            f"module = importlib.import_module({module_name!r})\n"
            f"fn = getattr(module, {candidate!r})\n"
            f"result = fn({call_args})\n"
            "print(json.dumps({'result': result if isinstance(result, (int, str, bool, float)) else str(result)}))\n"
        )
        completed = self._run_subprocess(script, cwd=workspace)
        result["returncode"] = completed.returncode
        result["stdout"] = completed.stdout.strip()
        result["stderr"] = completed.stderr.strip()
        result["executed"] = completed.returncode == 0
        return result

    def _sample_input_content(self, build_spec: BuildSpec) -> str:
        if self._is_jsonl_pipeline(build_spec):
            return '{"device_id":"device-1","timestamp":"2026-01-15T12:00:00Z","temperature_c":21.5}\n'
        atom_text = " ".join(atom.text.lower() for atom in build_spec.requirement_atoms)
        if "invoice" in atom_text or "due_date" in atom_text:
            return (
                "invoice_id,due_date,amount,customer_name\n"
                "INV-1,2026-01-15,100.00,Acme\n"
            )
        return "contract_id,expiration_date\nA,2026-01-15\n"

    def _sample_input_csv_content(self, build_spec: BuildSpec) -> str:
        return self._sample_input_content(build_spec)

    def _is_jsonl_pipeline(self, build_spec: BuildSpec) -> bool:
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

    def _run_subprocess(self, script: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.python_executable, "-c", script],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
