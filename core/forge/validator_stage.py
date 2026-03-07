import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

from core.forge.contracts import (
    BuildSpec,
    CodeArtifact,
    FailureCategory,
    FeasiblePlan,
    ValidationArtifact,
    ValidationLayerResult,
)


class ValidatorStage:
    def __init__(self, python_executable: str | None = None, timeout_seconds: int = 120):
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = timeout_seconds

    def validate(
        self,
        code_artifact: CodeArtifact,
        plan: FeasiblePlan,
        build_spec: BuildSpec,
    ) -> ValidationArtifact:
        failures: List[str] = []
        signatures: List[str] = []
        evidence: Dict[str, object] = {}
        metrics: Dict[str, object] = {}

        with tempfile.TemporaryDirectory(prefix="forge_validator_") as tmp_dir:
            workspace = Path(tmp_dir)
            materialized = self._materialize_workspace(code_artifact, workspace)
            evidence["workspace"] = str(workspace)
            evidence["materialized_files"] = sorted(str(path) for path in materialized.values())

            layer1 = self._validate_layer1(code_artifact, plan, materialized, workspace)
            layer2 = self._validate_layer2(code_artifact, plan, build_spec, materialized, workspace)
            layer3 = self._validate_layer3(code_artifact, plan, build_spec, materialized, workspace)

            for layer in (layer1, layer2, layer3):
                failures.extend(layer.failures)
                for signature in layer.evidence.get("failure_signatures", []):
                    self._append_unique(signatures, str(signature))

            evidence["layer1"] = layer1.evidence
            evidence["layer2"] = layer2.evidence
            evidence["layer3"] = layer3.evidence
            metrics["layer1"] = layer1.metrics
            metrics["layer2"] = layer2.metrics
            metrics["layer3"] = layer3.metrics

        passed = layer1.passed and layer2.passed and layer3.passed
        metrics["failure_count"] = len(failures)
        metrics["failure_signature_count"] = len(signatures)
        metrics["passed_layers"] = {
            "layer1": layer1.passed,
            "layer2": layer2.passed,
            "layer3": layer3.passed,
        }
        structured_evidence = self._build_structured_evidence(layer1, layer2, layer3)
        evidence["validated_entrypoints"] = structured_evidence["validated_entrypoints"]
        evidence["executed_tests"] = structured_evidence["executed_tests"]
        evidence["manifest_provenance_checks"] = structured_evidence["manifest_provenance_checks"]
        evidence["obligation_acceptance_checks"] = structured_evidence["obligation_acceptance_checks"]
        evidence["layer_status"] = {
            "layer1": layer1.passed,
            "layer2": layer2.passed,
            "layer3": layer3.passed,
        }
        evidence["failure_signatures"] = list(signatures)
        return ValidationArtifact(
            passed=passed,
            failures=failures,
            failure_signatures=signatures,
            evidence=evidence,
            metrics=metrics,
            layer1_result=layer1,
            layer2_result=layer2,
            layer3_result=layer3,
            failure_category=None if passed else self._classify_failure_category(signatures),
            next_route=None,
        )

    def _materialize_workspace(
        self,
        code_artifact: CodeArtifact,
        workspace: Path,
    ) -> Dict[str, Path]:
        materialized: Dict[str, Path] = {}
        for generated_file in code_artifact.files:
            target = workspace / generated_file.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(generated_file.content, encoding="utf-8")
            materialized[generated_file.path] = target
        return materialized

    def _validate_layer1(
        self,
        code_artifact: CodeArtifact,
        plan: FeasiblePlan,
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
        declared_cli_interfaces = [i for i in plan.interfaces if i.interface_type == "cli_entrypoint"]
        if declared_cli_interfaces and not code_artifact.runnable_entrypoints:
            failures.append("CLI entrypoint interface declared but no runnable_entrypoints were provided.")
            self._append_unique(signatures, "missing_entrypoint")

        for entrypoint in code_artifact.runnable_entrypoints:
            result = self._execute_entrypoint(workspace, materialized, entrypoint)
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

    def _validate_layer2(
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
        evidence: Dict[str, object] = {}

        actual_paths = set(materialized.keys())
        required_paths = {plan_file.path for plan_file in plan.file_tree_plan}
        missing_required_files = sorted(required_paths - actual_paths)
        if missing_required_files:
            failures.append(f"Missing required plan files: {missing_required_files}.")
            self._append_unique(signatures, "missing_required_file")
        evidence["missing_required_files"] = missing_required_files

        expected_test_paths = {f"tests/{test.test_name}.py" for test in plan.required_tests if test.required}
        artifact_test_paths = set(code_artifact.test_paths)
        missing_required_tests = sorted(expected_test_paths - artifact_test_paths)
        if missing_required_tests:
            failures.append(f"Required tests are missing from artifact.test_paths: {missing_required_tests}.")
            self._append_unique(signatures, "missing_acceptance_coverage")
        evidence["missing_required_tests"] = missing_required_tests

        manifest_required_obligations = set(
            code_artifact.artifact_manifest.get("required_obligations", [])
            if isinstance(code_artifact.artifact_manifest, dict)
            else []
        )
        missing_manifest_obligations = sorted(set(plan.required_obligations) - manifest_required_obligations)
        if missing_manifest_obligations:
            failures.append(
                "Manifest does not declare required obligations: "
                f"{missing_manifest_obligations}."
            )
            self._append_unique(signatures, "missing_obligation")
        evidence["missing_manifest_obligations"] = missing_manifest_obligations

        provenance_obligations = self._collect_prefixed_tokens(code_artifact, prefix="obligation:")
        missing_provenance_obligations = sorted(set(plan.required_obligations) - provenance_obligations)
        if missing_provenance_obligations:
            failures.append(
                "Provenance does not cover required obligations: "
                f"{missing_provenance_obligations}."
            )
            self._append_unique(signatures, "missing_obligation")
        evidence["missing_provenance_obligations"] = missing_provenance_obligations

        required_acceptance = set(plan.acceptance_criterion_ids)
        provenance_acceptance = self._collect_prefixed_tokens(code_artifact, prefix="acceptance:")
        missing_acceptance = sorted(required_acceptance - provenance_acceptance)
        if missing_acceptance:
            failures.append(f"Missing acceptance provenance coverage: {missing_acceptance}.")
            self._append_unique(signatures, "missing_acceptance_coverage")
        evidence["missing_acceptance_coverage"] = missing_acceptance

        requirement_failures, requirement_signatures, requirement_evidence = self._validate_requirement_coverage(
            build_spec=build_spec,
            plan=plan,
            code_artifact=code_artifact,
        )
        failures.extend(requirement_failures)
        for signature in requirement_signatures:
            self._append_unique(signatures, signature)
        evidence["requirement_coverage_checks"] = requirement_evidence

        test_result = self._run_required_tests(workspace, expected_test_paths, actual_paths)
        evidence["test_execution"] = test_result
        if not test_result["ran"]:
            failures.append("Required tests were not executed.")
            self._append_unique(signatures, "test_execution_failure")
        elif test_result["returncode"] != 0:
            failures.append("Required test execution failed.")
            self._append_unique(signatures, "test_execution_failure")

        evidence["failure_signatures"] = signatures
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        passed = len(failures) == 0
        metrics = {
            "duration_ms": elapsed_ms,
            "required_file_count": len(required_paths),
            "required_test_count": len(expected_test_paths),
            "required_obligation_count": len(plan.required_obligations),
            "required_acceptance_count": len(plan.acceptance_criterion_ids),
            "build_id": build_spec.build_id,
        }
        return ValidationLayerResult(
            layer_name="layer2_obligations_tests_acceptance",
            passed=passed,
            failures=failures,
            evidence=evidence,
            metrics=metrics,
        )

    def _validate_layer3(
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
        evidence: Dict[str, object] = {}

        actual_paths = set(materialized.keys())
        manifest_generated = code_artifact.artifact_manifest.get("generated_files", [])
        manifest_paths = set()
        if isinstance(manifest_generated, list):
            for item in manifest_generated:
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    manifest_paths.add(item["path"])
        missing_manifest_files = sorted(path for path in manifest_paths if path not in actual_paths)
        if missing_manifest_files:
            failures.append(f"Manifest references missing files: {missing_manifest_files}.")
            self._append_unique(signatures, "missing_required_file")
            self._append_unique(signatures, "manifest_mismatch")
        evidence["manifest_missing_files"] = missing_manifest_files

        provenance_mismatches: List[str] = []
        for generated_file in code_artifact.files:
            expected = generated_file.generated_from_plan_sections
            observed = code_artifact.traceability.get(generated_file.path)
            if observed != expected:
                provenance_mismatches.append(generated_file.path)
        extra_traceability = sorted(path for path in code_artifact.traceability.keys() if path not in actual_paths)
        if provenance_mismatches or extra_traceability:
            failures.append(
                "Traceability map mismatches generated file provenance."
            )
            self._append_unique(signatures, "provenance_mismatch")
        evidence["provenance_mismatches"] = provenance_mismatches
        evidence["traceability_extras"] = extra_traceability

        missing_entrypoint_declarations: List[str] = []
        for interface in plan.interfaces:
            if interface.interface_type != "cli_entrypoint":
                continue
            exists, has_function = self._interface_declared_in_entrypoint(interface.name, code_artifact, materialized)
            if not exists or not has_function:
                missing_entrypoint_declarations.append(interface.name)
        if missing_entrypoint_declarations:
            failures.append(
                "Entrypoint interfaces are not implemented in declared entrypoints: "
                f"{missing_entrypoint_declarations}."
            )
            self._append_unique(signatures, "missing_entrypoint")
        evidence["missing_entrypoint_interfaces"] = missing_entrypoint_declarations

        expected_test_paths = {f"tests/{test.test_name}.py" for test in plan.required_tests if test.required}
        if not expected_test_paths.issubset(set(code_artifact.test_paths)):
            failures.append("Declared tests do not align with required_tests.")
            self._append_unique(signatures, "missing_acceptance_coverage")

        non_semantic_tests = self._detect_non_semantic_tests(expected_test_paths, materialized)
        if non_semantic_tests:
            failures.append(f"Non-semantic tests detected: {non_semantic_tests}.")
            self._append_unique(signatures, "non_semantic_test")
            self._append_unique(signatures, "fake_acceptance_coverage")
        evidence["non_semantic_tests"] = non_semantic_tests

        superficial_interfaces = self._detect_superficial_interfaces(plan, materialized)
        if superficial_interfaces:
            failures.append(
                f"Core workflow appears superficial for interfaces: {superficial_interfaces}."
            )
            self._append_unique(signatures, "superficial_stub")
        evidence["superficial_interfaces"] = superficial_interfaces

        evidence["failure_signatures"] = signatures
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        passed = len(failures) == 0
        metrics = {
            "duration_ms": elapsed_ms,
            "manifest_declared_file_count": len(manifest_paths),
            "actual_file_count": len(actual_paths),
            "provenance_mismatch_count": len(provenance_mismatches) + len(extra_traceability),
            "interface_count": len(plan.interfaces),
            "build_id": build_spec.build_id,
        }
        return ValidationLayerResult(
            layer_name="layer3_adversarial_attack",
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
        candidate = "main" if "main" in function_names else ("run" if "run" in function_names else "")
        if not candidate:
            return result
        result["function_present"] = True

        module_name = target.stem
        input_csv = workspace / "validator_input.csv"
        output_csv = workspace / "validator_output.csv"
        input_csv.write_text(
            "contract_id,expiration_date\nA,2026-01-15\n",
            encoding="utf-8",
        )
        call_args = "[]"
        if candidate == "main" and entrypoint.lower().endswith(("src/cli.py", "src/main.py")):
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

    def _run_required_tests(
        self,
        workspace: Path,
        expected_test_paths: set[str],
        actual_paths: set[str],
    ) -> Dict[str, object]:
        runnable = sorted(path for path in expected_test_paths if path in actual_paths)
        if not runnable:
            return {
                "ran": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "tests": [],
            }
        command = [
            self.python_executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *runnable,
        ]
        completed = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        return {
            "ran": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "tests": runnable,
        }

    def _collect_prefixed_tokens(self, artifact: CodeArtifact, prefix: str) -> set[str]:
        tokens: set[str] = set()
        for generated in artifact.files:
            for section in generated.generated_from_plan_sections:
                if section.startswith(prefix):
                    tokens.add(section[len(prefix):])
        return tokens

    def _interface_declared_in_entrypoint(
        self,
        interface_name: str,
        code_artifact: CodeArtifact,
        materialized: Dict[str, Path],
    ) -> Tuple[bool, bool]:
        for entrypoint in code_artifact.runnable_entrypoints:
            target = materialized.get(entrypoint)
            if target is None or not target.exists():
                continue
            exists = True
            try:
                tree = ast.parse(target.read_text(encoding="utf-8"))
            except SyntaxError:
                return exists, False
            names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            }
            if interface_name in names:
                return exists, True
            if interface_name == "main" and "main" in names:
                return exists, True
            return exists, False
        return False, False

    def _detect_superficial_interfaces(
        self,
        plan: FeasiblePlan,
        materialized: Dict[str, Path],
    ) -> List[str]:
        python_sources: Dict[str, ast.AST] = {}
        for path, target in materialized.items():
            if not path.endswith(".py"):
                continue
            try:
                python_sources[path] = ast.parse(target.read_text(encoding="utf-8"))
            except SyntaxError:
                continue

        superficial: List[str] = []
        for interface in plan.interfaces:
            if interface.interface_type not in {"cli_entrypoint", "function", "entrypoint"}:
                continue
            nodes: List[ast.FunctionDef] = []
            for tree in python_sources.values():
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == interface.name:
                        nodes.append(node)
            if not nodes:
                self._append_unique(superficial, interface.name)
                continue
            has_nontrivial_call = False
            for node in nodes:
                calls = [inner for inner in ast.walk(node) if isinstance(inner, ast.Call)]
                if calls:
                    has_nontrivial_call = True
                    break
            if not has_nontrivial_call:
                self._append_unique(superficial, interface.name)
        return superficial

    def _run_subprocess(self, script: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.python_executable, "-c", script],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )

    def _classify_failure_category(self, signatures: List[str]) -> FailureCategory | None:
        signature_set = set(signatures)
        if not signature_set:
            return None
        if {"missing_required_file", "manifest_mismatch", "provenance_mismatch"} & signature_set:
            return FailureCategory.ARCHITECTURAL
        if {
            "missing_obligation",
            "missing_acceptance_coverage",
            "semantic_omission",
            "missing_requirement_coverage",
            "universal_constraint_unproven",
            "non_semantic_test",
            "fake_acceptance_coverage",
        } & signature_set:
            return FailureCategory.VALIDATION
        if {"syntax_error", "import_failure", "missing_entrypoint", "test_execution_failure", "superficial_stub"} & signature_set:
            return FailureCategory.IMPLEMENTATION
        return FailureCategory.UNKNOWN

    def _append_unique(self, collection: List[str], value: str) -> None:
        if value not in collection:
            collection.append(value)

    def _build_structured_evidence(
        self,
        layer1: ValidationLayerResult,
        layer2: ValidationLayerResult,
        layer3: ValidationLayerResult,
    ) -> Dict[str, object]:
        entrypoint_results = layer1.evidence.get("entrypoint_results", {})
        validated_entrypoints: Dict[str, object] = {}
        if isinstance(entrypoint_results, dict):
            for path, result in entrypoint_results.items():
                if not isinstance(result, dict):
                    continue
                validated_entrypoints[str(path)] = {
                    "exists": bool(result.get("exists", False)),
                    "function_present": bool(result.get("function_present", False)),
                    "executed": bool(result.get("executed", False)),
                    "returncode": result.get("returncode"),
                }

        test_execution = layer2.evidence.get("test_execution", {})
        if not isinstance(test_execution, dict):
            test_execution = {}

        manifest_provenance_checks = {
            "manifest_missing_files": layer3.evidence.get("manifest_missing_files", []),
            "provenance_mismatches": layer3.evidence.get("provenance_mismatches", []),
            "traceability_extras": layer3.evidence.get("traceability_extras", []),
            "missing_entrypoint_interfaces": layer3.evidence.get("missing_entrypoint_interfaces", []),
            "superficial_interfaces": layer3.evidence.get("superficial_interfaces", []),
            "non_semantic_tests": layer3.evidence.get("non_semantic_tests", []),
        }

        obligation_acceptance_checks = {
            "missing_required_files": layer2.evidence.get("missing_required_files", []),
            "missing_required_tests": layer2.evidence.get("missing_required_tests", []),
            "missing_manifest_obligations": layer2.evidence.get("missing_manifest_obligations", []),
            "missing_provenance_obligations": layer2.evidence.get("missing_provenance_obligations", []),
            "missing_acceptance_coverage": layer2.evidence.get("missing_acceptance_coverage", []),
            "requirement_coverage_checks": layer2.evidence.get("requirement_coverage_checks", {}),
        }

        return {
            "validated_entrypoints": validated_entrypoints,
            "executed_tests": {
                "ran": bool(test_execution.get("ran", False)),
                "returncode": test_execution.get("returncode"),
                "tests": test_execution.get("tests", []),
                "stdout": test_execution.get("stdout", ""),
                "stderr": test_execution.get("stderr", ""),
            },
            "manifest_provenance_checks": manifest_provenance_checks,
            "obligation_acceptance_checks": obligation_acceptance_checks,
        }

    def _validate_requirement_coverage(
        self,
        build_spec: BuildSpec,
        plan: FeasiblePlan,
        code_artifact: CodeArtifact,
    ) -> Tuple[List[str], List[str], Dict[str, object]]:
        failures: List[str] = []
        signatures: List[str] = []
        evidence: Dict[str, object] = {"requirements": {}}

        atoms = [atom for atom in build_spec.requirement_atoms if atom.category != "ambiguity"]
        artifact_requirement_ids = self._collect_prefixed_tokens(code_artifact, prefix="requirement:")
        universal_proofs = self._collect_prefixed_tokens(code_artifact, prefix="universal_proof:")
        acceptance_requirement_ids = set()
        for criterion in build_spec.acceptance_contract.criteria:
            acceptance_requirement_ids.update(criterion.requirement_ids)

        semantic_omissions: List[str] = []
        missing_coverage: List[str] = []
        universal_unproven: List[str] = []
        for atom in atoms:
            coverage_entry = plan.requirement_coverage.get(
                atom.requirement_id,
                {"files": [], "tests": [], "acceptance_criteria": []},
            )
            files = list(coverage_entry.get("files", []))
            tests = list(coverage_entry.get("tests", []))
            acceptance = list(coverage_entry.get("acceptance_criteria", []))

            has_plan_mapping = bool(files or tests or acceptance)
            has_artifact_mapping = atom.requirement_id in artifact_requirement_ids
            has_acceptance_mapping = atom.requirement_id in acceptance_requirement_ids and bool(acceptance)
            has_test_mapping = bool(tests)

            evidence["requirements"][atom.requirement_id] = {
                "text": atom.text,
                "category": atom.category,
                "strength": atom.strength,
                "files": files,
                "tests": tests,
                "acceptance_criteria": acceptance,
                "has_plan_mapping": has_plan_mapping,
                "has_artifact_mapping": has_artifact_mapping,
                "has_acceptance_mapping": has_acceptance_mapping,
            }

            if not has_plan_mapping or not has_artifact_mapping:
                semantic_omissions.append(atom.requirement_id)
            if not has_test_mapping or not has_acceptance_mapping:
                missing_coverage.append(atom.requirement_id)
            if atom.strength == "universal" and atom.requirement_id not in universal_proofs:
                universal_unproven.append(atom.requirement_id)

        if semantic_omissions:
            failures.append(f"Semantic omission detected for requirements: {semantic_omissions}.")
            self._append_unique(signatures, "semantic_omission")
        if missing_coverage:
            failures.append(f"Missing requirement coverage for requirements: {missing_coverage}.")
            self._append_unique(signatures, "missing_requirement_coverage")
        if universal_unproven:
            failures.append(
                "Universal constraints are unproven and fail closed: "
                f"{universal_unproven}."
            )
            self._append_unique(signatures, "universal_constraint_unproven")

        evidence["semantic_omissions"] = semantic_omissions
        evidence["missing_coverage"] = missing_coverage
        evidence["universal_unproven"] = universal_unproven
        return failures, signatures, evidence

    def _detect_non_semantic_tests(
        self,
        expected_test_paths: set[str],
        materialized: Dict[str, Path],
    ) -> List[str]:
        non_semantic: List[str] = []
        for test_path in sorted(expected_test_paths):
            target = materialized.get(test_path)
            if target is None or not target.exists():
                continue
            try:
                tree = ast.parse(target.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            test_functions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            ]
            if not test_functions:
                self._append_unique(non_semantic, test_path)
                continue
            file_non_semantic = True
            for function in test_functions:
                has_call = any(isinstance(node, ast.Call) for node in ast.walk(function))
                has_assert_true = any(
                    isinstance(node, ast.Assert)
                    and isinstance(node.test, ast.Constant)
                    and node.test.value is True
                    for node in ast.walk(function)
                )
                is_placeholder_name = function.name in {"test_acceptance_requirement", "test_stub"}
                if is_placeholder_name or has_assert_true:
                    continue
                if has_call:
                    file_non_semantic = False
                    break
            if file_non_semantic:
                self._append_unique(non_semantic, test_path)
        return non_semantic
