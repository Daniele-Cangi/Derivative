import ast
import time
from pathlib import Path
from typing import Dict, List, Tuple

from core.forge.cli_contract import cli_invocation_contract_failures
from core.forge.contracts import BuildSpec, CodeArtifact, FeasiblePlan, ValidationLayerResult
from core.forge.requirement_evidence import requirement_assertion_evidence
from core.forge.semantic_contracts import behaviorally_evidences, semantic_term_present
from core.forge.test_evidence import (
    non_semantic_test_reasons,
    source_module_names,
)
from core.forge.validation.common import ValidationLayerBase


class AdversarialValidationLayer(ValidationLayerBase):
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

        interface_contract_mismatches = self._detect_interface_contract_mismatches(plan, materialized)
        if interface_contract_mismatches:
            failures.append(
                "Generated interfaces violate their declared callable contracts: "
                f"{interface_contract_mismatches}."
            )
            self._append_unique(signatures, "interface_contract_mismatch")
        evidence["interface_contract_mismatches"] = interface_contract_mismatches

        expected_test_paths = {f"tests/{test.test_name}.py" for test in plan.required_tests if test.required}
        if not expected_test_paths.issubset(set(code_artifact.test_paths)):
            failures.append("Declared tests do not align with required_tests.")
            self._append_unique(signatures, "missing_acceptance_coverage")

        declared_test_paths = set(code_artifact.test_paths)
        non_semantic_tests, non_semantic_reasons = self._detect_non_semantic_tests(
            declared_test_paths,
            materialized,
            plan,
        )
        if non_semantic_tests:
            failures.append(f"Non-semantic tests detected: {non_semantic_tests}.")
            self._append_unique(signatures, "non_semantic_test")
            self._append_unique(signatures, "fake_acceptance_coverage")
        observed_reasons = {
            reason
            for reasons in non_semantic_reasons.values()
            for reason in reasons
        }
        for signature in (
            "ambiguous_exit_status_assertion",
            "disconnected_assertion",
            "missing_target_invocation",
            "tautological_assertion",
        ):
            if signature in observed_reasons:
                self._append_unique(signatures, signature)
        evidence["non_semantic_tests"] = non_semantic_tests
        evidence["non_semantic_test_reasons"] = non_semantic_reasons

        semantic_requirement_failures, semantic_requirement_signatures, semantic_requirement_evidence = (
            self._validate_semantic_requirement_test_coverage(
                build_spec=build_spec,
                plan=plan,
                expected_test_paths=expected_test_paths,
                non_semantic_tests=non_semantic_tests,
                materialized=materialized,
            )
        )
        failures.extend(semantic_requirement_failures)
        for signature in semantic_requirement_signatures:
            self._append_unique(signatures, signature)
        evidence["semantic_requirement_test_coverage"] = semantic_requirement_evidence

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

    def _detect_interface_contract_mismatches(
        self,
        plan: FeasiblePlan,
        materialized: Dict[str, Path],
    ) -> List[str]:
        contract_interfaces = {
            interface.name
            for interface in plan.interfaces
            if interface.interface_type in {"function", "entrypoint"}
            or (
                interface.interface_type == "cli_entrypoint"
                and interface.module_path
            )
        }
        if not contract_interfaces:
            return []

        mismatches: List[str] = []
        for interface in plan.interfaces:
            if not interface.module_path:
                continue
            expected_path = f"src/{interface.module_path.replace('.', '/')}.py"
            target = materialized.get(expected_path)
            if target is None or not target.exists():
                mismatches.append(f"{expected_path}:{interface.name}:missing_public_module")
                continue
            try:
                tree = ast.parse(target.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            if not self._module_exports_name(tree, interface.name):
                mismatches.append(f"{expected_path}:{interface.name}:missing_public_export")
        for path, target in materialized.items():
            if not path.startswith("src/") or not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(target.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name not in contract_interfaces:
                    continue
                decorators = [self._decorator_name(item).lower() for item in node.decorator_list]
                if any(
                    name.endswith(("click.command", "click.group", "typer.command"))
                    or name in {"command", "click.command", "click.group", "app.command"}
                    for name in decorators
                ):
                    mismatches.append(f"{path}:{node.name}:decorated_cli_command")
        contents = {
            path: target.read_text(encoding="utf-8")
            for path, target in materialized.items()
            if target.exists() and path.endswith(".py")
        }
        for failure in cli_invocation_contract_failures(contents, plan):
            mismatches.append(
                f"{failure['path']}:{failure['interface']}:{failure['reason']}:"
                f"line={failure['line']}"
            )
        return mismatches

    @staticmethod
    def _module_exports_name(tree: ast.AST, name: str) -> bool:
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return True
            if isinstance(node, ast.ImportFrom):
                if any(alias.asname == name or (alias.asname is None and alias.name == name) for alias in node.names):
                    return True
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                    return True
        return False

    def _decorator_name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Call):
            return self._decorator_name(node.func)
        if isinstance(node, ast.Attribute):
            prefix = self._decorator_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        if isinstance(node, ast.Name):
            return node.id
        return ""

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

    def _detect_non_semantic_tests(
        self,
        expected_test_paths: set[str],
        materialized: Dict[str, Path],
        plan: FeasiblePlan,
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        contents = {
            path: target.read_text(encoding="utf-8")
            for path, target in materialized.items()
            if path in expected_test_paths and target.exists()
        }
        reasons = non_semantic_test_reasons(
            expected_test_paths,
            contents,
            target_names={
                interface.name
                for interface in plan.interfaces
                if interface.name.isidentifier()
            },
            target_modules=source_module_names(materialized),
        )
        return sorted(reasons), reasons

    def _validate_semantic_requirement_test_coverage(
        self,
        build_spec: BuildSpec,
        plan: FeasiblePlan,
        expected_test_paths: set[str],
        non_semantic_tests: List[str],
        materialized: Dict[str, Path],
    ) -> Tuple[List[str], List[str], Dict[str, object]]:
        failures: List[str] = []
        signatures: List[str] = []

        non_semantic_set = set(non_semantic_tests)
        requirement_evidence: Dict[str, Dict[str, object]] = {}
        missing_semantic_coverage: List[str] = []
        hard_atoms = [
            atom
            for atom in build_spec.requirement_atoms
            if atom.category != "ambiguity"
            and atom.strength in {"hard", "universal"}
        ]
        file_contents = {
            path: target.read_text(encoding="utf-8")
            for path, target in materialized.items()
            if path in expected_test_paths and target.exists()
        }
        public_interface_names = {
            interface.name
            for interface in plan.interfaces
            if interface.name.isidentifier()
        }
        assertion_report = requirement_assertion_evidence(
            {
                atom.requirement_id: list(atom.evidence_terms)
                for atom in hard_atoms
            },
            {
                atom.requirement_id: [
                    f"tests/{test_name}.py"
                    for test_name in plan.requirement_coverage.get(
                        atom.requirement_id,
                        {},
                    ).get("tests", [])
                ]
                for atom in hard_atoms
            },
            file_contents,
            target_names=public_interface_names,
            target_modules=source_module_names(materialized),
            term_matcher=lambda term, function_source: (
                semantic_term_present(term, function_source, is_test=True)
                or behaviorally_evidences(
                    term,
                    function_source,
                    public_interface_names,
                )
            ),
        )
        missing_assertion_evidence: List[str] = []

        for atom in hard_atoms:

            coverage_entry = plan.requirement_coverage.get(
                atom.requirement_id,
                {"tests": []},
            )
            mapped_test_names = list(coverage_entry.get("tests", []))
            mapped_test_paths = [
                f"tests/{test_name}.py"
                for test_name in mapped_test_names
                if f"tests/{test_name}.py" in expected_test_paths
            ]
            semantic_test_paths = [
                path for path in mapped_test_paths if path not in non_semantic_set
            ]
            has_semantic_test = bool(semantic_test_paths)
            assertion_evidence = assertion_report.get(
                atom.requirement_id,
                {
                    "passed": False,
                    "failure_reason": "missing_mapped_test",
                    "assertions": [],
                },
            )

            requirement_evidence[atom.requirement_id] = {
                "strength": atom.strength,
                "mapped_tests": mapped_test_paths,
                "semantic_tests": semantic_test_paths,
                "non_semantic_tests": [
                    path for path in mapped_test_paths if path in non_semantic_set
                ],
                "has_semantic_test": has_semantic_test,
                "assertion_evidence": assertion_evidence,
            }

            if not has_semantic_test:
                missing_semantic_coverage.append(atom.requirement_id)
            if not assertion_evidence["passed"]:
                missing_assertion_evidence.append(atom.requirement_id)

        if missing_semantic_coverage:
            failures.append(
                "Hard/universal requirements are missing semantic test coverage: "
                f"{missing_semantic_coverage}."
            )
            self._append_unique(signatures, "missing_semantic_requirement_coverage")
        if missing_assertion_evidence:
            failures.append(
                "Hard/universal requirements lack requirement-specific causal assertions: "
                f"{missing_assertion_evidence}."
            )
            self._append_unique(
                signatures,
                "missing_requirement_assertion_evidence",
            )
            self._append_unique(signatures, "fake_acceptance_coverage")

        evidence = {
            "requirements": requirement_evidence,
            "missing_semantic_coverage": missing_semantic_coverage,
            "missing_requirement_assertion_evidence": missing_assertion_evidence,
        }
        return failures, signatures, evidence
