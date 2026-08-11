import ast
import hashlib
import json
from dataclasses import asdict
from typing import Dict, List

from core.forge.contracts import CodeArtifact, FeasiblePlan, GeneratedFile, PlanFile, PlanInterface, PlanTest
from core.forge.domains.base import BaseDomainAdapter, DomainAdapterError
from core.forge.domains.registry import DomainAdapterRegistry


class CoderStageError(Exception):
    """Base error for plan-to-code expansion failures."""


class MalformedPlanError(CoderStageError):
    """Raised when a feasible plan is structurally incomplete for code generation."""


class CoderStage:
    def __init__(self, registry: DomainAdapterRegistry | None = None):
        self.registry = registry or DomainAdapterRegistry()

    def generate(self, plan: FeasiblePlan) -> CodeArtifact:
        self._validate_plan(plan)
        adapter = self.registry.select(plan)

        generated: Dict[str, GeneratedFile] = {}

        for plan_file in plan.file_tree_plan:
            file_obj = self._generate_from_plan_file(plan, plan_file, adapter)
            generated[file_obj.path] = file_obj

        for plan_test in plan.required_tests:
            test_file = self._generate_from_test_requirement(plan, plan_test, adapter)
            generated[test_file.path] = test_file

        runnable_entrypoints = self._resolve_runnable_entrypoints(plan, generated)
        manifest_path = "forge_artifact_manifest.json"
        traceability = {
            path: list(file.generated_from_plan_sections)
            for path, file in generated.items()
        }
        artifact_manifest = self._build_manifest(
            plan,
            generated,
            runnable_entrypoints,
            traceability,
            manifest_path,
            adapter.name,
        )
        generated[manifest_path] = GeneratedFile(
            path=manifest_path,
            content=json.dumps(artifact_manifest, indent=2, sort_keys=True),
            kind="manifest",
            generated_from_plan_sections=[
                f"plan:{plan.plan_id}",
                f"obligation_mode:{plan.obligation_mode}",
                "artifact_manifest",
            ],
        )
        traceability[manifest_path] = list(generated[manifest_path].generated_from_plan_sections)

        sorted_files = [generated[path] for path in sorted(generated.keys())]
        test_paths = sorted(path for path in generated.keys() if path.startswith("tests/"))
        manifest_paths = [manifest_path]
        artifact_id = self._artifact_id(plan.plan_id)
        return CodeArtifact(
            artifact_id=artifact_id,
            plan_id=plan.plan_id,
            files=sorted_files,
            test_paths=test_paths,
            manifest_paths=manifest_paths,
            runnable_entrypoints=runnable_entrypoints,
            artifact_manifest=artifact_manifest,
            traceability=traceability,
        )

    def _validate_plan(self, plan: FeasiblePlan) -> None:
        if not plan.plan_id.strip():
            raise MalformedPlanError("FeasiblePlan.plan_id is required.")
        if not plan.file_tree_plan:
            raise MalformedPlanError("FeasiblePlan.file_tree_plan is required.")
        if not plan.interfaces:
            raise MalformedPlanError("FeasiblePlan.interfaces is required.")
        if not plan.required_tests:
            raise MalformedPlanError("FeasiblePlan.required_tests is required.")

        missing_paths = [file.path for file in plan.file_tree_plan if not file.path.strip()]
        if missing_paths:
            raise MalformedPlanError("All planned files must have non-empty paths.")

        cli_interfaces = [interface for interface in plan.interfaces if interface.interface_type == "cli_entrypoint"]
        if cli_interfaces:
            has_cli_file = any(
                path.path in {"src/cli.py", "src/main.py"} for path in plan.file_tree_plan
            )
            if not has_cli_file:
                raise MalformedPlanError(
                    "Plan declares cli_entrypoint interface but lacks src/cli.py or src/main.py."
                )

    def _artifact_id(self, plan_id: str) -> str:
        digest = hashlib.sha256(plan_id.encode("utf-8")).hexdigest()[:12]
        return f"code-{digest}"

    def _generate_from_plan_file(
        self,
        plan: FeasiblePlan,
        plan_file: PlanFile,
        adapter: BaseDomainAdapter,
    ) -> GeneratedFile:
        path = plan_file.path
        try:
            content = adapter.render_file(plan, path, plan.interfaces)
        except DomainAdapterError as exc:
            raise MalformedPlanError(str(exc)) from exc
        generated_from = [
            f"plan_file:{path}",
            f"plan_purpose:{plan_file.purpose}",
        ]
        interface_refs = self._interfaces_for_path(path, plan.interfaces)
        generated_from.extend(f"interface:{name}" for name in interface_refs)
        generated_from.extend(f"requirement:{requirement_id}" for requirement_id in plan_file.source_requirement_refs)
        return GeneratedFile(
            path=path,
            content=content,
            kind=self._infer_kind(path),
            generated_from_plan_sections=generated_from,
        )

    def _generate_from_test_requirement(
        self,
        plan: FeasiblePlan,
        plan_test: PlanTest,
        adapter: BaseDomainAdapter,
    ) -> GeneratedFile:
        path = f"tests/{plan_test.test_name}.py"
        try:
            content = adapter.render_test(plan, plan_test)
        except DomainAdapterError as exc:
            raise MalformedPlanError(str(exc)) from exc
        generated_from = [f"test_requirement:{plan_test.test_name}"]
        generated_from.extend(f"acceptance:{criterion_id}" for criterion_id in plan_test.acceptance_criterion_ids)
        generated_from.extend(f"obligation:{field}" for field in plan_test.obligation_fields)
        generated_from.extend(f"requirement:{requirement_id}" for requirement_id in plan_test.requirement_ids)
        return GeneratedFile(
            path=path,
            content=content,
            kind="python_test",
            generated_from_plan_sections=generated_from,
        )

    def _resolve_runnable_entrypoints(
        self,
        plan: FeasiblePlan,
        generated_files: Dict[str, GeneratedFile],
    ) -> List[str]:
        entrypoints: List[str] = []
        for interface in plan.interfaces:
            if interface.interface_type not in {"cli_entrypoint", "entrypoint"}:
                continue
            matching_paths: List[str] = []
            for path, generated_file in sorted(generated_files.items()):
                if not path.startswith("src/") or not path.endswith(".py"):
                    continue
                try:
                    tree = ast.parse(generated_file.content)
                except SyntaxError:
                    continue
                function_names = {
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                }
                if interface.name in function_names:
                    matching_paths.append(path)
            if not matching_paths:
                raise MalformedPlanError(
                    f"Unable to resolve runnable entrypoint for interface '{interface.name}'."
                )
            entrypoints.append(matching_paths[0])
        deduplicated: List[str] = []
        for entrypoint in entrypoints:
            if entrypoint not in deduplicated:
                deduplicated.append(entrypoint)
        return deduplicated

    def _build_manifest(
        self,
        plan: FeasiblePlan,
        generated_files: Dict[str, GeneratedFile],
        runnable_entrypoints: List[str],
        traceability: Dict[str, List[str]],
        manifest_path: str,
        domain_adapter_name: str,
    ) -> Dict[str, object]:
        return {
            "plan_id": plan.plan_id,
            "build_id": plan.build_spec.build_id,
            "architecture_summary": plan.architecture_summary,
            "packaging_target": plan.packaging_target,
            "obligation_mode": plan.obligation_mode,
            "required_obligations": list(plan.required_obligations),
            "acceptance_criterion_ids": list(plan.acceptance_criterion_ids),
            "requirement_coverage": plan.requirement_coverage,
            "quality_contract": asdict(plan.quality_contract),
            "validation_strategy": asdict(plan.validation_strategy),
            "runnable_entrypoints": runnable_entrypoints,
            "generated_file_count": len(generated_files) + 1,
            "generated_files": [
                {
                    "path": file.path,
                    "kind": file.kind,
                    "generated_from": list(file.generated_from_plan_sections),
                }
                for file in [generated_files[path] for path in sorted(generated_files.keys())]
            ],
            "traceability": traceability,
            "manifest_path": manifest_path,
            "metadata": {
                "generator": "forge_coder_stage",
                "deterministic_templates": True,
                "domain_adapter": domain_adapter_name,
            },
        }

    def _interfaces_for_path(self, path: str, interfaces: List[PlanInterface]) -> List[str]:
        refs: List[str] = []
        lowered = path.lower()
        for interface in interfaces:
            name = interface.name.lower()
            if interface.interface_type == "cli_entrypoint" and lowered in {"src/cli.py", "src/main.py"}:
                refs.append(interface.name)
                continue
            if name in lowered:
                refs.append(interface.name)
                continue
            if "contracts_csv" in lowered and "load_contracts_csv" == interface.name:
                refs.append(interface.name)
                continue
            if "expiration_rules" in lowered and "flag_expiring_contracts" == interface.name:
                refs.append(interface.name)
                continue
            if "summary_writer" in lowered and "write_summary_csv" == interface.name:
                refs.append(interface.name)
                continue
        return refs

    def _infer_kind(self, path: str) -> str:
        if path.endswith(".py"):
            if path.startswith("tests/"):
                return "python_test"
            return "python_module"
        if path.endswith(".json"):
            return "json"
        return "text"
