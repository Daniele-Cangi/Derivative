import ast
import copy
import hashlib
import json
from dataclasses import asdict
from typing import Dict, List

from core.forge.contracts import (
    CodeArtifact,
    FeasiblePlan,
    GeneratedFile,
    PlanFile,
    PlanInterface,
    PlanTest,
    RepairDirective,
    RepairResult,
    ValidationArtifact,
)
from core.forge.domains.base import BaseDomainAdapter, DomainAdapterError
from core.forge.domains.registry import DomainAdapterRegistry
from core.forge.repair_backend import ArtifactRepairBackend
from core.forge.validation.adapter_capabilities import AdapterCapabilityContractChecker


class CoderStageError(Exception):
    """Base error for plan-to-code expansion failures."""


class MalformedPlanError(CoderStageError):
    """Raised when a feasible plan is structurally incomplete for code generation."""


class CoderStage:
    def __init__(
        self,
        registry: DomainAdapterRegistry | None = None,
        repair_backend: ArtifactRepairBackend | None = None,
        candidate_compiler: ArtifactRepairBackend | None = None,
    ):
        self.registry = registry or DomainAdapterRegistry()
        self.repair_backend = repair_backend
        self.candidate_compiler = candidate_compiler

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
            sorted(adapter.provided_capabilities(plan)),
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

    def repair(
        self,
        plan: FeasiblePlan,
        previous_artifact: CodeArtifact,
        validation: ValidationArtifact,
        directive: RepairDirective,
    ) -> RepairResult:
        previous_digest = self._artifact_digest(previous_artifact)
        if not directive.repairable:
            return RepairResult(
                directive=directive,
                artifact=previous_artifact,
                changed=False,
                previous_digest=previous_digest,
                repaired_digest=previous_digest,
                stop_reason=directive.stop_reason,
            )

        canonical = self.generate(plan)
        changed_paths = self._changed_paths(previous_artifact, canonical)
        candidate_compilation = any(
            operation in directive.operations
            for operation in (
                "compile_uncovered_capabilities",
                "recompile_candidate_transaction",
            )
        )
        if previous_artifact.revision == 1 and changed_paths and not candidate_compilation:
            return self._finalize_repair(
                plan=plan,
                previous_artifact=previous_artifact,
                repaired_artifact=canonical,
                directive=directive,
                changed_paths=changed_paths,
                previous_digest=previous_digest,
                backend_name="canonical",
                backend_evidence={"strategy": "deterministic_plan_regeneration"},
            )

        backend = self.candidate_compiler if candidate_compilation else self.repair_backend
        if backend is None:
            return RepairResult(
                directive=directive,
                artifact=previous_artifact,
                changed=False,
                previous_digest=previous_digest,
                repaired_digest=previous_digest,
                stop_reason="Canonical regeneration produced no material changes.",
            )

        try:
            candidate = backend.propose(
                plan,
                previous_artifact,
                validation,
                directive,
            )
        except Exception as exc:
            return RepairResult(
                directive=directive,
                artifact=previous_artifact,
                changed=False,
                previous_digest=previous_digest,
                repaired_digest=previous_digest,
                backend_name=type(backend).__name__,
                backend_evidence={"error_type": type(exc).__name__},
                stop_reason="Grounded repair backend failed before producing a candidate.",
            )
        candidate_evidence = dict(candidate.evidence)
        candidate_evidence.setdefault("available", candidate.available)
        candidate_evidence.setdefault("rejected_paths", list(candidate.rejected_paths))
        revised = copy.deepcopy(previous_artifact)
        files_by_path = {generated.path.replace("\\", "/"): generated for generated in revised.files}
        allowed_targets = {
            path.replace("\\", "/") for path in directive.target_paths
        }
        impact_expanded_paths = {
            str(path).replace("\\", "/")
            for path in candidate_evidence.get("impact_expanded_paths", [])
        }
        if any(path.startswith("src/") for path in allowed_targets):
            allowed_targets.update(
                path
                for path in impact_expanded_paths
                if path in previous_artifact.test_paths
            )
        manifest_paths = {
            path.replace("\\", "/") for path in revised.manifest_paths
        }
        backend_changed_paths: List[str] = []
        for raw_path, content in candidate.files.items():
            path = raw_path.replace("\\", "/")
            generated = files_by_path.get(path)
            if generated is None or path not in allowed_targets or path in manifest_paths:
                continue
            if generated.content == content:
                continue
            generated.content = content
            backend_changed_paths.append(generated.path)

        if not backend_changed_paths:
            return RepairResult(
                directive=directive,
                artifact=previous_artifact,
                changed=False,
                previous_digest=previous_digest,
                repaired_digest=previous_digest,
                backend_name=candidate.backend_name,
                backend_evidence=candidate_evidence,
                stop_reason=candidate.stop_reason or "Backend revision produced no material changes.",
            )

        return self._finalize_repair(
            plan=plan,
            previous_artifact=previous_artifact,
            repaired_artifact=revised,
            directive=directive,
            changed_paths=backend_changed_paths,
            previous_digest=previous_digest,
            backend_name=candidate.backend_name,
            backend_evidence=candidate_evidence,
        )

    def _finalize_repair(
        self,
        plan: FeasiblePlan,
        previous_artifact: CodeArtifact,
        repaired_artifact: CodeArtifact,
        directive: RepairDirective,
        changed_paths: List[str],
        previous_digest: str,
        backend_name: str,
        backend_evidence: Dict[str, object],
    ) -> RepairResult:
        repaired_artifact.revision = max(1, previous_artifact.revision) + 1
        repaired_artifact.parent_artifact_id = previous_artifact.artifact_id
        repaired_artifact.artifact_id = (
            f"{self._artifact_id(plan.plan_id)}-r{repaired_artifact.revision:02d}"
        )
        repair_record = {
            "repair_id": directive.repair_id,
            "attempt": directive.attempt,
            "route": directive.route.value,
            "failure_signatures": list(directive.failure_signatures),
            "target_paths": list(directive.target_paths),
            "operations": list(directive.operations),
            "evidence_refs": list(directive.evidence_refs),
            "changed_paths": list(changed_paths),
            "previous_artifact_id": previous_artifact.artifact_id,
            "previous_digest": previous_digest,
            "backend_name": backend_name,
            "backend_evidence": backend_evidence,
        }
        repaired_artifact.repair_history = [*previous_artifact.repair_history, repair_record]
        if backend_name == "candidate_compiler":
            self._stamp_candidate_transaction(
                plan,
                repaired_artifact,
                directive,
                backend_evidence,
            )
        metadata = repaired_artifact.artifact_manifest.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["artifact_revision"] = repaired_artifact.revision
            metadata["parent_artifact_id"] = repaired_artifact.parent_artifact_id
            metadata["repair_history"] = list(repaired_artifact.repair_history)

        manifest_file = next(
            (
                generated
                for generated in repaired_artifact.files
                if generated.path in repaired_artifact.manifest_paths
            ),
            None,
        )
        if manifest_file is not None:
            manifest_file.content = json.dumps(
                repaired_artifact.artifact_manifest,
                indent=2,
                sort_keys=True,
            )
            if manifest_file.path not in changed_paths:
                changed_paths.append(manifest_file.path)

        repaired_digest = self._artifact_digest(repaired_artifact)

        return RepairResult(
            directive=directive,
            artifact=repaired_artifact,
            changed=True,
            changed_paths=sorted(changed_paths),
            previous_digest=previous_digest,
            repaired_digest=repaired_digest,
            backend_name=backend_name,
            backend_evidence=backend_evidence,
        )

    def _stamp_candidate_transaction(
        self,
        plan: FeasiblePlan,
        artifact: CodeArtifact,
        directive: RepairDirective,
        backend_evidence: Dict[str, object],
    ) -> None:
        transaction_paths = sorted(
            {
                str(path).replace("\\", "/")
                for path in backend_evidence.get("accepted_paths", [])
            }
        )
        required_capabilities = sorted(
            AdapterCapabilityContractChecker(self.registry).required_capabilities(plan)
        )
        files_by_path = {
            generated.path.replace("\\", "/"): generated
            for generated in artifact.files
        }
        transaction_token = f"candidate_transaction:{directive.repair_id}"
        compiler_token = "candidate_compiler:openai"
        for path in transaction_paths:
            generated = files_by_path.get(path)
            if generated is None:
                continue
            generated.generated_from_plan_sections = [
                token
                for token in generated.generated_from_plan_sections
                if not token.startswith("candidate_transaction:")
                and not token.startswith("candidate_capability:")
                and not token.startswith("candidate_compiler:")
            ]
            tokens = [
                compiler_token,
                transaction_token,
                *(f"candidate_capability:{capability}" for capability in required_capabilities),
            ]
            for token in tokens:
                if token not in generated.generated_from_plan_sections:
                    generated.generated_from_plan_sections.append(token)
            artifact.traceability[generated.path] = list(generated.generated_from_plan_sections)

        file_digests = {
            path: hashlib.sha256(files_by_path[path].content.encode("utf-8")).hexdigest()
            for path in transaction_paths
            if path in files_by_path
        }
        metadata = artifact.artifact_manifest.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.update(
                {
                    "generator": "forge_candidate_compiler",
                    "deterministic_templates": False,
                    "domain_adapter": "candidate",
                    "adapter_capabilities": required_capabilities,
                    "candidate_compilation": {
                        "backend": "openai",
                        "repair_id": directive.repair_id,
                        "transaction_paths": transaction_paths,
                        "file_sha256": file_digests,
                        "required_capabilities": required_capabilities,
                        "preflight_passed": bool(backend_evidence.get("preflight_passed", False)),
                    },
                }
            )
        artifact.artifact_manifest["traceability"] = copy.deepcopy(artifact.traceability)
        generated_entries = artifact.artifact_manifest.get("generated_files", [])
        if isinstance(generated_entries, list):
            for entry in generated_entries:
                if not isinstance(entry, dict):
                    continue
                generated = files_by_path.get(str(entry.get("path", "")).replace("\\", "/"))
                if generated is not None:
                    entry["generated_from"] = list(generated.generated_from_plan_sections)

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
            declared_entrypoint = plan.implementation_blueprint.entrypoint_path.replace("\\", "/")
            has_cli_file = any(
                path.path.replace("\\", "/") in {"src/cli.py", "src/main.py", declared_entrypoint}
                for path in plan.file_tree_plan
            )
            if not has_cli_file:
                raise MalformedPlanError(
                    "Plan declares cli_entrypoint interface but its blueprint entrypoint is not planned."
                )

    def _artifact_id(self, plan_id: str) -> str:
        digest = hashlib.sha256(plan_id.encode("utf-8")).hexdigest()[:12]
        return f"code-{digest}"

    def _artifact_digest(self, artifact: CodeArtifact) -> str:
        payload = {
            "artifact_id": artifact.artifact_id,
            "plan_id": artifact.plan_id,
            "revision": artifact.revision,
            "parent_artifact_id": artifact.parent_artifact_id,
            "files": [
                {
                    "path": generated.path,
                    "content": generated.content,
                    "kind": generated.kind,
                    "provenance": list(generated.generated_from_plan_sections),
                }
                for generated in sorted(artifact.files, key=lambda item: item.path)
            ],
            "test_paths": sorted(artifact.test_paths),
            "manifest_paths": sorted(artifact.manifest_paths),
            "runnable_entrypoints": sorted(artifact.runnable_entrypoints),
            "artifact_manifest": artifact.artifact_manifest,
            "traceability": artifact.traceability,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _changed_paths(previous: CodeArtifact, repaired: CodeArtifact) -> List[str]:
        previous_by_path = {generated.path: generated for generated in previous.files}
        repaired_by_path = {generated.path: generated for generated in repaired.files}
        changed: List[str] = []
        for path in sorted(set(previous_by_path) | set(repaired_by_path)):
            before = previous_by_path.get(path)
            after = repaired_by_path.get(path)
            if before is None or after is None:
                changed.append(path)
                continue
            if (
                before.content != after.content
                or before.kind != after.kind
                or before.generated_from_plan_sections != after.generated_from_plan_sections
            ):
                changed.append(path)
        if previous.artifact_manifest != repaired.artifact_manifest:
            for manifest_path in repaired.manifest_paths:
                if manifest_path not in changed:
                    changed.append(manifest_path)
        if previous.traceability != repaired.traceability:
            changed.extend(
                path
                for path in sorted(set(previous.traceability) | set(repaired.traceability))
                if path not in changed
            )
        return changed

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
        interface_refs = self._interfaces_for_path(
            path,
            plan.interfaces,
            entrypoint_path=plan.implementation_blueprint.entrypoint_path,
        )
        generated_from.extend(f"interface:{name}" for name in interface_refs)
        generated_from.extend(f"requirement:{requirement_id}" for requirement_id in plan_file.source_requirement_refs)
        capability = next(
            (
                item
                for item in plan.implementation_blueprint.capabilities
                if item.module_path.replace("\\", "/").lower() == path.replace("\\", "/").lower()
            ),
            None,
        )
        if capability is not None:
            generated_from.append(f"capability:{capability.capability_id}")
            generated_from.extend(
                f"capability_dependency:{dependency_id}"
                for dependency_id in capability.dependencies
            )
            generated_from.extend(
                f"quality_field:{field_name}"
                for field_name in capability.quality_fields
            )
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
            declared_module_path = (
                f"src/{interface.module_path.replace('.', '/')}.py"
                if interface.module_path
                else ""
            )
            matching_paths: List[str] = []
            for path, generated_file in sorted(generated_files.items()):
                if not path.startswith("src/") or not path.endswith(".py"):
                    continue
                if declared_module_path and path.replace("\\", "/") != declared_module_path:
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
        adapter_capabilities: List[str],
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
            "implementation_blueprint": asdict(plan.implementation_blueprint),
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
                "adapter_capabilities": adapter_capabilities,
            },
        }

    def _interfaces_for_path(
        self,
        path: str,
        interfaces: List[PlanInterface],
        entrypoint_path: str = "",
    ) -> List[str]:
        refs: List[str] = []
        lowered = path.replace("\\", "/").lower()
        normalized_entrypoint = entrypoint_path.replace("\\", "/").lower()
        for interface in interfaces:
            name = interface.name.lower()
            declared_module_path = (
                f"src/{interface.module_path.replace('.', '/')}.py".lower()
                if interface.module_path
                else ""
            )
            if declared_module_path and lowered == declared_module_path:
                refs.append(interface.name)
                continue
            if interface.interface_type == "cli_entrypoint" and lowered in {
                "src/cli.py",
                "src/main.py",
                normalized_entrypoint,
            }:
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
