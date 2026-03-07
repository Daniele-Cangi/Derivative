import json
import re
from typing import Any, Dict, List, Optional

from audit.trail import AuditTrail
from core.forge.contracts import (
    ArtifactTargetType,
    BuildSpec,
    FeasiblePlan,
    InfeasibilityCertificate,
    PlanFile,
    PlanInterface,
    PlannerStageOutput,
    PlanTest,
    ValidationStrategy,
)
from core.kernel import ReasoningKernel, ReasoningResult
from core.substrate import CognitiveSubstrate
from memory.delta import DeltaMemory
from memory.gene_pool import DesignGenePool


class PlannerStage:
    def __init__(
        self,
        execution_mode: str = "local-only",
        audit_log_file: str = "audit_trail.json",
        memory_file: str = "memory_deltas.json",
        gene_pool_file: str = "verified_gene_pool.json",
        substrate: Optional[CognitiveSubstrate] = None,
        kernel: Optional[ReasoningKernel] = None,
        audit_trail: Optional[AuditTrail] = None,
        memory: Optional[DeltaMemory] = None,
        gene_pool: Optional[DesignGenePool] = None,
    ):
        self.execution_mode = execution_mode
        self.substrate = substrate or CognitiveSubstrate(execution_mode=execution_mode)
        self.kernel = kernel or ReasoningKernel(execution_mode=execution_mode)
        self.audit_trail = audit_trail or AuditTrail(log_file=audit_log_file)
        self.memory = memory or DeltaMemory(storage_file=memory_file)
        self.gene_pool = gene_pool or DesignGenePool(storage_file=gene_pool_file)

    def plan(self, build_spec: BuildSpec) -> PlannerStageOutput:
        requirement = build_spec.normalized_requirement
        design_context = self.memory.retrieve_design_context(requirement)
        framings = self.substrate.decompose(requirement)
        reasoning_result = self.kernel.synthesize(
            requirement,
            framings,
            design_context=design_context,
            audit=self.audit_trail,
        )
        self._persist_learning(reasoning_result, requirement)

        execution_evidence = self._extract_execution_evidence(reasoning_result, design_context_count=len(design_context))
        if execution_evidence.get("result_mode") == "infeasible" and execution_evidence.get("is_satisfiable") is False:
            return self._to_infeasibility_certificate(build_spec, reasoning_result, execution_evidence)
        return self._to_feasible_plan(build_spec, reasoning_result, execution_evidence)

    def _persist_learning(self, reasoning_result: ReasoningResult, requirement: str) -> None:
        try:
            self.memory.record(reasoning_result, requirement)
        except OSError:
            pass
        try:
            self.gene_pool.record_execution(reasoning_result, requirement)
        except OSError:
            pass

    def _extract_execution_evidence(self, reasoning_result: ReasoningResult, design_context_count: int) -> Dict[str, Any]:
        execution_result = reasoning_result.execution_result
        payload = self._load_execution_payload(execution_result.final_output if execution_result else "")
        result_payload = payload.get("result", {}) if isinstance(payload, dict) else {}
        if not isinstance(result_payload, dict):
            result_payload = {}
        contradictions = result_payload.get("contradictions", [])
        if not isinstance(contradictions, list):
            contradictions = []

        return {
            "result_mode": str(result_payload.get("mode", "")),
            "is_satisfiable": result_payload.get("is_satisfiable"),
            "contradiction_count": int(result_payload.get("contradiction_count", len(contradictions)) or 0),
            "contradictions": contradictions,
            "execution_converged": bool(execution_result.converged) if execution_result else False,
            "cycles_used": int(execution_result.cycles_used) if execution_result else 0,
            "final_output": execution_result.final_output if execution_result else "",
            "final_prediction": execution_result.final_prediction if execution_result else "",
            "audit_log_path": self.audit_trail.log_file,
            "design_context_count": design_context_count,
            "terminal_status": (
                "infeasible_proven"
                if str(result_payload.get("mode", "")) == "infeasible" and result_payload.get("is_satisfiable") is False
                else "planning_result"
            ),
        }

    def _to_feasible_plan(
        self,
        build_spec: BuildSpec,
        reasoning_result: ReasoningResult,
        execution_evidence: Dict[str, Any],
    ) -> FeasiblePlan:
        architecture_summary = self._build_architecture_summary(build_spec)
        file_tree = self._derive_file_tree_plan(build_spec)
        interfaces = self._derive_interfaces(build_spec)
        required_tests = self._derive_required_tests(build_spec)
        required_obligations = (
            list(build_spec.obligation_contract.required_fields)
            if build_spec.obligation_contract is not None
            else []
        )
        strategy = ValidationStrategy(
            layer1_checks=[
                "Syntax/import checks for generated Python modules.",
                "Runnable entrypoint smoke execution.",
            ],
            layer2_checks=[
                "Acceptance contract criteria coverage.",
                (
                    f"Obligation contract mode: "
                    f"{build_spec.obligation_contract.mode if build_spec.obligation_contract else 'none'}."
                ),
                (
                    f"Required obligation fields: "
                    f"{', '.join(required_obligations) if required_obligations else 'none'}."
                ),
            ],
            layer3_checks=[
                "Adversarial validation pass over generated outputs.",
                "Regression checks against known contradictory/underspecified inputs.",
            ],
            stop_on_first_failure=True,
        )
        implementation_notes = [
            "Planner is execution-grounded: substrate decomposition + kernel synthesis + execution evidence.",
            f"Execution mode observed: {execution_evidence.get('result_mode', 'unknown')}.",
            f"Cycles used during grounding: {execution_evidence.get('cycles_used', 0)}.",
            f"Audit trace persisted at: {execution_evidence.get('audit_log_path', '')}.",
        ]
        acceptance_ids = [criterion.criterion_id for criterion in build_spec.acceptance_contract.criteria]
        obligation_mode = build_spec.obligation_contract.mode if build_spec.obligation_contract else "none"
        requirement_coverage = self._build_requirement_coverage(build_spec, file_tree, required_tests)

        return FeasiblePlan(
            plan_id=f"plan-{build_spec.build_id}",
            build_spec=build_spec,
            architecture_summary=architecture_summary,
            file_tree_plan=file_tree,
            interfaces=interfaces,
            required_tests=required_tests,
            required_obligations=required_obligations,
            acceptance_criterion_ids=acceptance_ids,
            requirement_coverage=requirement_coverage,
            obligation_mode=obligation_mode,
            validation_strategy=strategy,
            implementation_notes=implementation_notes,
            packaging_target=self._packaging_target(build_spec.target_artifact_type),
        )

    def _to_infeasibility_certificate(
        self,
        build_spec: BuildSpec,
        reasoning_result: ReasoningResult,
        execution_evidence: Dict[str, Any],
    ) -> InfeasibilityCertificate:
        contradictions = list(execution_evidence.get("contradictions", []))
        violated_obligations = (
            list(build_spec.obligation_contract.required_fields)
            if build_spec.obligation_contract is not None
            else []
        )
        proof_summary = (
            reasoning_result.execution_result.conclusion
            if reasoning_result.execution_result is not None
            else "Constraint execution produced an infeasibility signal."
        )
        minimal_relaxations = self._derive_minimal_relaxations(contradictions)

        return InfeasibilityCertificate(
            certificate_id=f"infeasible-{build_spec.build_id}",
            build_spec=build_spec,
            contradictions=contradictions,
            violated_obligations=violated_obligations,
            proof_summary=proof_summary,
            terminal_status="infeasible_proven",
            minimal_relaxations=minimal_relaxations,
            execution_evidence=execution_evidence,
        )

    def _build_architecture_summary(self, build_spec: BuildSpec) -> str:
        goals = " ".join(build_spec.functional_goals).lower()
        if build_spec.target_artifact_type == ArtifactTargetType.CLI:
            if "csv" in goals and "expiration" in goals:
                return (
                    "Python CLI with modular pipeline: CSV input loader, expiration-date extractor, "
                    "horizon-based contract flagger, and summary CSV writer."
                )
            return "Python CLI with command entrypoint, processing pipeline, and output writer modules."
        if build_spec.target_artifact_type == ArtifactTargetType.SERVICE:
            return "Python service architecture with API entrypoint, domain logic, and validation boundary."
        if build_spec.target_artifact_type == ArtifactTargetType.LIBRARY:
            return "Python library architecture with public API module, core workflow module, and tests."
        return "Python executable architecture with explicit entrypoint, workflow module, and tests."

    def _derive_file_tree_plan(self, build_spec: BuildSpec) -> List[PlanFile]:
        if build_spec.target_artifact_type == ArtifactTargetType.CLI:
            return [
                PlanFile(
                    path="src/cli.py",
                    purpose="CLI argument parsing and workflow dispatch.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/cli.py"),
                ),
                PlanFile(
                    path="src/contracts_csv.py",
                    purpose="CSV loading and normalization for contract records.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/contracts_csv.py"),
                ),
                PlanFile(
                    path="src/expiration_rules.py",
                    purpose="Expiration extraction and <N days flagging logic.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/expiration_rules.py"),
                ),
                PlanFile(
                    path="src/summary_writer.py",
                    purpose="Summary CSV generation and persistence.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/summary_writer.py"),
                ),
                PlanFile(
                    path="tests/test_cli_flow.py",
                    purpose="End-to-end CLI behavior tests.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_cli_flow.py"),
                ),
                PlanFile(
                    path="tests/test_expiration_rules.py",
                    purpose="Rule-level unit tests.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_expiration_rules.py"),
                ),
            ]
        if build_spec.target_artifact_type == ArtifactTargetType.SERVICE:
            return [
                PlanFile(
                    path="src/service.py",
                    purpose="Service/API entrypoint.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/service.py"),
                ),
                PlanFile(
                    path="src/domain.py",
                    purpose="Domain logic and constraints.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/domain.py"),
                ),
                PlanFile(
                    path="tests/test_service.py",
                    purpose="Service contract tests.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_service.py"),
                ),
            ]
        if build_spec.target_artifact_type == ArtifactTargetType.LIBRARY:
            return [
                PlanFile(
                    path="src/library/__init__.py",
                    purpose="Library public exports.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/library/__init__.py"),
                ),
                PlanFile(
                    path="src/library/core.py",
                    purpose="Core library logic.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/library/core.py"),
                ),
                PlanFile(
                    path="tests/test_library.py",
                    purpose="Library behavior tests.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_library.py"),
                ),
            ]
        return [
            PlanFile(
                path="src/main.py",
                purpose="Primary executable workflow.",
                source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/main.py"),
            ),
            PlanFile(
                path="tests/test_main.py",
                purpose="Baseline behavior tests.",
                source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_main.py"),
            ),
        ]

    def _derive_interfaces(self, build_spec: BuildSpec) -> List[PlanInterface]:
        if build_spec.target_artifact_type == ArtifactTargetType.CLI:
            return [
                PlanInterface(
                    name="main",
                    interface_type="cli_entrypoint",
                    signature="main(argv: Optional[list[str]] = None) -> int",
                    description="Runs the CLI workflow and returns process exit code.",
                ),
                PlanInterface(
                    name="load_contracts_csv",
                    interface_type="function",
                    signature="load_contracts_csv(path: str) -> list[dict[str, str]]",
                    description="Loads and normalizes contract rows from input CSV.",
                ),
                PlanInterface(
                    name="flag_expiring_contracts",
                    interface_type="function",
                    signature=(
                        "flag_expiring_contracts(records: list[dict[str, str]], horizon_days: int = 90) "
                        "-> list[dict[str, str]]"
                    ),
                    description="Flags contracts expiring within the configured horizon.",
                ),
                PlanInterface(
                    name="write_summary_csv",
                    interface_type="function",
                    signature="write_summary_csv(rows: list[dict[str, str]], output_path: str) -> None",
                    description="Writes summary CSV output with expiration flags.",
                ),
            ]
        return [
            PlanInterface(
                name="run",
                interface_type="entrypoint",
                signature="run() -> int",
                description="Runs the planned workflow and returns status code.",
            )
        ]

    def _derive_required_tests(self, build_spec: BuildSpec) -> List[PlanTest]:
        tests: List[PlanTest] = []
        obligation_fields = (
            list(build_spec.obligation_contract.required_fields)
            if build_spec.obligation_contract is not None
            else []
        )
        for index, criterion in enumerate(build_spec.acceptance_contract.criteria, start=1):
            name, test_type = self._semantic_test_spec(criterion.description, index)
            tests.append(
                PlanTest(
                    test_name=name,
                    objective=criterion.description,
                    test_type=test_type,
                    required=criterion.required,
                    acceptance_criterion_ids=[criterion.criterion_id],
                    obligation_fields=obligation_fields,
                    requirement_ids=list(criterion.requirement_ids),
                )
            )
        if not tests:
            tests.append(
                PlanTest(
                    test_name="test_smoke",
                    objective="Smoke-run the primary entrypoint.",
                    test_type="smoke",
                    required=True,
                    acceptance_criterion_ids=[],
                    obligation_fields=obligation_fields,
                    requirement_ids=[],
                )
            )
        return tests

    def _packaging_target(self, artifact_type: ArtifactTargetType) -> str:
        if artifact_type == ArtifactTargetType.CLI:
            return "python_cli_package"
        if artifact_type == ArtifactTargetType.SERVICE:
            return "python_service_package"
        if artifact_type == ArtifactTargetType.LIBRARY:
            return "python_library_package"
        return "python_package"

    def _derive_minimal_relaxations(self, contradictions: List[str]) -> List[str]:
        relaxations: List[str] = []
        for contradiction in contradictions:
            lowered = contradiction.lower()
            if "complete graph has diameter 1" in lowered:
                relaxations.append(
                    "Relax the diameter constraint to <= 1, or remove the all-pairs direct-connectivity requirement."
                )
            elif "needs" in lowered and "edge budget" in lowered:
                relaxations.append(
                    "Increase the edge budget to satisfy complete-connectivity edge requirements."
                )
        if not relaxations:
            relaxations.append("Relax at least one conflicting numeric bound and rerun planning.")
        return relaxations

    def _load_execution_payload(self, output: str) -> Dict[str, Any]:
        try:
            payload = json.loads(output) if output else {}
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _semantic_test_spec(self, objective: str, index: int) -> tuple[str, str]:
        lowered = objective.lower()
        if "malformed" in lowered and "invalid date" in lowered:
            return "test_handles_malformed_rows_and_invalid_dates", "validation"
        if "malformed" in lowered:
            return "test_handles_malformed_rows", "validation"
        if "invalid date" in lowered:
            return "test_rejects_invalid_dates", "validation"
        if "every possible date format" in lowered or "universal constraint" in lowered:
            return "test_universal_date_format_support", "proof"
        if "read" in lowered and "csv" in lowered:
            return "test_reads_contracts_csv", "integration"
        if "extract" in lowered and "expiration" in lowered:
            return "test_extracts_expiration_dates", "unit"
        if "flag" in lowered and ("90" in lowered or "less than" in lowered):
            return "test_flags_contracts_within_horizon", "unit"
        if "write" in lowered and "summary" in lowered and "totals" in lowered and "counts" in lowered:
            return "test_writes_summary_csv_with_totals_and_counts", "integration"
        if "write" in lowered and "summary" in lowered and "csv" in lowered:
            return "test_writes_summary_csv", "integration"
        if "test" in lowered:
            return "test_suite_executes", "quality"

        compact = "_".join(re.findall(r"[a-z0-9]+", lowered)[:6]) or f"acceptance_{index:02d}"
        return f"test_{compact}", "acceptance"

    def _requirement_ids_for_file(self, build_spec: BuildSpec, path: str) -> List[str]:
        lowered_path = path.lower()
        ids: List[str] = []
        for atom in build_spec.requirement_atoms:
            lowered_atom = atom.text.lower()
            if lowered_path in {"src/cli.py", "src/main.py"} and atom.category != "ambiguity":
                ids.append(atom.requirement_id)
                continue
            if lowered_path.endswith("src/contracts_csv.py"):
                if any(
                    token in lowered_atom
                    for token in ("csv", "invoice", "contract", "column", "customer", "due_date", "amount")
                ):
                    ids.append(atom.requirement_id)
                    continue
            if lowered_path.endswith("src/expiration_rules.py"):
                if any(
                    token in lowered_atom
                    for token in ("date", "expiration", "overdue", "invalid", "format", "flag")
                ):
                    ids.append(atom.requirement_id)
                    continue
            if lowered_path.endswith("src/summary_writer.py"):
                if any(token in lowered_atom for token in ("summary", "totals", "counts", "report", "write")):
                    ids.append(atom.requirement_id)
                    continue
            if lowered_path.startswith("tests/") and atom.category != "ambiguity":
                ids.append(atom.requirement_id)
        deduped: List[str] = []
        for item in ids:
            if item not in deduped:
                deduped.append(item)
        return deduped

    def _build_requirement_coverage(
        self,
        build_spec: BuildSpec,
        file_tree: List[PlanFile],
        required_tests: List[PlanTest],
    ) -> Dict[str, Dict[str, List[str]]]:
        coverage: Dict[str, Dict[str, List[str]]] = {
            atom.requirement_id: {"files": [], "tests": [], "acceptance_criteria": []}
            for atom in build_spec.requirement_atoms
        }
        for plan_file in file_tree:
            for req_id in plan_file.source_requirement_refs:
                if req_id in coverage and plan_file.path not in coverage[req_id]["files"]:
                    coverage[req_id]["files"].append(plan_file.path)
        for plan_test in required_tests:
            for req_id in plan_test.requirement_ids:
                if req_id in coverage and plan_test.test_name not in coverage[req_id]["tests"]:
                    coverage[req_id]["tests"].append(plan_test.test_name)
            for req_id in plan_test.requirement_ids:
                if req_id not in coverage:
                    continue
                for criterion_id in plan_test.acceptance_criterion_ids:
                    if criterion_id not in coverage[req_id]["acceptance_criteria"]:
                        coverage[req_id]["acceptance_criteria"].append(criterion_id)
        for criterion in build_spec.acceptance_contract.criteria:
            for req_id in criterion.requirement_ids:
                if req_id in coverage and criterion.criterion_id not in coverage[req_id]["acceptance_criteria"]:
                    coverage[req_id]["acceptance_criteria"].append(criterion.criterion_id)
        return coverage
