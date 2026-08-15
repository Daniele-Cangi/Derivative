import json
import re
from typing import Any, Dict, List, Optional

from audit.trail import AuditTrail
from core.forge.contracts import (
    ArtifactTargetType,
    BuildSpec,
    CapabilitySpec,
    FeasiblePlan,
    ImplementationBlueprint,
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
        persistence_warnings = self._persist_learning(reasoning_result, requirement)

        execution_evidence = self._extract_execution_evidence(
            reasoning_result,
            design_context_count=len(design_context),
            persistence_warnings=persistence_warnings,
        )
        if execution_evidence.get("result_mode") == "infeasible" and execution_evidence.get("is_satisfiable") is False:
            return self._to_infeasibility_certificate(build_spec, reasoning_result, execution_evidence)
        return self._to_feasible_plan(build_spec, reasoning_result, execution_evidence)

    def _persist_learning(self, reasoning_result: ReasoningResult, requirement: str) -> List[str]:
        warnings: List[str] = []
        try:
            self.memory.record(reasoning_result, requirement)
        except OSError as exc:
            warnings.append(f"memory_persist_failed:{exc}")
        try:
            self.gene_pool.record_execution(reasoning_result, requirement)
        except OSError as exc:
            warnings.append(f"gene_pool_persist_failed:{exc}")
        return warnings

    def _extract_execution_evidence(
        self,
        reasoning_result: ReasoningResult,
        design_context_count: int,
        persistence_warnings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        execution_result = reasoning_result.execution_result
        payload = self._load_execution_payload(execution_result.final_output if execution_result else "")
        result_payload = payload.get("result", {}) if isinstance(payload, dict) else {}
        if not isinstance(result_payload, dict):
            result_payload = {}
        contradictions = result_payload.get("contradictions", [])
        if not isinstance(contradictions, list):
            contradictions = []
        execution_warnings = (
            list(getattr(execution_result, "warnings", []))
            if execution_result is not None
            else []
        )
        if not isinstance(execution_warnings, list):
            execution_warnings = []

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
            "execution_warnings": execution_warnings,
            "persistence_warnings": list(persistence_warnings or []),
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
        implementation_blueprint = self._derive_implementation_blueprint(build_spec)
        file_tree = self._derive_file_tree_plan(build_spec, implementation_blueprint)
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
        execution_warnings = execution_evidence.get("execution_warnings", [])
        if isinstance(execution_warnings, list):
            implementation_notes.extend(f"Execution warning: {warning}" for warning in execution_warnings)
        persistence_warnings = execution_evidence.get("persistence_warnings", [])
        if isinstance(persistence_warnings, list):
            implementation_notes.extend(f"Persistence warning: {warning}" for warning in persistence_warnings)
        acceptance_ids = [criterion.criterion_id for criterion in build_spec.acceptance_contract.criteria]
        obligation_mode = build_spec.obligation_contract.mode if build_spec.obligation_contract else "none"
        requirement_coverage = self._build_requirement_coverage(build_spec, file_tree, required_tests)

        return FeasiblePlan(
            plan_id=f"plan-{build_spec.build_id}",
            build_spec=build_spec,
            architecture_summary=architecture_summary,
            quality_contract=build_spec.quality_contract,
            implementation_blueprint=implementation_blueprint,
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
        if self._is_pipeline_build(build_spec):
            if self._is_sales_jsonl_pipeline(build_spec):
                return (
                    "Python JSON Lines sales pipeline with record validation, malformed-event quarantine, "
                    "per-customer aggregation, and summary JSON output."
                )
            if self._is_telemetry_jsonl_pipeline(build_spec):
                return (
                    "Python CLI data pipeline with JSON Lines telemetry ingestion, record validation, "
                    "JSONL quarantine output, per-device aggregation, and summary CSV generation."
                )
            if self._is_jsonl_pipeline(build_spec):
                return (
                    "Python JSON Lines data pipeline with the declared public interface, "
                    "requirement-defined record validation, deterministic output persistence, "
                    "and end-to-end behavioral tests."
                )
            return (
                "Python data pipeline with watched-directory ingestion, configurable row-schema validation, "
                "quarantine handling for invalid rows, SQLite persistence with audit trail, and REST health stats."
            )
        goals = " ".join(build_spec.functional_goals).lower()
        if build_spec.target_artifact_type == ArtifactTargetType.CLI:
            if self._is_json_log_cli(build_spec):
                return (
                    "Python CLI with JSON Lines application-log parsing, malformed-line accounting, "
                    "per-level aggregation, and JSON report output."
                )
            if self._is_recursive_json_merge_cli(build_spec):
                return (
                    "Python CLI with recursive JSON object merge, list replacement, root validation, "
                    "and JSON file output."
                )
            if self._is_csv_date_cli(build_spec):
                return (
                    "Python CLI with modular pipeline: CSV input loader, expiration-date extractor, "
                    "horizon-based contract flagger, and summary CSV writer."
                )
            return (
                "Python CLI with one declared command entrypoint, requirement-specific input processing, "
                "validation, output persistence, and behavioral tests."
            )
        if build_spec.target_artifact_type == ArtifactTargetType.SERVICE:
            if self._is_idempotent_event_service(build_spec):
                return (
                    "Python service module with API-key authentication, SQLite-backed idempotent event "
                    "creation, and integration tests for duplicate and unauthorized requests."
                )
            return (
                "Python service composed from separate API, domain, authentication, rate-limit, SQLite storage, "
                "audit, and observability capability modules."
            )
        if build_spec.target_artifact_type == ArtifactTargetType.LIBRARY:
            if build_spec.public_module:
                return (
                    f"Python library exposing the declared public module '{build_spec.public_module}' "
                    "with its callable contract and behavioral tests."
                )
            return "Python library architecture with public API module, core workflow module, and tests."
        return "Python executable architecture with explicit entrypoint, workflow module, and tests."

    def _derive_file_tree_plan(
        self,
        build_spec: BuildSpec,
        implementation_blueprint: ImplementationBlueprint | None = None,
    ) -> List[PlanFile]:
        if self._is_pipeline_build(build_spec):
            if self._is_sales_jsonl_pipeline(build_spec):
                return [
                    PlanFile(
                        path="src/pipeline.py",
                        purpose="Sales-event orchestration, per-customer aggregation, and summary JSON writing.",
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/pipeline.py"),
                    ),
                    PlanFile(
                        path="src/watcher.py",
                        purpose="JSON Lines input iteration with line-number and parse-error preservation.",
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/watcher.py"),
                    ),
                    PlanFile(
                        path="src/validator.py",
                        purpose="Sales-event customer_id and amount validation.",
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/validator.py"),
                    ),
                    PlanFile(
                        path="src/quarantine.py",
                        purpose="Malformed sales-event persistence to the requested quarantine JSONL file.",
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/quarantine.py"),
                    ),
                    PlanFile(
                        path="tests/test_pipeline.py",
                        purpose="End-to-end sales parsing, quarantine, aggregation, and summary JSON tests.",
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_pipeline.py"),
                    ),
                ]
            if self._is_telemetry_jsonl_pipeline(build_spec):
                return [
                    PlanFile(
                        path="src/pipeline.py",
                        purpose=(
                            "JSON Lines telemetry orchestration, per-device aggregation, "
                            "summary CSV writing, and CLI dispatch."
                        ),
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/pipeline.py"),
                    ),
                    PlanFile(
                        path="src/watcher.py",
                        purpose="JSON Lines input iteration with line-number preservation.",
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/watcher.py"),
                    ),
                    PlanFile(
                        path="src/validator.py",
                        purpose="Telemetry field, type, and timestamp validation.",
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/validator.py"),
                    ),
                    PlanFile(
                        path="src/quarantine.py",
                        purpose="Malformed telemetry persistence to the requested quarantine JSONL file.",
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/quarantine.py"),
                    ),
                    PlanFile(
                        path="tests/test_pipeline.py",
                        purpose="End-to-end JSONL parsing, quarantine, aggregation, summary, and CLI tests.",
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_pipeline.py"),
                    ),
                ]
            if self._is_jsonl_pipeline(build_spec):
                return self._derive_generic_jsonl_file_tree(build_spec)
            return [
                PlanFile(
                    path="src/pipeline.py",
                    purpose="Pipeline orchestration entrypoint and health stats.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/pipeline.py"),
                ),
                PlanFile(
                    path="src/watcher.py",
                    purpose="Watched-directory file discovery and polling behavior.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/watcher.py"),
                ),
                PlanFile(
                    path="src/validator.py",
                    purpose="Configurable schema validation for incoming rows.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/validator.py"),
                ),
                PlanFile(
                    path="src/quarantine.py",
                    purpose="Quarantine handler and structured JSON error logging.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/quarantine.py"),
                ),
                PlanFile(
                    path="tests/test_pipeline.py",
                    purpose="End-to-end pipeline behavior tests.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_pipeline.py"),
                ),
            ]
        if build_spec.target_artifact_type == ArtifactTargetType.CLI:
            if not self._is_csv_date_cli(build_spec):
                return [
                    PlanFile(
                        path="src/cli.py",
                        purpose=(
                            "CLI argument parsing and complete requirement-specific workflow implementation."
                        ),
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/cli.py"),
                    ),
                    PlanFile(
                        path="tests/test_cli_flow.py",
                        purpose="End-to-end behavioral verification of the declared CLI contract.",
                        source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_cli_flow.py"),
                    ),
                ]
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
            blueprint = implementation_blueprint or self._derive_implementation_blueprint(build_spec)
            planned_files = [
                PlanFile(
                    path=capability.module_path,
                    purpose=capability.purpose,
                    source_requirement_refs=list(capability.requirement_ids),
                )
                for capability in blueprint.capabilities
                if capability.enabled
            ]
            planned_files.append(
                PlanFile(
                    path="tests/test_service.py",
                    purpose="Service contract tests.",
                    source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_service.py"),
                )
            )
            return planned_files
        if build_spec.target_artifact_type == ArtifactTargetType.LIBRARY:
            if build_spec.public_module:
                module_path = f"src/{build_spec.public_module.replace('.', '/')}.py"
                return [
                    PlanFile(
                        path=module_path,
                        purpose=f"Declared public module '{build_spec.public_module}' and callable implementation.",
                        source_requirement_refs=self._library_requirement_ids(build_spec),
                    ),
                    PlanFile(
                        path=f"tests/test_{build_spec.public_module.replace('.', '_')}.py",
                        purpose="Behavioral verification of the declared public module contract.",
                        source_requirement_refs=self._requirement_ids_for_file(
                            build_spec,
                            f"tests/test_{build_spec.public_module.replace('.', '_')}.py",
                        ),
                    ),
                ]
            return [
                PlanFile(
                    path="src/library/__init__.py",
                    purpose="Library public exports.",
                    source_requirement_refs=self._library_requirement_ids(build_spec),
                ),
                PlanFile(
                    path="src/library/core.py",
                    purpose="Core library logic.",
                    source_requirement_refs=self._library_requirement_ids(build_spec),
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

    def _derive_generic_jsonl_file_tree(self, build_spec: BuildSpec) -> List[PlanFile]:
        files = [
            PlanFile(
                path="src/pipeline.py",
                purpose=(
                    "JSON Lines workflow orchestration through the public interface and "
                    "requirement-defined deterministic output writing."
                ),
                source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/pipeline.py"),
            ),
            PlanFile(
                path="src/watcher.py",
                purpose="JSON Lines input iteration with parse-error preservation.",
                source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/watcher.py"),
            ),
            PlanFile(
                path="src/validator.py",
                purpose="Requirement-defined event field and type validation.",
                source_requirement_refs=self._requirement_ids_for_file(build_spec, "src/validator.py"),
            ),
            PlanFile(
                path="tests/test_pipeline.py",
                purpose="End-to-end verification of parsing, validation, output, and return status.",
                source_requirement_refs=self._requirement_ids_for_file(build_spec, "tests/test_pipeline.py"),
            ),
        ]
        if "quarantine" in self._requirement_corpus(build_spec):
            files.insert(
                3,
                PlanFile(
                    path="src/quarantine.py",
                    purpose="Persistence of malformed events required by the input contract.",
                    source_requirement_refs=self._requirement_ids_for_file(
                        build_spec,
                        "src/quarantine.py",
                    ),
                ),
            )
        return files

    def _derive_interfaces(self, build_spec: BuildSpec) -> List[PlanInterface]:
        if self._is_pipeline_build(build_spec):
            declared_interfaces = self._library_interfaces(build_spec)
            if declared_interfaces:
                return [
                    PlanInterface(
                        name=interface.name,
                        interface_type=(
                            "entrypoint"
                            if interface.name == "run"
                            else interface.interface_type
                        ),
                        signature=interface.signature,
                        description=interface.description,
                    )
                    for interface in declared_interfaces
                ]
            if self._is_sales_jsonl_pipeline(build_spec):
                return [
                    PlanInterface(
                        name="run",
                        interface_type="function",
                        signature=(
                            "run(input_path: str, quarantine_path: str, "
                            "summary_json_path: str) -> int"
                        ),
                        description=(
                            "Processes sales JSON Lines, quarantines malformed events, writes the "
                            "per-customer summary, and returns a status code."
                        ),
                    )
                ]
            if self._requires_pipeline_cli(build_spec):
                return [
                    PlanInterface(
                        name="run",
                        interface_type="function",
                        signature=(
                            "run(input_path: str, quarantine_path: str, "
                            "summary_csv_path: str) -> int"
                        ),
                        description=(
                            "Processes one input file, writes rejected records and the summary, "
                            "and returns a status code."
                        ),
                    ),
                    PlanInterface(
                        name="main",
                        interface_type="cli_entrypoint",
                        signature="main(argv: Optional[list[str]] = None) -> int",
                        description="Parses CLI paths and delegates exactly once to run.",
                    ),
                ]
            return [
                PlanInterface(
                    name="run",
                    interface_type="entrypoint",
                    signature=(
                        "run(watch_dir: str, db_path: str, quarantine_dir: str, "
                        "poll_once: bool = True) -> int"
                    ),
                    description="Runs one polling cycle of the data pipeline and returns status code.",
                )
            ]
        if build_spec.target_artifact_type == ArtifactTargetType.CLI:
            interfaces = [
                PlanInterface(
                    name="main",
                    interface_type="cli_entrypoint",
                    signature="main(argv: Optional[list[str]] = None) -> int",
                    description="Runs the CLI workflow and returns process exit code.",
                )
            ]
            if not self._is_csv_date_cli(build_spec):
                return interfaces
            return [
                *interfaces,
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
        if build_spec.target_artifact_type == ArtifactTargetType.SERVICE:
            if self._is_idempotent_event_service(build_spec):
                return [
                    PlanInterface(
                        name="register_user",
                        interface_type="function",
                        signature="register_user(user_id: str, api_key: str, db_path: str) -> None",
                        description="Registers an API key in the configured SQLite database.",
                    ),
                    PlanInterface(
                        name="create_event",
                        interface_type="function",
                        signature=(
                            "create_event(api_key: str, event_id: str, payload: dict, "
                            "db_path: str) -> tuple[int, dict]"
                        ),
                        description=(
                            "Authenticates the key and creates an event exactly once for each event_id."
                        ),
                    ),
                ]
            return [
                PlanInterface(
                    name="run",
                    interface_type="entrypoint",
                    signature="run() -> int",
                    description="Initializes the composed service and returns a status code.",
                ),
                PlanInterface(
                    name="authenticate",
                    interface_type="function",
                    signature="authenticate(api_key: str, db_path: str) -> str | None",
                    description="Authenticates credentials under the selected auth capability.",
                ),
                PlanInterface(
                    name="enforce_rate_limit",
                    interface_type="function",
                    signature="enforce_rate_limit(user_id: str, limit: int, now: float, db_path: str) -> bool",
                    description="Applies the configured per-user or distributed rate-limit capability.",
                ),
                PlanInterface(
                    name="handle_request",
                    interface_type="function",
                    signature="handle_request(api_key: str, payload: dict, db_path: str, now: float) -> tuple[int, dict]",
                    description="Runs authentication, rate limiting, audit, and response behavior.",
                ),
            ]
        if build_spec.target_artifact_type == ArtifactTargetType.LIBRARY:
            interfaces = self._library_interfaces(build_spec)
            if interfaces:
                return interfaces
        return [
            PlanInterface(
                name="run",
                interface_type="entrypoint",
                signature="run() -> int",
                description="Runs the planned workflow and returns status code.",
            )
        ]

    def _derive_required_tests(self, build_spec: BuildSpec) -> List[PlanTest]:
        tests_by_name: Dict[str, PlanTest] = {}
        obligation_fields = (
            list(build_spec.obligation_contract.required_fields)
            if build_spec.obligation_contract is not None
            else []
        )
        for index, criterion in enumerate(build_spec.acceptance_contract.criteria, start=1):
            name, test_type = self._semantic_test_spec(criterion.description, index)
            existing = tests_by_name.get(name)
            if existing is None:
                tests_by_name[name] = PlanTest(
                    test_name=name,
                    objective=criterion.description,
                    test_type=test_type,
                    required=criterion.required,
                    acceptance_criterion_ids=[criterion.criterion_id],
                    obligation_fields=obligation_fields,
                    requirement_ids=list(criterion.requirement_ids),
                )
                continue
            if criterion.criterion_id not in existing.acceptance_criterion_ids:
                existing.acceptance_criterion_ids.append(criterion.criterion_id)
            for requirement_id in criterion.requirement_ids:
                if requirement_id not in existing.requirement_ids:
                    existing.requirement_ids.append(requirement_id)
            existing.objective = f"{existing.objective} {criterion.description}".strip()
        tests = list(tests_by_name.values())
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

    def _library_requirement_ids(self, build_spec: BuildSpec) -> List[str]:
        return [
            atom.requirement_id
            for atom in build_spec.requirement_atoms
            if atom.category != "ambiguity" and self._requires_library_source(atom.text)
        ]

    @staticmethod
    def _requires_library_source(text: str) -> bool:
        lowered = text.lower()
        return not bool(
            re.search(
                r"\bincludes?\s+(?:behavioral\s+|integration\s+|unit\s+)?tests?\b|\btests?\s+for\b",
                lowered,
            )
        )

    def _library_interfaces(self, build_spec: BuildSpec) -> List[PlanInterface]:
        declarations = build_spec.normalized_requirement
        signatures = re.findall(
            r"\b([a-zA-Z_]\w*)\s*\(([^()]*)\)\s*->\s*([a-zA-Z_]\w*(?:\[[^\]]+\])?)",
            declarations,
        )
        interfaces: List[PlanInterface] = []
        for name, arguments, return_type in signatures:
            signature = f"{name}({arguments.strip()}) -> {return_type.strip()}"
            interfaces.append(
                PlanInterface(
                    name=name,
                    interface_type="function",
                    signature=signature,
                    description=f"Public library API declared by the requirement: {signature}.",
                    module_path=build_spec.public_module,
                )
            )
        return interfaces

    def _packaging_target(self, artifact_type: ArtifactTargetType) -> str:
        if artifact_type == ArtifactTargetType.CLI:
            return "python_cli_package"
        if artifact_type == ArtifactTargetType.PIPELINE:
            return "python_pipeline_package"
        if artifact_type == ArtifactTargetType.SERVICE:
            return "python_service_package"
        if artifact_type == ArtifactTargetType.LIBRARY:
            return "python_library_package"
        return "python_package"

    def _derive_implementation_blueprint(self, build_spec: BuildSpec) -> ImplementationBlueprint:
        if self._is_pipeline_build(build_spec):
            return ImplementationBlueprint(
                target_artifact_type=build_spec.target_artifact_type,
                entrypoint_path="src/pipeline.py",
            )
        if build_spec.target_artifact_type != ArtifactTargetType.SERVICE:
            entrypoint = "src/cli.py" if build_spec.target_artifact_type == ArtifactTargetType.CLI else ""
            return ImplementationBlueprint(
                target_artifact_type=build_spec.target_artifact_type,
                entrypoint_path=entrypoint,
            )

        if self._is_idempotent_event_service(build_spec):
            requirement_ids = [
                atom.requirement_id
                for atom in build_spec.requirement_atoms
                if atom.category != "ambiguity"
            ]
            return ImplementationBlueprint(
                target_artifact_type=ArtifactTargetType.SERVICE,
                entrypoint_path="src/service.py",
                capabilities=[
                    CapabilitySpec(
                        capability_id="cap_service_api",
                        capability_type="service_api",
                        module_path="src/service.py",
                        purpose="Public register_user and idempotent create_event service API.",
                        interfaces=["register_user", "create_event"],
                        dependencies=["cap_auth", "cap_storage"],
                        requirement_ids=requirement_ids,
                    ),
                    CapabilitySpec(
                        capability_id="cap_storage",
                        capability_type="sqlite_storage",
                        module_path="src/storage.py",
                        purpose="SQLite users and unique event storage with atomic insert-or-read behavior.",
                        interfaces=["init_db", "insert_event_once"],
                        requirement_ids=requirement_ids,
                    ),
                    CapabilitySpec(
                        capability_id="cap_auth",
                        capability_type="authentication",
                        module_path="src/auth.py",
                        purpose="API-key registration and lookup backed by SQLite.",
                        interfaces=["register_user", "authenticate"],
                        dependencies=["cap_storage"],
                        requirement_ids=requirement_ids,
                    ),
                ],
            )

        quality = build_spec.quality_contract
        definitions = [
            (
                "cap_service_api",
                "service_api",
                "src/service.py",
                "Thin API entrypoint and public service exports.",
                ["run", "create_app"],
                ["cap_domain", "cap_observability", "cap_storage"],
                ["health_endpoint", "overall_level"],
                {"health_endpoint": quality.health_endpoint, "overall_level": quality.overall_level},
            ),
            (
                "cap_domain",
                "request_workflow",
                "src/domain.py",
                "Request workflow composed from auth, rate-limit, and audit capabilities.",
                ["handle_request"],
                ["cap_auth", "cap_rate_limit", "cap_audit"],
                [],
                {},
            ),
            (
                "cap_storage",
                "sqlite_storage",
                "src/storage.py",
                "SQLite schema, connections, and schema-version metadata.",
                ["init_db"],
                [],
                ["secrets_in_plaintext", "rate_limit_persistent", "schema_versioned", "audit_trail"],
                {
                    "secrets_in_plaintext": quality.secrets_in_plaintext,
                    "rate_limit_persistent": quality.rate_limit_persistent,
                    "schema_versioned": quality.schema_versioned,
                    "audit_trail": quality.audit_trail,
                },
            ),
            (
                "cap_auth",
                "authentication",
                "src/auth.py",
                "Credential registration and authentication policy.",
                ["register_user", "authenticate"],
                ["cap_storage"],
                ["auth_level", "secrets_in_plaintext"],
                {"auth_level": quality.auth_level, "secrets_in_plaintext": quality.secrets_in_plaintext},
            ),
            (
                "cap_rate_limit",
                "rate_limit",
                "src/rate_limit.py",
                "Per-user, persistent, or distributed rate-limit policy.",
                ["enforce_rate_limit"],
                ["cap_storage"],
                ["rate_limit_scope", "rate_limit_persistent"],
                {
                    "scope": quality.rate_limit_scope,
                    "persistent": quality.rate_limit_persistent,
                },
            ),
            (
                "cap_audit",
                "audit_trail",
                "src/audit.py",
                "Request audit persistence and structured event logging.",
                ["record_event", "get_recent_events"],
                ["cap_storage"],
                ["audit_trail", "structured_logging"],
                {"enabled": quality.audit_trail, "structured_logging": quality.structured_logging},
            ),
            (
                "cap_observability",
                "observability",
                "src/observability.py",
                "Health statistics and structured operational logging.",
                ["health_status"],
                ["cap_storage"],
                ["health_endpoint", "structured_logging"],
                {"health_endpoint": quality.health_endpoint, "structured_logging": quality.structured_logging},
            ),
        ]
        capabilities = [
            CapabilitySpec(
                capability_id=capability_id,
                capability_type=capability_type,
                module_path=module_path,
                purpose=purpose,
                interfaces=interfaces,
                dependencies=dependencies,
                requirement_ids=self._capability_requirement_ids(build_spec, capability_type),
                quality_fields=quality_fields,
                config=config,
            )
            for (
                capability_id,
                capability_type,
                module_path,
                purpose,
                interfaces,
                dependencies,
                quality_fields,
                config,
            ) in definitions
        ]
        return ImplementationBlueprint(
            target_artifact_type=ArtifactTargetType.SERVICE,
            entrypoint_path="src/service.py",
            capabilities=capabilities,
        )

    def _capability_requirement_ids(self, build_spec: BuildSpec, capability_type: str) -> List[str]:
        token_map = {
            "service_api": ("service", "api", "rest", "endpoint", "health"),
            "request_workflow": ("request", "service", "api"),
            "sqlite_storage": (
                "persist",
                "persistent",
                "storage",
                "sqlite",
                "database",
                "schema",
                "restart",
                "restarts",
                "survives restarts",
            ),
            "authentication": ("auth", "api key", "bcrypt", "jwt", "oauth", "credential"),
            "rate_limit": ("rate limit", "per-user", "per user", "redis"),
            "audit_trail": ("audit", "event log", "logging", "requests"),
            "observability": ("health", "monitor", "observability", "logging"),
        }
        tokens = token_map.get(capability_type, ())
        requirement_ids = [
            atom.requirement_id
            for atom in build_spec.requirement_atoms
            if atom.category != "ambiguity"
            and any(self._matches_capability_token(atom.text, token) for token in tokens)
        ]
        if capability_type == "service_api" and not requirement_ids:
            requirement_ids = [
                atom.requirement_id for atom in build_spec.requirement_atoms if atom.category != "ambiguity"
            ]
        return requirement_ids

    @staticmethod
    def _matches_capability_token(text: str, token: str) -> bool:
        return bool(
            re.search(
                rf"(?<![a-z0-9_]){re.escape(token.lower())}(?![a-z0-9_])",
                text.lower(),
            )
        )

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
            elif "capacity exactly zero" in lowered:
                relaxations.append(
                    "Increase queue capacity to at least one, or remove the requirement to retain an item."
                )
            elif "zero-byte file" in lowered:
                relaxations.append(
                    "Allow non-zero output length, or remove the requirement that the file contain a JSON object."
                )
            elif "pigeonhole principle" in lowered:
                relaxations.append(
                    "Increase the encoded output capacity to at least the input capacity, restrict the input domain, "
                    "or permit explicit external metadata."
                )
            elif "erase_all" in lowered and "permanent retention" in lowered:
                relaxations.append(
                    "Remove permanent same-log retrieval after erase_all, or permit a separate retained archive."
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
        if "merge" in lowered and "json" in lowered and "recurs" in lowered:
            return "test_recursive_json_merge", "integration"
        if "replac" in lowered and "list" in lowered:
            return "test_replaces_json_lists", "unit"
        if "non-object root" in lowered or "non object root" in lowered:
            return "test_rejects_non_object_json_root", "validation"
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
            if lowered_path.endswith("src/pipeline.py"):
                if atom.category != "ambiguity":
                    ids.append(atom.requirement_id)
                    continue
            if lowered_path.endswith("src/watcher.py"):
                input_terms = {"input_jsonl", "input_csv"} & set(atom.evidence_terms)
                parser_terms = {"malformed_records"} & set(atom.evidence_terms)
                if input_terms or parser_terms or any(
                    token in lowered_atom for token in ("watch", "directory", "poll", "discover")
                ):
                    ids.append(atom.requirement_id)
                    continue
            if lowered_path.endswith("src/validator.py"):
                if any(token in lowered_atom for token in ("validate", "schema", "row", "invalid", "malformed")):
                    ids.append(atom.requirement_id)
                    continue
            if lowered_path.endswith("src/quarantine.py"):
                if any(token in lowered_atom for token in ("quarantine", "reject", "error", "invalid")):
                    ids.append(atom.requirement_id)
                    continue
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

    def _is_pipeline_build(self, build_spec: BuildSpec) -> bool:
        if build_spec.target_artifact_type == ArtifactTargetType.LIBRARY:
            return False
        combined = " ".join(
            [
                build_spec.normalized_requirement.lower(),
                " ".join(goal.lower() for goal in build_spec.functional_goals),
                " ".join(atom.text.lower() for atom in build_spec.requirement_atoms),
            ]
        )
        pipeline_tokens = (
            "pipeline",
            "data pipeline",
            "watched directory",
            "validate each row",
            "validates each row",
            "schema validation",
            "quarantine",
        )
        return any(token in combined for token in pipeline_tokens)

    def _is_csv_date_cli(self, build_spec: BuildSpec) -> bool:
        if build_spec.target_artifact_type != ArtifactTargetType.CLI:
            return False
        combined = " ".join(
            [
                build_spec.normalized_requirement.lower(),
                " ".join(goal.lower() for goal in build_spec.functional_goals),
                " ".join(atom.text.lower() for atom in build_spec.requirement_atoms),
            ]
        )
        date_domain_tokens = (
            "expiration",
            "expiring",
            "due_date",
            "due date",
            "overdue",
        )
        return "csv" in combined and any(token in combined for token in date_domain_tokens)

    def _is_json_log_cli(self, build_spec: BuildSpec) -> bool:
        if build_spec.target_artifact_type != ArtifactTargetType.CLI:
            return False
        combined = self._requirement_corpus(build_spec)
        return all(
            token in combined
            for token in ("json lines", "level", "message", "total_valid", "counts_by_level")
        )

    def _is_recursive_json_merge_cli(self, build_spec: BuildSpec) -> bool:
        if build_spec.target_artifact_type != ArtifactTargetType.CLI:
            return False
        combined = self._requirement_corpus(build_spec)
        return all(
            token in combined
            for token in ("json", "merge", "recurs", "replaces lists", "non-object root")
        )

    def _requires_pipeline_cli(self, build_spec: BuildSpec) -> bool:
        if build_spec.target_artifact_type == ArtifactTargetType.CLI:
            return True
        return any(
            "cli_entrypoint" in atom.evidence_terms
            for atom in build_spec.requirement_atoms
        )

    def _is_jsonl_pipeline(self, build_spec: BuildSpec) -> bool:
        return any(
            bool({"input_jsonl", "jsonl"} & set(atom.evidence_terms))
            for atom in build_spec.requirement_atoms
        )

    def _is_telemetry_jsonl_pipeline(self, build_spec: BuildSpec) -> bool:
        if not self._is_jsonl_pipeline(build_spec):
            return False
        combined = self._requirement_corpus(build_spec)
        return all(
            token in combined
            for token in ("device_id", "timestamp", "temperature_c")
        )

    def _is_sales_jsonl_pipeline(self, build_spec: BuildSpec) -> bool:
        if not self._is_jsonl_pipeline(build_spec):
            return False
        combined = self._requirement_corpus(build_spec)
        return all(
            token in combined
            for token in ("customer_id", "amount", "transaction_count", "total_amount")
        ) and "summary json" in combined

    def _is_idempotent_event_service(self, build_spec: BuildSpec) -> bool:
        if build_spec.target_artifact_type != ArtifactTargetType.SERVICE:
            return False
        combined = self._requirement_corpus(build_spec)
        return all(token in combined for token in ("create_event", "event_id", "idempotent", "sqlite"))

    @staticmethod
    def _requirement_corpus(build_spec: BuildSpec) -> str:
        return " ".join(
            [
                build_spec.normalized_requirement.lower(),
                *(goal.lower() for goal in build_spec.functional_goals),
                *(atom.text.lower() for atom in build_spec.requirement_atoms),
            ]
        )

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
