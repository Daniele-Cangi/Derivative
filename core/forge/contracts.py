from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ArtifactTargetType(str, Enum):
    CLI = "cli"
    LIBRARY = "library"
    SERVICE = "service"
    SCRIPT = "script"
    UNKNOWN = "unknown"


class FailureCategory(str, Enum):
    IMPLEMENTATION = "implementation"
    ARCHITECTURAL = "architectural"
    CONTRADICTION = "contradiction"
    UNDERSPECIFIED = "underspecified"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class ForgeRoute(str, Enum):
    TO_PLANNER = "to_planner"
    TO_CODER = "to_coder"
    TERMINAL_VERIFIED = "terminal_verified"
    TERMINAL_INFEASIBLE = "terminal_infeasible"
    TERMINAL_VALIDATION_FAILED = "terminal_validation_failed"


@dataclass
class RequirementAtom:
    requirement_id: str
    text: str
    category: str
    strength: str
    source_fragment: str


@dataclass
class AcceptanceCriterion:
    criterion_id: str
    description: str
    required: bool = True
    verification_hint: str = ""
    requirement_ids: List[str] = field(default_factory=list)


@dataclass
class AcceptanceContract:
    criteria: List[AcceptanceCriterion] = field(default_factory=list)
    pass_condition: str = "all_required"
    notes: List[str] = field(default_factory=list)


@dataclass
class ObligationContract:
    mode: str
    schema: Dict[str, str] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildSpec:
    build_id: str
    raw_requirement: str
    normalized_requirement: str
    functional_goals: List[str] = field(default_factory=list)
    non_functional_constraints: List[str] = field(default_factory=list)
    requirement_atoms: List[RequirementAtom] = field(default_factory=list)
    acceptance_contract: AcceptanceContract = field(default_factory=AcceptanceContract)
    obligation_contract: Optional[ObligationContract] = None
    target_artifact_type: ArtifactTargetType = ArtifactTargetType.UNKNOWN
    risk_hints: List[str] = field(default_factory=list)
    ambiguity_flags: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


@dataclass
class PlanFile:
    path: str
    purpose: str
    source_requirement_refs: List[str] = field(default_factory=list)


@dataclass
class PlanInterface:
    name: str
    interface_type: str
    signature: str = ""
    description: str = ""


@dataclass
class PlanTest:
    test_name: str
    objective: str
    test_type: str = "unit"
    required: bool = True
    acceptance_criterion_ids: List[str] = field(default_factory=list)
    obligation_fields: List[str] = field(default_factory=list)
    requirement_ids: List[str] = field(default_factory=list)


@dataclass
class ValidationStrategy:
    layer1_checks: List[str] = field(default_factory=list)
    layer2_checks: List[str] = field(default_factory=list)
    layer3_checks: List[str] = field(default_factory=list)
    stop_on_first_failure: bool = True


@dataclass
class FeasiblePlan:
    plan_id: str
    build_spec: BuildSpec
    architecture_summary: str
    file_tree_plan: List[PlanFile] = field(default_factory=list)
    interfaces: List[PlanInterface] = field(default_factory=list)
    required_tests: List[PlanTest] = field(default_factory=list)
    required_obligations: List[str] = field(default_factory=list)
    acceptance_criterion_ids: List[str] = field(default_factory=list)
    requirement_coverage: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    obligation_mode: str = "none"
    validation_strategy: ValidationStrategy = field(default_factory=ValidationStrategy)
    implementation_notes: List[str] = field(default_factory=list)
    packaging_target: str = ""


@dataclass
class InfeasibilityCertificate:
    certificate_id: str
    build_spec: BuildSpec
    contradictions: List[str] = field(default_factory=list)
    violated_obligations: List[str] = field(default_factory=list)
    proof_summary: str = ""
    terminal_status: str = "infeasible_proven"
    minimal_relaxations: List[str] = field(default_factory=list)
    execution_evidence: Dict[str, Any] = field(default_factory=dict)


PlannerStageOutput = Union[FeasiblePlan, InfeasibilityCertificate]


@dataclass
class GeneratedFile:
    path: str
    content: str
    kind: str
    generated_from_plan_sections: List[str] = field(default_factory=list)


@dataclass
class CodeArtifact:
    artifact_id: str
    plan_id: str
    files: List[GeneratedFile] = field(default_factory=list)
    test_paths: List[str] = field(default_factory=list)
    manifest_paths: List[str] = field(default_factory=list)
    runnable_entrypoints: List[str] = field(default_factory=list)
    artifact_manifest: Dict[str, Any] = field(default_factory=dict)
    traceability: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ValidationLayerResult:
    layer_name: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationArtifact:
    passed: bool
    failures: List[str] = field(default_factory=list)
    failure_signatures: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    layer1_result: Optional[ValidationLayerResult] = None
    layer2_result: Optional[ValidationLayerResult] = None
    layer3_result: Optional[ValidationLayerResult] = None
    failure_category: Optional[FailureCategory] = None
    next_route: Optional[ForgeRoute] = None


@dataclass
class PackagedArtifact:
    package_id: str
    package_root: str
    manifest_path: str
    packaged_files: List[str] = field(default_factory=list)
    evidence_paths: Dict[str, str] = field(default_factory=dict)
    verification_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForgeResult:
    route: ForgeRoute
    terminal_status: str
    summary: str
    validation: Optional[ValidationArtifact] = None
    packaged_artifact: Optional[PackagedArtifact] = None
    infeasibility_certificate: Optional[InfeasibilityCertificate] = None
    artifact_path: str = ""
    execution_time_seconds: float = 0.0
