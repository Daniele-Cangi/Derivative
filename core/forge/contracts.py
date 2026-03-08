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
class QualityContract:
    auth_level: str = "plaintext"
    secrets_in_plaintext: bool = True
    rate_limit_scope: str = "per_user"
    rate_limit_persistent: bool = False
    schema_versioned: bool = False
    audit_trail: bool = False
    health_endpoint: bool = False
    structured_logging: bool = False
    test_coverage_target: float = 0.6
    integration_tests: bool = False
    overall_level: int = 5

    def __post_init__(self) -> None:
        self.overall_level = self.compute_level()

    def compute_level(self) -> int:
        score = 0.0
        if self.auth_level in ("hashed", "jwt"):
            score += 2.0
        elif self.auth_level == "plaintext":
            score += 1.0
        if not self.secrets_in_plaintext:
            score += 1.0
        if self.rate_limit_scope == "distributed":
            score += 2.0
        elif self.rate_limit_scope == "per_user":
            score += 1.0
        if self.rate_limit_persistent:
            score += 1.0
        if self.schema_versioned:
            score += 1.0
        if self.audit_trail:
            score += 1.0
        if self.health_endpoint:
            score += 0.5
        if self.structured_logging:
            score += 0.5
        if self.integration_tests:
            score += 1.0
        return min(10, max(1, int(round(score))))


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
    quality_contract: QualityContract = field(default_factory=QualityContract)
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
    quality_contract: QualityContract = field(default_factory=QualityContract)
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
