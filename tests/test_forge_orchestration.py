from pathlib import Path

from core.forge.contracts import (
    AcceptanceContract,
    AcceptanceCriterion,
    ArtifactTargetType,
    BuildSpec,
    CodeArtifact,
    FeasiblePlan,
    ForgeRoute,
    GeneratedFile,
    InfeasibilityCertificate,
    ObligationContract,
    PackagedArtifact,
    PlanFile,
    PlanInterface,
    PlanTest,
    ValidationArtifact,
    ValidationStrategy,
)
from forge import (
    TERMINAL_INFEASIBLE_PROVEN,
    TERMINAL_VALIDATION_FAILED,
    TERMINAL_VERIFIED,
    render_cli_output,
    run_forge,
)


class _StubRequirementCompiler:
    def __init__(self, build_spec: BuildSpec):
        self.build_spec = build_spec

    def compile(self, requirement: str) -> BuildSpec:
        return self.build_spec


class _StubPlannerStage:
    def __init__(self, output):
        self.output = output
        self.called = 0

    def plan(self, build_spec: BuildSpec):
        self.called += 1
        return self.output


class _StubCoderStage:
    def __init__(self, code_artifact: CodeArtifact):
        self.code_artifact = code_artifact
        self.called = 0

    def generate(self, plan: FeasiblePlan) -> CodeArtifact:
        self.called += 1
        return self.code_artifact


class _StubValidatorStage:
    def __init__(self, validation: ValidationArtifact):
        self.validation = validation
        self.called = 0

    def validate(
        self,
        code_artifact: CodeArtifact,
        plan: FeasiblePlan,
        build_spec: BuildSpec,
    ) -> ValidationArtifact:
        self.called += 1
        return self.validation


class _StubPackagingStage:
    def __init__(self, packaged_artifact: PackagedArtifact):
        self.packaged_artifact = packaged_artifact
        self.called = 0

    def package(
        self,
        build_spec: BuildSpec,
        plan: FeasiblePlan,
        code_artifact: CodeArtifact,
        validation: ValidationArtifact,
    ) -> PackagedArtifact:
        self.called += 1
        return self.packaged_artifact


def _build_spec() -> BuildSpec:
    return BuildSpec(
        build_id="build-test",
        raw_requirement="build requirement",
        normalized_requirement="build requirement",
        functional_goals=["Implement CLI workflow."],
        non_functional_constraints=["Include tests."],
        acceptance_contract=AcceptanceContract(
            criteria=[AcceptanceCriterion(criterion_id="ac1", description="Read CSV input.")]
        ),
        obligation_contract=ObligationContract(
            mode="software_build",
            schema={"reads_csv": "bool"},
            required_fields=["reads_csv"],
        ),
        target_artifact_type=ArtifactTargetType.CLI,
    )


def _feasible_plan(build_spec: BuildSpec) -> FeasiblePlan:
    return FeasiblePlan(
        plan_id="plan-build-test",
        build_spec=build_spec,
        architecture_summary="CSV reader and expiration pipeline.",
        file_tree_plan=[PlanFile(path="src/cli.py", purpose="CLI entrypoint.")],
        interfaces=[PlanInterface(name="main", interface_type="cli_entrypoint")],
        required_tests=[PlanTest(test_name="test_reads_contracts_csv", objective="Read CSV input.")],
        required_obligations=["reads_csv"],
        acceptance_criterion_ids=["ac1"],
        obligation_mode="software_build",
        validation_strategy=ValidationStrategy(
            layer1_checks=["syntax/import/run"],
            layer2_checks=["obligation/acceptance/tests"],
            layer3_checks=["adversarial checks"],
        ),
        implementation_notes=[],
        packaging_target="python_cli_package",
    )


def _infeasibility_certificate(build_spec: BuildSpec) -> InfeasibilityCertificate:
    return InfeasibilityCertificate(
        certificate_id="infeasible-build-test",
        build_spec=build_spec,
        contradictions=["Constraint set is contradictory."],
        violated_obligations=["reads_csv"],
        proof_summary="Execution proved the constraint set infeasible.",
        terminal_status=TERMINAL_INFEASIBLE_PROVEN,
        minimal_relaxations=["Relax one contradictory constraint."],
        execution_evidence={"result_mode": "infeasible", "is_satisfiable": False},
    )


def _code_artifact() -> CodeArtifact:
    generated = GeneratedFile(
        path="src/cli.py",
        content="def main(argv=None):\n    return 0\n",
        kind="python_module",
        generated_from_plan_sections=["plan_file:src/cli.py", "interface:main"],
    )
    return CodeArtifact(
        artifact_id="code-test",
        plan_id="plan-build-test",
        files=[generated],
        test_paths=[],
        manifest_paths=["forge_artifact_manifest.json"],
        runnable_entrypoints=["src/cli.py"],
        artifact_manifest={},
        traceability={"src/cli.py": list(generated.generated_from_plan_sections)},
    )


def _passing_validation() -> ValidationArtifact:
    return ValidationArtifact(
        passed=True,
        failures=[],
        failure_signatures=[],
        evidence={
            "validated_entrypoints": {"src/cli.py": {"exists": True, "function_present": True, "executed": True}},
            "executed_tests": {"ran": True, "returncode": 0, "tests": []},
            "manifest_provenance_checks": {},
            "obligation_acceptance_checks": {},
        },
        metrics={"passed_layers": {"layer1": True, "layer2": True, "layer3": True}},
        failure_category=None,
    )


def _failing_validation() -> ValidationArtifact:
    return ValidationArtifact(
        passed=False,
        failures=["Required test execution failed."],
        failure_signatures=["test_execution_failure"],
        evidence={
            "validated_entrypoints": {"src/cli.py": {"exists": True, "function_present": True, "executed": True}},
            "executed_tests": {"ran": True, "returncode": 1, "tests": ["tests/test_reads_contracts_csv.py"]},
            "manifest_provenance_checks": {},
            "obligation_acceptance_checks": {"missing_required_tests": ["tests/test_reads_contracts_csv.py"]},
        },
        metrics={"passed_layers": {"layer1": True, "layer2": False, "layer3": True}},
        failure_category=None,
    )


def test_forge_orchestration_feasible_verified_path(tmp_path):
    build_spec = _build_spec()
    plan = _feasible_plan(build_spec)
    artifact = _code_artifact()
    validation = _passing_validation()
    packaged = PackagedArtifact(
        package_id="pkg-test",
        package_root=str((tmp_path / "package").resolve()),
        manifest_path=str((tmp_path / "package" / "forge_package_manifest.json").resolve()),
        packaged_files=["src/cli.py", "forge_package_manifest.json"],
        evidence_paths={"validation_evidence": "validation_evidence.json"},
        verification_metadata={"terminal_status": TERMINAL_VERIFIED},
    )
    planner = _StubPlannerStage(plan)
    coder = _StubCoderStage(artifact)
    validator = _StubValidatorStage(validation)
    packaging = _StubPackagingStage(packaged)

    result = run_forge(
        requirement="build requirement",
        output_root=str(tmp_path / "runs"),
        requirement_compiler=_StubRequirementCompiler(build_spec),
        planner_stage=planner,
        coder_stage=coder,
        validator_stage=validator,
        packaging_stage=packaging,
    )

    assert result.route == ForgeRoute.TERMINAL_VERIFIED
    assert result.terminal_status == TERMINAL_VERIFIED
    assert result.packaged_artifact is not None
    assert result.validation is not None and result.validation.passed is True
    assert result.artifact_path == packaged.package_root
    assert planner.called == 1
    assert coder.called == 1
    assert validator.called == 1
    assert packaging.called == 1
    assert Path(tmp_path / "runs").exists()
    assert "Status: verified" in render_cli_output(result)


def test_forge_orchestration_infeasible_proven_path(tmp_path):
    build_spec = _build_spec()
    certificate = _infeasibility_certificate(build_spec)
    planner = _StubPlannerStage(certificate)
    coder = _StubCoderStage(_code_artifact())
    validator = _StubValidatorStage(_passing_validation())
    packaging = _StubPackagingStage(
        PackagedArtifact(
            package_id="pkg-never",
            package_root="unused",
            manifest_path="unused",
            packaged_files=[],
            evidence_paths={},
            verification_metadata={},
        )
    )

    result = run_forge(
        requirement="contradictory requirement",
        output_root=str(tmp_path / "runs"),
        requirement_compiler=_StubRequirementCompiler(build_spec),
        planner_stage=planner,
        coder_stage=coder,
        validator_stage=validator,
        packaging_stage=packaging,
    )

    assert result.route == ForgeRoute.TERMINAL_INFEASIBLE
    assert result.terminal_status == TERMINAL_INFEASIBLE_PROVEN
    assert result.infeasibility_certificate is not None
    assert result.validation is None
    assert result.packaged_artifact is None
    assert planner.called == 1
    assert coder.called == 0
    assert validator.called == 0
    assert packaging.called == 0
    assert Path(result.artifact_path).exists()
    assert "Status: infeasible_proven" in render_cli_output(result)


def test_forge_orchestration_validation_failed_path(tmp_path):
    build_spec = _build_spec()
    plan = _feasible_plan(build_spec)
    artifact = _code_artifact()
    validation = _failing_validation()
    planner = _StubPlannerStage(plan)
    coder = _StubCoderStage(artifact)
    validator = _StubValidatorStage(validation)
    packaging = _StubPackagingStage(
        PackagedArtifact(
            package_id="pkg-never",
            package_root="unused",
            manifest_path="unused",
            packaged_files=[],
            evidence_paths={},
            verification_metadata={},
        )
    )

    result = run_forge(
        requirement="build requirement with failing validation",
        output_root=str(tmp_path / "runs"),
        requirement_compiler=_StubRequirementCompiler(build_spec),
        planner_stage=planner,
        coder_stage=coder,
        validator_stage=validator,
        packaging_stage=packaging,
    )

    assert result.route == ForgeRoute.TERMINAL_VALIDATION_FAILED
    assert result.terminal_status == TERMINAL_VALIDATION_FAILED
    assert result.validation is not None and result.validation.passed is False
    assert result.packaged_artifact is None
    assert planner.called == 1
    assert coder.called == 1
    assert validator.called == 1
    assert packaging.called == 0
    assert Path(result.artifact_path).exists()
    rendered = render_cli_output(result)
    assert "Status: validation_failed" in rendered
    assert "Validation failures:" in rendered


def test_packaging_not_called_when_validation_fails(tmp_path):
    build_spec = _build_spec()
    plan = _feasible_plan(build_spec)
    artifact = _code_artifact()
    validation = _failing_validation()
    packaging = _StubPackagingStage(
        PackagedArtifact(
            package_id="pkg-never",
            package_root="unused",
            manifest_path="unused",
            packaged_files=[],
            evidence_paths={},
            verification_metadata={},
        )
    )

    run_forge(
        requirement="build requirement with failing validation",
        output_root=str(tmp_path / "runs"),
        requirement_compiler=_StubRequirementCompiler(build_spec),
        planner_stage=_StubPlannerStage(plan),
        coder_stage=_StubCoderStage(artifact),
        validator_stage=_StubValidatorStage(validation),
        packaging_stage=packaging,
    )

    assert packaging.called == 0
