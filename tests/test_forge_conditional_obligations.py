from pathlib import Path

from core.forge.conditional_evidence import (
    ConditionalEvidenceValidator,
    analyze_observation_fidelity,
    analyze_test_expectations,
)
from core.forge.candidate_preflight import run_semantic_preflight
from core.forge.contracts import (
    ArtifactTargetType,
    CodeArtifact,
    FeasiblePlan,
    ForgeRoute,
    GeneratedFile,
    PlanFile,
    PlanInterface,
    RepairDirective,
    ValidationArtifact,
)
from core.forge.execution import LocalProcessExecutor
from core.forge.planner_stage import PlannerStage
from core.forge.repair_backend import SubstrateRepairBackend
from core.forge.repair_support import (
    behavioral_contract_seal,
    behavioral_generation_contracts,
    test_generation_contracts as build_test_generation_contracts,
)
from core.forge.requirement_compiler import RequirementCompiler


COMPOUND_REQUIREMENT = (
    "Define a Python CLI tool branch_tool that reads a filename from argv[1] and a chunk size "
    "from argv[2]. If the file is empty, or chunk size exceeds input length, output is empty "
    "with exit code 0. Edge cases cover empty files. No separator is inserted between chunks. "
    "Public import contract: from branch_tool import main."
)


def _planner_without_substrate() -> PlannerStage:
    return PlannerStage.__new__(PlannerStage)


def _plan_for(spec) -> FeasiblePlan:
    planner = _planner_without_substrate()
    tests = planner._derive_required_tests(spec)
    files = [
        PlanFile(
            path="src/branch_tool.py",
            purpose="Implement the declared CLI.",
            source_requirement_refs=[
                atom.requirement_id
                for atom in spec.requirement_atoms
                if atom.category != "coverage_directive"
            ],
        )
    ]
    interfaces = [
        PlanInterface(
            name="main",
            interface_type="cli_entrypoint",
            signature="main(argv: list[str]) -> int",
            module_path="branch_tool",
            explicit_argv_excludes_program_name=True,
            explicit_argv_count=2,
        )
    ]
    return FeasiblePlan(
        plan_id=f"plan-{spec.build_id}",
        build_spec=spec,
        architecture_summary="Python CLI implementing the compiled requirement contract.",
        file_tree_plan=files,
        interfaces=interfaces,
        required_tests=tests,
        acceptance_criterion_ids=[
            criterion.criterion_id for criterion in spec.acceptance_contract.criteria
        ],
        requirement_coverage=planner._build_requirement_coverage(spec, files, tests),
        conditional_obligation_coverage=planner._build_conditional_obligation_coverage(
            spec,
            tests,
        ),
        packaging_target="python_cli_package",
    )


def test_shared_consequent_conditions_compile_to_independent_branch_obligations():
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    parent = next(
        atom for atom in spec.requirement_atoms if atom.text.startswith("If the file")
    )
    obligations = [
        item
        for item in spec.conditional_obligations
        if item.parent_requirement_id == parent.requirement_id
    ]

    assert {item.trigger for item in obligations} == {
        "the file is empty",
        "chunk size exceeds input length",
    }
    assert len(obligations) == 4
    assert {item.observable_channel for item in obligations} == {"stdout", "exit_code"}
    assert all(item.source_fragment == parent.source_fragment for item in obligations)
    assert all(item.verification_method == "deterministic_probe" for item in obligations)


def test_explicit_negative_is_hard_and_never_compiled_as_ambiguity():
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    atom = next(atom for atom in spec.requirement_atoms if atom.text.startswith("No separator"))
    obligation = next(
        item
        for item in spec.conditional_obligations
        if item.parent_requirement_id == atom.requirement_id
    )

    assert atom.category == "negative_constraint"
    assert atom.strength == "hard"
    assert obligation.polarity == "negative"
    assert obligation.comparison_relation == "not_contains"
    assert obligation.observation_fidelity == "exact_text"


def test_reusable_predicate_taxonomy_partitions_read_and_decode_failures():
    spec = RequirementCompiler().compile(
        "Build a Python CLI. If file reading/decoding fails, the tool outputs exactly "
        "'invalid' to stderr and exits with code 1."
    )

    obligations = spec.conditional_obligations
    assert {item.trigger for item in obligations} == {
        "file reading fails",
        "file decoding fails",
    }
    assert {item.witness_class for item in obligations} == {
        "file_read_failure",
        "utf8_decode_failure",
    }
    assert len(obligations) == 4


def test_edge_cases_are_coverage_directives_referencing_existing_obligations():
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    edge_atom = next(
        atom for atom in spec.requirement_atoms if atom.category == "coverage_directive"
    )
    directive = next(
        item
        for item in spec.coverage_directives
        if item.parent_requirement_id == edge_atom.requirement_id
    )

    assert directive.witness_classes == ["empty_input"]
    assert directive.referenced_obligation_ids
    assert all(
        obligation_id.startswith("R")
        for obligation_id in directive.referenced_obligation_ids
    )
    assert all(
        edge_atom.requirement_id not in criterion.requirement_ids
        for criterion in spec.acceptance_contract.criteria
    )


def test_planner_and_generation_contracts_are_branch_aware():
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    plan = _plan_for(spec)
    branch_tests = [test for test in plan.required_tests if test.conditional_obligation_ids]
    artifact = CodeArtifact(
        artifact_id="artifact-branch",
        plan_id=plan.plan_id,
        files=[],
        test_paths=[f"tests/{test.test_name}.py" for test in branch_tests],
        traceability={},
    )
    contracts = build_test_generation_contracts(artifact.test_paths, plan, artifact)

    empty_test = next(test for test in branch_tests if "empty_input" in test.witness_classes)
    exceeds_test = next(
        test
        for test in branch_tests
        if "numeric_argument_exceeds_input_length" in test.witness_classes
    )
    assert empty_test.test_name != exceeds_test.test_name
    assert plan.conditional_obligation_coverage
    for path, contract in contracts.items():
        if path in artifact.test_paths:
            assert contract["conditional_obligations"]
            assert contract["observation_fidelities"]


def test_shared_generation_context_preserves_branch_and_exact_output_contracts():
    spec = RequirementCompiler().compile(
        COMPOUND_REQUIREMENT
        + " If the chunk size is invalid, the tool outputs exactly "
        + "'error: invalid input' to stderr and exits with code 1."
    )
    plan = _plan_for(spec)

    contracts = behavioral_generation_contracts(plan)
    context = SubstrateRepairBackend._repair_context(
        plan,
        ValidationArtifact(
            passed=False,
            failures=["Behavioral contract mismatch."],
            failure_signatures=["semantic_content_mismatch"],
        ),
        RepairDirective(
            repair_id="repair-behavioral-context",
            attempt=2,
            route=ForgeRoute.TO_CODER,
            failure_signatures=["semantic_content_mismatch"],
            target_paths=["src/branch_tool.py"],
            operations=["implement_missing_requirement_semantics"],
        ),
    )

    assert context["behavioral_contracts"] == contracts
    assert {
        item["trigger"] for item in contracts["conditional_obligations"]
    } >= {
        "the file is empty",
        "chunk size exceeds input length",
        "the chunk size is invalid",
    }
    exact = contracts["exact_output_contracts"]
    assert exact == [
        {
            "stream": "stderr",
            "expected": "error: invalid input",
            "source_fragment": "outputs exactly 'error: invalid input' to stderr",
            "precondition": "the chunk size is invalid",
            "observation_fidelity": "exact_text",
            "additional_output_allowed": False,
            "trailing_newline_included": False,
        }
    ]
    assert contracts["coverage_directives"][0]["referenced_obligation_ids"]


def test_behavioral_contract_seal_is_deterministic_and_semantically_sensitive():
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    plan = _plan_for(spec)

    seal = behavioral_contract_seal(plan)

    assert seal == behavioral_contract_seal(plan)
    assert seal["schema_version"] == 1
    assert seal["digest_mode"] == "canonical_json_utf8_v1"
    assert len(seal["sha256"]) == 64
    assert seal["build_id"] == spec.build_id
    assert seal["plan_id"] == plan.plan_id
    assert seal["contract_counts"]["conditional_obligations"] == len(
        spec.conditional_obligations
    )

    original = spec.conditional_obligations[0].expected_value
    spec.conditional_obligations[0].expected_value = "different observation"
    try:
        changed = behavioral_contract_seal(plan)
    finally:
        spec.conditional_obligations[0].expected_value = original

    assert changed["sha256"] != seal["sha256"]


def test_supplementary_test_contradiction_is_detected_without_traceability_mapping():
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    plan = _plan_for(spec)
    supplementary = '''from branch_tool import main


def test_size_exceeds_input_length(tmp_path, capsys):
    path = tmp_path / "input.txt"
    path.write_text("abc", encoding="utf-8")
    result = main([str(path), "4"])
    captured = capsys.readouterr()
    assert captured.out == "abc"
    assert result == 0
'''

    evidence = analyze_test_expectations(
        {"tests/test_supplementary.py": supplementary},
        spec,
        plan,
    )

    contradictions = [
        item
        for function in evidence
        for item in function["contradictions"]
    ]
    assert contradictions
    assert any(item["expected_value"] == "" for item in contradictions)
    assert any(item["asserted_value"] == "abc" for item in contradictions)


def test_lossy_transform_does_not_count_as_exact_output_evidence():
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    plan = _plan_for(spec)
    empty_test = next(
        test for test in plan.required_tests if "empty_input" in test.witness_classes
    )
    source = '''from branch_tool import main


def test_empty_file(tmp_path, capsys):
    path = tmp_path / "input.txt"
    path.write_text("", encoding="utf-8")
    result = main([str(path), "2"])
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    assert result == 0
'''
    path = f"tests/{empty_test.test_name}.py"

    evidence = analyze_observation_fidelity({path: source}, spec, plan)

    lossy = [
        item
        for function in evidence
        for item in function["lossy_observations"]
    ]
    assert lossy
    assert "strip" in lossy[0]["transformations"]


def test_candidate_preflight_rejects_conditional_expectation_contradiction():
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    plan = _plan_for(spec)
    supplementary = '''from branch_tool import main


def test_size_exceeds_input_length(tmp_path, capsys):
    path = tmp_path / "input.txt"
    path.write_text("abc", encoding="utf-8")
    result = main([str(path), "4"])
    captured = capsys.readouterr()
    assert captured.out == "abc"
    assert result == 0
'''

    result = run_semantic_preflight(
        {"tests/test_supplementary.py": supplementary},
        plan,
        {},
        {"phase": "tests", "passed": True, "failures": []},
    )

    failure = next(
        item
        for item in result["failures"]
        if item["kind"] == "conditional_test_expectation_contradiction"
    )
    assert result["passed"] is False
    assert result["test_failed_paths"] == ["tests/test_supplementary.py"]
    assert failure["contradictions"][0]["channel"] == "stdout"
    assert any(
        "Do not change the implementation" in requirement
        for requirement in result["correction_requirements"]
    )


def test_candidate_preflight_rejects_lossy_conditional_observation():
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    plan = _plan_for(spec)
    empty_test = next(
        test for test in plan.required_tests if "empty_input" in test.witness_classes
    )
    path = f"tests/{empty_test.test_name}.py"
    source = '''from branch_tool import main


def test_empty_file(tmp_path, capsys):
    input_path = tmp_path / "input.txt"
    input_path.write_text("", encoding="utf-8")
    result = main([str(input_path), "2"])
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    assert result == 0
'''

    result = run_semantic_preflight(
        {path: source},
        plan,
        {},
        {"phase": "tests", "passed": True, "failures": []},
    )

    failure = next(
        item
        for item in result["failures"]
        if item["kind"] == "conditional_observation_fidelity_failure"
    )
    assert result["passed"] is False
    assert result["test_failed_paths"] == [path]
    assert failure["lossy_observations"][0]["transformations"] == ["strip"]
    assert any(
        "without lossy normalization" in requirement
        for requirement in result["correction_requirements"]
    )


def test_candidate_preflight_preserves_consistent_exact_branch_evidence():
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    plan = _plan_for(spec)
    source = '''from branch_tool import main


def test_size_exceeds_input_length(tmp_path, capsys):
    input_path = tmp_path / "input.txt"
    input_path.write_text("abc", encoding="utf-8")
    result = main([str(input_path), "4"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert result == 0
'''

    result = run_semantic_preflight(
        {"tests/test_consistent_branch.py": source},
        plan,
        {},
        {"phase": "tests", "passed": True, "failures": []},
    )

    assert result["passed"] is True
    assert result["phase"] == "tests"


def test_interface_contract_is_structural_and_does_not_require_behavioral_test():
    spec = RequirementCompiler().compile(
        "Public import contract: from codec import encode."
    )
    planner = _planner_without_substrate()

    assert spec.requirement_atoms[0].verification_method == "interface_contract"
    assert planner._derive_required_tests(spec) == []


def test_uncompiled_hard_compound_condition_fails_closed():
    spec = RequirementCompiler().compile(
        "Build a Python library. If mode is alpha or policy is externally defined, "
        "the result must mirror a context-dependent external decision."
    )

    assert spec.conditional_normalization_issues
    assert spec.conditional_normalization_issues[0].hard
    assert any(
        "Materially unspecified conditional semantics" in flag
        for flag in spec.ambiguity_flags
    )
    plan = _plan_for(spec)
    validator = ConditionalEvidenceValidator(LocalProcessExecutor(), timeout_seconds=10)
    failures, signatures, _evidence = validator.validate(
        spec,
        plan,
        {},
        Path.cwd(),
    )
    assert failures
    assert "uncompiled_hard_conditional" in signatures


def test_validator_owned_probe_rejects_wrong_branch_behavior(tmp_path):
    spec = RequirementCompiler().compile(COMPOUND_REQUIREMENT)
    plan = _plan_for(spec)
    source_path = tmp_path / "src" / "branch_tool.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "def main(argv):\n"
        "    print('wrong')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    validator = ConditionalEvidenceValidator(LocalProcessExecutor(), timeout_seconds=10)

    failures, signatures, evidence = validator.validate(
        spec,
        plan,
        {"src/branch_tool.py": source_path},
        tmp_path,
    )

    assert failures
    assert "conditional_obligation_mismatch" in signatures
    assert any(
        item["status"] == "failed"
        for item in evidence["validator_branch_probes"]
    )
