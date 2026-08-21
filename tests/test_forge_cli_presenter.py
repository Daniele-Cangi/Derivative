from io import BytesIO, StringIO, TextIOWrapper

from rich.console import Console

from core.forge.cli_presenter import print_cli_output, render_cli_output
from core.forge.contracts import (
    BuildSpec,
    ForgeResult,
    ForgeRoute,
    ForgeRunMetrics,
    InfeasibilityCertificate,
    PackagedArtifact,
    ValidationArtifact,
)


def _metrics() -> ForgeRunMetrics:
    return ForgeRunMetrics(
        planner_attempts=1,
        validation_attempts=2,
        repair_count=1,
        model_request_count=2,
        model_total_tokens=1200,
        estimated_model_cost_usd=0.0125,
        model_cost_pricing_source="environment",
    )


def _verified_result() -> ForgeResult:
    validation = ValidationArtifact(
        passed=True,
        metrics={
            "passed_layers": {
                "layer1": True,
                "layer2": True,
                "layer3": True,
            }
        },
    )
    package = PackagedArtifact(
        package_id="pkg-test",
        package_root="C:/artifacts/pkg-test",
        manifest_path="C:/artifacts/pkg-test/forge_package_manifest.json",
        verification_metadata={"code_artifact_digest": "abcdef0123456789fedcba"},
    )
    return ForgeResult(
        route=ForgeRoute.TERMINAL_VERIFIED,
        terminal_status="verified",
        summary="All evidence layers passed and packaging completed.",
        validation=validation,
        packaged_artifact=package,
        artifact_path=package.package_root,
        execution_time_seconds=4.25,
        run_metrics=_metrics(),
    )


def _failed_result() -> ForgeResult:
    validation = ValidationArtifact(
        passed=False,
        failures=["Generated tests did not exercise the target function."],
        failure_signatures=["non_semantic_test", "fake_acceptance_coverage"],
        metrics={
            "passed_layers": {
                "layer1": True,
                "layer2": False,
                "layer3": False,
            }
        },
    )
    return ForgeResult(
        route=ForgeRoute.TERMINAL_VALIDATION_FAILED,
        terminal_status="validation_failed",
        summary="Validation failed. Packaging was not attempted.",
        validation=validation,
        artifact_path="C:/artifacts/run-failed",
        execution_time_seconds=2.5,
        run_metrics=_metrics(),
    )


def _infeasible_result() -> ForgeResult:
    build_spec = BuildSpec(
        build_id="build-impossible",
        raw_requirement="contradictory requirement",
        normalized_requirement="contradictory requirement",
    )
    certificate = InfeasibilityCertificate(
        certificate_id="infeasible-build-impossible",
        build_spec=build_spec,
        contradictions=["diameter conflict", "edge-count conflict"],
        proof_summary="The constraints cannot hold simultaneously.",
    )
    return ForgeResult(
        route=ForgeRoute.TERMINAL_INFEASIBLE,
        terminal_status="infeasible_proven",
        summary="Planning terminated with an execution-grounded certificate.",
        infeasibility_certificate=certificate,
        artifact_path="C:/artifacts/infeasible-build-impossible",
        execution_time_seconds=1.75,
        run_metrics=ForgeRunMetrics(planner_attempts=1),
    )


def test_verified_presentation_exposes_the_complete_evidence_rail_and_code_seal():
    rendered = render_cli_output(_verified_result())

    assert "FORGE // DERIVATIVE" in rendered
    assert "01 / COMPILE" in rendered and "-> PASS" in rendered
    assert "04 / VALIDATE -> PASS" in rendered
    assert "05 / PACKAGE  -> SEALED" in rendered
    assert "Status: verified" in rendered
    assert "Trace seal: code:abcdef0123456789" in rendered
    assert "Attempts: planner 1 | validation 2 | repairs 1" in rendered
    assert "Model usage: 2 request(s) | 1200 token(s) | cost $0.012500" in rendered


def test_failed_presentation_blocks_packaging_and_retains_failure_signatures():
    rendered = render_cli_output(_failed_result())

    assert "04 / VALIDATE -> FAIL" in rendered
    assert "05 / PACKAGE  -> BLOCK" in rendered
    assert "Status: validation_failed" in rendered
    assert "Validation failures: non_semantic_test, fake_acceptance_coverage" in rendered
    assert "Packaged artifact:" not in rendered
    assert "Artifacts: C:/artifacts/run-failed" in rendered
    assert "Trace seal: failure:" in rendered


def test_infeasible_presentation_stops_before_generation_and_uses_certificate_seal():
    rendered = render_cli_output(_infeasible_result())

    assert "02 / PLAN     -> PROVEN / 2 contradiction(s) witnessed" in rendered
    assert "03 / GENERATE -> SKIP" in rendered
    assert "05 / PACKAGE  -> BLOCK" in rendered
    assert "Status: infeasible_proven" in rendered
    assert "Trace seal: certificate:infeasible-build-impossible" in rendered
    assert "Certificate artifacts: C:/artifacts/infeasible-build-impossible" in rendered


def test_rich_terminal_renderer_adds_color_without_changing_terminal_semantics():
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system="standard",
        width=100,
    )

    print_cli_output(_verified_result(), console=console)

    rendered = stream.getvalue()
    assert "\x1b[" in rendered
    assert "FORGE" in rendered
    assert "EVIDENCE RAIL" in rendered
    assert "Status:" in rendered and "verified" in rendered
    assert "code:abcdef0123456789" in rendered


def test_rich_terminal_renderer_is_safe_on_legacy_windows_encoding():
    buffer = BytesIO()
    stream = TextIOWrapper(buffer, encoding="cp1252")
    console = Console(
        file=stream,
        force_terminal=False,
        legacy_windows=True,
        width=80,
    )

    print_cli_output(_failed_result(), console=console)
    stream.flush()
    rendered = buffer.getvalue().decode("cp1252")
    stream.detach()

    assert "FORGE // DERIVATIVE" in rendered
    assert "Status: validation_failed" in rendered
    assert "PACKAGE" in rendered and "BLOCK" in rendered
