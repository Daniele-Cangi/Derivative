"""Quality recognizers must not turn literal values or rejections into demands."""
from dataclasses import asdict

import pytest

from core.forge.contracts import CodeArtifact
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.requirement_text import explicit_test_coverage_target
from core.forge.validation.quality import QualityContractChecker


FEATURES = [
    ("JWT", {"auth_level": "jwt", "secrets_in_plaintext": False}),
    ("JWTs", {"auth_level": "jwt", "secrets_in_plaintext": False}),
    ("bearer token", {"auth_level": "jwt", "secrets_in_plaintext": False}),
    ("bearer tokens", {"auth_level": "jwt", "secrets_in_plaintext": False}),
    ("OAuth", {"auth_level": "jwt", "secrets_in_plaintext": False}),
    ("OAuth2", {"auth_level": "jwt", "secrets_in_plaintext": False}),
    ("bcrypt", {"auth_level": "hashed", "secrets_in_plaintext": False}),
    ("Argon2", {"auth_level": "hashed", "secrets_in_plaintext": False}),
    ("hashed API keys", {"auth_level": "hashed", "secrets_in_plaintext": False}),
    ("migrations", {"schema_versioned": True}),
    ("versioned schema", {"schema_versioned": True}),
    ("versioned schemas", {"schema_versioned": True}),
    ("Alembic", {"schema_versioned": True}),
    ("audit trail", {"audit_trail": True}),
    ("audit trails", {"audit_trail": True}),
    ("audit log", {"audit_trail": True}),
    ("audit logs", {"audit_trail": True}),
    ("event log", {"audit_trail": True}),
    ("event logs", {"audit_trail": True}),
    ("health check", {"health_endpoint": True, "structured_logging": True}),
    ("health checks", {"health_endpoint": True, "structured_logging": True}),
    ("monitoring", {"health_endpoint": True, "structured_logging": True}),
    ("observability", {"health_endpoint": True, "structured_logging": True}),
    ("structured JSON logging", {"health_endpoint": True, "structured_logging": True}),
    ("structured error logging", {"health_endpoint": True, "structured_logging": True}),
    ("integration tests", {"integration_tests": True}),
    ("end-to-end", {"integration_tests": True}),
    ("e2e", {"integration_tests": True}),
    ("production-grade", {
        "schema_versioned": True, "audit_trail": True, "health_endpoint": True,
        "integration_tests": True, "test_coverage_target": 0.8,
    }),
]


@pytest.mark.parametrize("feature,expected", FEATURES)
@pytest.mark.parametrize("form", ["positive", "negative", "literal"])
def test_quality_features_respect_literal_and_negative_scope(feature, expected, form):
    compiler = RequirementCompiler()
    base = "Build a Python CLI."
    clause = {
        "positive": f"Require {feature}.",
        "negative": f"Do not use {feature}.",
        "literal": f'Print the exact value "{feature}".',
    }[form]
    spec = compiler.compile(base + " " + clause)
    quality = asdict(spec.quality_contract)
    if form == "positive":
        for field, value in expected.items():
            assert quality[field] == value
    else:
        assert quality == asdict(compiler.compile(base).quality_contract)

    # Filtering quality triggers must not discard the original acceptance atom.
    assert clause in spec.normalized_requirement
    matching = {
        atom.requirement_id for atom in spec.requirement_atoms
        if feature.lower() in atom.source_fragment.lower()
    }
    assert matching
    linked = {rid for criterion in spec.acceptance_contract.criteria for rid in criterion.requirement_ids}
    assert matching <= linked


@pytest.mark.parametrize("clause", [
    "Don't require integration tests.",
    "Integration tests aren't required.",
    "Integration tests should not be required.",
    "Integration tests are optional.",
    "Never add integration tests.",
    "Without integration tests.",
    "No integration tests.",
    "Neither an audit trail nor integration tests.",
    "Do not add an audit trail or integration tests.",
    "Not integration tests, only unit tests.",
])
def test_negative_quality_forms_do_not_create_positive_obligations(clause):
    quality = RequirementCompiler().compile("Build a Python CLI. " + clause).quality_contract
    assert quality.integration_tests is False
    assert quality.audit_trail is False


@pytest.mark.parametrize("clause", [
    "Do not mutate input and require integration tests.",
    "Do not mutate input, include integration tests.",
    "No audit trail is required, integration tests are required.",
    "Do not add an audit trail and the application requires integration tests.",
    "Do not add an audit trail but require integration tests.",
    "No audit trail; use integration tests.",
    "Not only integration tests but also unit tests.",
    "Not just integration tests but also unit tests.",
    "Integration tests must not be ignored.",
    "Integration tests are required, without an audit trail.",
])
def test_positive_quality_survives_unrelated_negation(clause):
    quality = RequirementCompiler().compile("Build a Python CLI. " + clause).quality_contract
    assert quality.integration_tests is True
    assert quality.audit_trail is False


@pytest.mark.parametrize("clause", [
    "Do not mutate input and require audit logs, migrations and integration tests.",
    "Do not mutate input, add audit logs and migrations and integration tests.",
])
def test_a_new_positive_directive_applies_to_its_coordinated_feature_list(clause):
    quality = RequirementCompiler().compile("Build a Python CLI. " + clause).quality_contract
    assert quality.audit_trail is True
    assert quality.schema_versioned is True
    assert quality.integration_tests is True


@pytest.mark.parametrize("clause,level", [
    ("Use API key authentication; do not use JWT.", "plaintext"),
    ("Do not use JWT; use bcrypt for API keys.", "hashed"),
    ('Use API key authentication and print "JWT".', "plaintext"),
    ("JWT authentication is not required.", "plaintext"),
    ("Do not mutate input and enable JWT authentication.", "jwt"),
    ("Do not store raw API keys, and authentication must use JWT.", "jwt"),
    ("Do not use JWT, and authentication must not use OAuth2.", "plaintext"),
    ("Do not use bcrypt but use JWT authentication.", "jwt"),
    ("Use secure API key authentication.", "hashed"),
    ('Use API key authentication and print "secure".', "plaintext"),
    ("Secure the files. Use API key authentication.", "plaintext"),
])
def test_auth_quality_priority_respects_scope(clause, level):
    quality = RequirementCompiler().compile("Build a Python CLI. " + clause).quality_contract
    assert quality.auth_level == level
    assert quality.secrets_in_plaintext is (level == "plaintext")


@pytest.mark.parametrize("value", ["forest", "capillary", "unhashed", "microservice"])
def test_literal_or_unrelated_words_do_not_raise_quality_level(value):
    compiler = RequirementCompiler()
    clause = f'Print "{value}".' if value == "microservice" else f"Print the word {value}."
    actual = compiler.compile("Build a Python CLI. " + clause).quality_contract
    assert actual == compiler.compile("Build a Python CLI.").quality_contract


@pytest.mark.parametrize("service", ["microservices", "services", "APIs"])
def test_positive_service_plurals_keep_quality_floor(service):
    quality = RequirementCompiler().compile(f"Build Python {service}.").quality_contract
    assert quality.overall_level == 5


@pytest.mark.parametrize("feature", ["JWT", "audit trail", "migrations", "structured JSON logging", "integration tests", "production-grade"])
def test_quality_checker_still_requires_evidence_for_positive_features(feature):
    for literal in (False, True):
        clause = f'Print "{feature}".' if literal else f"Require {feature}."
        spec = RequirementCompiler().compile("Build a Python CLI. " + clause)
        artifact = CodeArtifact(
            artifact_id="quality-scoping", plan_id="quality-scoping",
            artifact_manifest={"quality_contract": asdict(spec.quality_contract)},
        )
        failures, evidence = QualityContractChecker().check({}, artifact, spec)
        assert bool(failures) is (not literal)
        assert evidence["passed"] is literal


@pytest.mark.parametrize("clause", [
    "Not just test coverage of 80 percent but also integration tests.",
    "Not less than the test coverage of 80 percent may be accepted.",
])
def test_positive_numeric_coverage_remains_fail_closed(clause):
    assert explicit_test_coverage_target(clause) == 0.8
    spec = RequirementCompiler().compile("Build a Python CLI. " + clause)
    assert spec.quality_contract.test_coverage_target == 0.8
    artifact = CodeArtifact(
        artifact_id="quality-scoping", plan_id="quality-scoping",
        artifact_manifest={"quality_contract": asdict(spec.quality_contract)},
    )
    failures, evidence = QualityContractChecker().check({}, artifact, spec)
    assert failures
    assert evidence["checks"]["explicit_coverage_target_evidenced"] is False
