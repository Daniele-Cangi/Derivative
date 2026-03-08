from core.forge.requirement_compiler import RequirementCompiler


BASE_SERVICE_REQUIREMENT = (
    "Build a Python REST microservice with API key authentication "
    "and rate limiting."
)

PRODUCTION_SERVICE_REQUIREMENT = (
    "Build a production-grade Python REST microservice with hashed API keys "
    "using bcrypt, persistent per-user rate limiting that survives restarts, "
    "a full audit trail of all requests, structured JSON logging, "
    "and integration tests."
)


def test_extract_quality_contract_for_base_service_requirement():
    spec = RequirementCompiler().compile(BASE_SERVICE_REQUIREMENT)
    qc = spec.quality_contract

    assert qc.auth_level == "plaintext"
    assert qc.secrets_in_plaintext is True
    assert qc.rate_limit_scope == "per_user"
    assert qc.rate_limit_persistent is False
    assert qc.audit_trail is False
    assert qc.schema_versioned is False
    assert 5 <= qc.overall_level <= 6


def test_extract_quality_contract_for_production_service_requirement():
    spec = RequirementCompiler().compile(PRODUCTION_SERVICE_REQUIREMENT)
    qc = spec.quality_contract

    assert qc.auth_level == "hashed"
    assert qc.secrets_in_plaintext is False
    assert qc.rate_limit_scope == "per_user"
    assert qc.rate_limit_persistent is True
    assert qc.schema_versioned is True
    assert qc.audit_trail is True
    assert qc.health_endpoint is True
    assert qc.structured_logging is True
    assert qc.integration_tests is True
    assert 8 <= qc.overall_level <= 9
