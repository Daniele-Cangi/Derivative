import pytest

from core.forge.contracts import ArtifactTargetType
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

PIPELINE_REQUIREMENT = (
    "Build a production-grade Python data pipeline that reads CSV files from a watched directory, "
    "validates each row against a configurable schema, persists valid records to SQLite with full audit trail, "
    "rejects and quarantines invalid rows with structured error logging, and exposes a REST health endpoint "
    "showing pipeline statistics."
)

TELEMETRY_REQUIREMENT = (
    "Build a Python CLI that reads JSON Lines telemetry events with fields device_id, timestamp, and temperature_c, "
    "rejects malformed records, missing fields, and invalid timestamps into a quarantine JSONL file, "
    "computes per-device minimum, maximum, and average temperature, writes a summary CSV, and includes "
    "behavioral tests for parsing, quarantine handling, aggregation, and the complete CLI flow."
)

AMBIGUOUS_RISK_REQUIREMENT = (
    "Build a tool that processes business records, identifies risky entries, "
    "produces an appropriate report, and includes tests."
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

    atoms_by_text = {atom.text.lower(): atom for atom in spec.requirement_atoms}
    assert atoms_by_text["survives restarts"].category == "quality"
    assert atoms_by_text["a full audit trail of all requests"].category == "quality"
    assert atoms_by_text["structured json logging"].category == "quality"
    assert all(
        atom.strength != "ambiguous"
        for atom in spec.requirement_atoms
        if any(token in atom.text.lower() for token in ("restart", "audit", "logging"))
    )


def test_pipeline_requirement_is_typed_and_gets_software_build_obligations():
    spec = RequirementCompiler().compile(PIPELINE_REQUIREMENT)

    assert spec.target_artifact_type == ArtifactTargetType.PIPELINE
    assert spec.obligation_contract is not None
    assert spec.obligation_contract.mode == "software_build"
    assert spec.obligation_contract.required_fields
    assert spec.quality_contract.audit_trail is True
    assert spec.quality_contract.health_endpoint is True
    assert spec.quality_contract.structured_logging is True


def test_telemetry_requirement_is_split_into_evidence_bearing_atoms():
    spec = RequirementCompiler().compile(TELEMETRY_REQUIREMENT)
    atoms = {atom.requirement_id: atom for atom in spec.requirement_atoms}

    assert len(atoms) == 6
    assert atoms["R001"].evidence_terms == ["cli_entrypoint"]
    assert atoms["R002"].evidence_terms == ["input_jsonl", "device_id", "temperature_c", "timestamp"]
    assert {"malformed_records", "missing_fields", "invalid_timestamp", "quarantine"} <= set(
        atoms["R003"].evidence_terms
    )
    assert {"minimum", "maximum", "average", "per_device"} <= set(atoms["R004"].evidence_terms)
    assert atoms["R005"].evidence_terms == ["summary_csv"]


def test_material_business_policy_ambiguities_are_preserved():
    spec = RequirementCompiler().compile(AMBIGUOUS_RISK_REQUIREMENT)

    assert "Risk classification criteria are materially unspecified." in spec.ambiguity_flags
    assert "Report schema and output format are materially unspecified." in spec.ambiguity_flags


def test_internal_that_clause_does_not_truncate_callable_contract():
    requirement = (
        "Provide a Python service module exposing def transform_items(items, predicate) "
        "that yields accepted items. If the predicate raises for an item, that item must "
        "be skipped. It must preserve order and tolerate infinite iterators."
    )

    spec = RequirementCompiler().compile(requirement)
    atom_text = " ".join(atom.text.lower() for atom in spec.requirement_atoms)

    assert spec.target_artifact_type == ArtifactTargetType.LIBRARY
    assert "def transform_items(items, predicate)" in atom_text
    assert "yields accepted items" in atom_text
    assert "predicate raises for an item" in atom_text
    assert "item must be skipped" in atom_text
    assert "preserve order" in atom_text
    assert "infinite iterators" in atom_text
    assert all(atom.source_fragment for atom in spec.requirement_atoms)


def test_declared_public_module_and_callable_drive_library_contract():
    requirement = (
        "Create a codec module exposing def encode_stream(stream: bytes) -> str "
        "that returns a deterministic digest and includes tests."
    )

    spec = RequirementCompiler().compile(requirement)

    assert spec.target_artifact_type == ArtifactTargetType.LIBRARY
    assert spec.public_module == "codec"
    interface_atom = next(atom for atom in spec.requirement_atoms if "encode_stream" in atom.text)
    assert interface_atom.verification_method == "interface_contract"


def test_named_callable_component_outranks_pipeline_domain_noun():
    requirement = (
        "Design a data pipeline component 'select_records' accepting an iterator and a predicate. "
        "It yields matching records in input order and includes tests."
    )

    spec = RequirementCompiler().compile(requirement)

    assert spec.target_artifact_type == ArtifactTargetType.LIBRARY
    assert spec.public_module == "select_records"


def test_public_python_names_are_preserved_across_declared_artifact_shapes():
    cases = (
        (
            "Implement a CLI utility 'dupfilter' that reads lines from standard input.",
            ArtifactTargetType.CLI,
            "dupfilter",
        ),
        (
            "Provide a Python library function 'invert_dictionary' with signature "
            "'def invert_dictionary(d: dict[str, str]) -> dict[str, list[str]]'.",
            ArtifactTargetType.LIBRARY,
            "invert_dictionary",
        ),
        (
            "Design a data pipeline component 'filter_by_predicate' accepting an iterator and predicate.",
            ArtifactTargetType.LIBRARY,
            "filter_by_predicate",
        ),
        (
            "Create a service module exposing 'def hash_stream(stream: io.BufferedReader) -> str'.",
            ArtifactTargetType.LIBRARY,
            "service",
        ),
        (
            "Develop a CLI tool 'jsoncompact' that reads JSON from standard input.",
            ArtifactTargetType.CLI,
            "jsoncompact",
        ),
        (
            "Implement a function 'def parse_time_delta(s: str) -> datetime.timedelta'.",
            ArtifactTargetType.LIBRARY,
            "parse_time_delta",
        ),
    )

    for requirement, expected_type, expected_module in cases:
        spec = RequirementCompiler().compile(requirement)

        assert spec.target_artifact_type == expected_type
        assert spec.public_module == expected_module


def test_seeded_pseudorandom_output_without_algorithm_is_materially_ambiguous():
    spec = RequirementCompiler().compile(
        "Define a CLI command 'random_walk' that accepts steps and seed and emits a reproducible "
        "pseudo-random walk seeded by seed."
    )

    assert spec.public_module == "random_walk"
    assert any(
        "pseudo-random algorithm is materially unspecified" in flag.lower()
        for flag in spec.ambiguity_flags
    )


def test_explicitly_unprovable_or_ambiguous_semantics_are_material_flags():
    requirements = (
        "Specify a function that merges infinite streams. Its behavioral equivalence is unprovable "
        "and the ordering semantics are inherently ambiguous.",
        "Create a synchronous method that claims asynchronous behavior, but no mechanism for true "
        "asynchrony is defined, making the semantics formally unprovable.",
    )

    for requirement in requirements:
        spec = RequirementCompiler().compile(requirement)

        assert any("materially unspecified or unprovable" in flag for flag in spec.ambiguity_flags)


def test_normative_must_not_clause_remains_a_hard_requirement():
    spec = RequirementCompiler().compile(
        "Create a service module exposing def hash_stream(stream) -> str. "
        "The function must read the stream sequentially and must not close it."
    )

    atom = next(item for item in spec.requirement_atoms if "must not close it" in item.text.lower())
    assert atom.category == "functional"
    assert atom.strength == "hard"


def test_forbidden_runtime_capabilities_use_static_verification():
    spec = RequirementCompiler().compile(
        "Implement a CLI named 'pyenvlines'. No network, socket, or subprocess "
        "usage is allowed in the contract."
    )

    atom = next(
        item
        for item in spec.requirement_atoms
        if "no network, socket, or subprocess" in item.text.lower()
    )

    assert atom.strength == "hard"
    assert atom.verification_method == "static_analysis"


def test_universal_proof_scope_distinguishes_open_guarantees_from_properties():
    open_guarantee = RequirementCompiler().compile(
        "Build a parser that guarantees support for every possible external encoding."
    )
    property_requirement = RequirementCompiler().compile(
        "Implement a function that accepts arbitrary payload objects and preserves their order."
    )
    exact_cardinality = RequirementCompiler().compile(
        "Implement a parser where the delimiter must appear exactly once."
    )

    assert any(
        atom.verification_method == "universal_proof"
        for atom in open_guarantee.requirement_atoms
    )
    assert any(
        atom.category == "universal_constraint" and atom.verification_method == "property_test"
        for atom in property_requirement.requirement_atoms
    )
    assert all(
        atom.category != "universal_constraint"
        for atom in exact_cardinality.requirement_atoms
        if "exactly once" in atom.text.lower()
    )


@pytest.mark.parametrize(
    ("requirement", "expected_type", "expected_module"),
    [
        (
            "Implement a verified CLI utility named 'pycolmask' that reads a required CSV file.",
            ArtifactTargetType.CLI,
            "pycolmask",
        ),
        (
            "Implement a verified Python library function called 'groupby_runs' in a module "
            "'pygroupbyrun'. The public function is groupby_runs(iterable) -> list[tuple].",
            ArtifactTargetType.LIBRARY,
            "pygroupbyrun",
        ),
        (
            "Implement a verified Python data-pipeline module called 'pyrotatefields' exposing a "
            "single function rotate_fields(rows, field_order, shift=1) -> list[dict].",
            ArtifactTargetType.LIBRARY,
            "pyrotatefields",
        ),
    ],
)
def test_blind_v5_public_artifact_shapes_are_preserved(
    requirement,
    expected_type,
    expected_module,
):
    spec = RequirementCompiler().compile(requirement)

    assert spec.target_artifact_type == expected_type
    assert spec.public_module == expected_module
