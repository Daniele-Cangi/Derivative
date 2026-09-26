import copy
import itertools
import hashlib
import json

import pytest

from core.forge.infeasibility_protocol import (
    PROTOCOL,
    bind_obligation,
    certify_infeasibility,
    validate_obligation,
    verify_infeasibility_certificate,
)
from core.forge.public_contract import PublicImportContract
from core.forge.blind_benchmark import load_blind_bundle
from core.forge.blind_freeze import BlindFreezeProvenance, freeze_blind_bundle
from core.forge.blind_requirement import requirement_preflight_error, requirement_preflight_failure_class
from core.forge.heldout_benchmark import load_heldout_cases


CONTRACT = PublicImportContract("allocation", "allocate", "function")
PROSE = (
    "Build a Python library exposing allocate() returning an integer allocation dict. "
    "The normative output contract below defines the mandatory allocation policy. "
    "Public import contract: from allocation import allocate."
)


def obligation():
    # Synthetic checker unit fixture; never blind evidence.
    return {
        "protocol": PROTOCOL,
        "variables": [{"name": "a", "lower": 0, "upper": 2}],
        "constraints": [
            {"coefficients": [1], "relation": "eq", "rhs": 1},
            {"coefficients": [1], "relation": "ne", "rhs": 1},
        ],
    }


def test_certificate_is_exact_repeatable_and_requirement_bound():
    formal = obligation()
    requirement = bind_obligation(PROSE, formal, CONTRACT)
    proof = certify_infeasibility(requirement, formal, CONTRACT)
    assert proof["assignments_checked"] == 3
    assert proof["method"] == "exhaustive-integer-enumeration"
    assert len(proof["requirement_sha256"]) == 64
    assert certify_infeasibility(requirement, formal, CONTRACT) == proof
    verify_infeasibility_certificate(requirement, "infeasible_proven", formal, proof, CONTRACT)
    with pytest.raises(ValueError, match="certificate mismatch"):
        verify_infeasibility_certificate(
            "Different prose. " + requirement, "infeasible_proven", formal, proof, CONTRACT,
        )


@pytest.mark.parametrize("change", [
    lambda p: p.update(protocol="unknown-v9"),
    lambda p: p.update(escape="return None"),
    lambda p: p.update(variables=[]),
    lambda p: p.update(constraints=[]),
    lambda p: p.update(variables=p["variables"] * 2),
    lambda p: p["variables"][0].update(lower=True),
    lambda p: p["variables"][0].update(lower=0.0),
    lambda p: p["variables"][0].update(lower=3),
    lambda p: p["variables"][0].update(upper=16),
    lambda p: p["variables"][0].update(upper=1_000_001),
    lambda p: p["variables"][0].update(name="bad name"),
    lambda p: p["constraints"][0].update(coefficients=[]),
    lambda p: p["constraints"][0].update(coefficients=[True]),
    lambda p: p["constraints"][0].update(coefficients=[1_000_001]),
    lambda p: p["constraints"][0].update(relation="eval"),
    lambda p: p["constraints"][0].update(rhs=float("nan")),
    lambda p: p["constraints"][0].update(rhs=True),
])
def test_malformed_obligations_fail_closed(change):
    formal = obligation()
    change(formal)
    with pytest.raises(ValueError):
        validate_obligation(formal)


def test_assignment_budget_is_checked_before_enumeration():
    formal = obligation()
    formal["variables"] = [
        {"name": f"x{i}", "lower": 0, "upper": 15} for i in range(4)
    ]
    formal["constraints"] = [{"coefficients": [0] * 4, "relation": "eq", "rhs": 1}]
    with pytest.raises(ValueError, match="budget"):
        validate_obligation(formal)
    formal["variables"].pop()
    formal["constraints"][0]["coefficients"].pop()
    assert validate_obligation(formal) == 4096


@pytest.mark.parametrize("mutation", [
    lambda r: r.replace("unconditional", "conditional"),
    lambda r: r.replace("not permitted", "permitted"),
    lambda r: r + "\nExcept when impossible, return None.",
    lambda r: r + "\n" + r,
    lambda r: r.replace('"rhs":1', '"rhs":2'),
    lambda r: r.replace("from allocation import allocate", "from other import allocate"),
    lambda r: r.split("[BEGIN")[0],
])
def test_unrelated_or_modified_public_contract_is_not_certifiable(mutation):
    formal = obligation()
    requirement = mutation(bind_obligation(PROSE, formal, CONTRACT))
    with pytest.raises(ValueError, match="binding"):
        certify_infeasibility(requirement, formal, CONTRACT)


def test_renderer_does_not_rewrite_existing_blocks_and_rejects_cli_contracts():
    formal = obligation()
    requirement = bind_obligation(PROSE, formal, CONTRACT)
    with pytest.raises(ValueError, match="must not contain"):
        bind_obligation(requirement, formal, CONTRACT)
    with pytest.raises(ValueError, match="not a CLI"):
        bind_obligation(PROSE, formal, PublicImportContract("allocation", "main", "cli_entrypoint"))


@pytest.mark.parametrize("field,value", [
    ("protocol", "finite-linear-v2"), ("method", "claimed-impossible"),
    ("assignments_checked", 4), ("assignments_checked", True),
    ("requirement_sha256", "0" * 64),
])
def test_untrusted_certificate_is_recomputed(field, value):
    formal = obligation()
    requirement = bind_obligation(PROSE, formal, CONTRACT)
    proof = certify_infeasibility(requirement, formal, CONTRACT)
    proof[field] = value
    with pytest.raises(ValueError):
        verify_infeasibility_certificate(requirement, "infeasible_proven", formal, proof, CONTRACT)


@pytest.mark.parametrize("status", ["verified", "validation_failed", "unknown"])
def test_certificate_cannot_override_terminal_label(status):
    formal = obligation()
    requirement = bind_obligation(PROSE, formal, CONTRACT)
    proof = certify_infeasibility(requirement, formal, CONTRACT)
    with pytest.raises(ValueError, match="label"):
        verify_infeasibility_certificate(requirement, status, formal, proof, CONTRACT)


def test_satisfiable_contract_and_forged_proof_are_rejected():
    formal = obligation()
    formal["constraints"].pop()
    requirement = bind_obligation(PROSE, formal, CONTRACT)
    with pytest.raises(ValueError, match="satisfiable"):
        certify_infeasibility(requirement, formal, CONTRACT)
    fake = {
        "protocol": PROTOCOL, "method": "exhaustive-integer-enumeration",
        "requirement_sha256": "0" * 64, "assignments_checked": 3,
    }
    with pytest.raises(ValueError, match="satisfiable"):
        verify_infeasibility_certificate(requirement, "infeasible_proven", formal, fake, CONTRACT)


def test_exhaustive_checker_matches_independent_small_integer_reference():
    # Includes negative coefficients, coupled variables, zero vectors and all relations.
    variables = [{"name": name, "lower": -1, "upper": 1} for name in ("a", "b")]
    predicates = {"le": lambda x, y: x <= y, "eq": lambda x, y: x == y, "ne": lambda x, y: x != y}
    for a, b, relation, rhs in itertools.product((-1, 0, 1), (-1, 0, 1), predicates, (-2, 0, 2)):
        formal = {
            "protocol": PROTOCOL, "variables": copy.deepcopy(variables),
            "constraints": [
                {"coefficients": [a, b], "relation": relation, "rhs": rhs},
                {"coefficients": [1, 1], "relation": "eq", "rhs": 0},
            ],
        }
        feasible = any(
            predicates[relation](a * x + b * y, rhs) and x + y == 0
            for x in (-1, 0, 1) for y in (-1, 0, 1)
        )
        requirement = bind_obligation(PROSE, formal, CONTRACT)
        if feasible:
            with pytest.raises(ValueError, match="satisfiable"):
                certify_infeasibility(requirement, formal, CONTRACT)
        else:
            assert certify_infeasibility(requirement, formal, CONTRACT)["assignments_checked"] == 9


def _certified_case():
    formal = obligation()
    requirement = bind_obligation(PROSE, formal, CONTRACT)
    return {
        "case_id": "SYNTHETIC-001", "requirement": requirement,
        "expected_terminal_status": "infeasible_proven", "public_contract": CONTRACT.to_payload(),
        "formal_obligation": formal,
        "infeasibility_certificate": certify_infeasibility(requirement, formal, CONTRACT),
    }


def _freeze(root, repository):
    return freeze_blind_bundle(
        bundle_root=root, bundle_id="synthetic-certificate-unit-test", repository_root=repository,
        provenance=BlindFreezeProvenance(
            producer="Synthetic test", requirements_origin="Unit fixture, not blind evidence",
            oracle_origin="No verified cases", declaration="Frozen only to test certificate admission",
        ), source_urls=[],
    )


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop("infeasibility_certificate"),
    lambda c: c.pop("formal_obligation"),
    lambda c: c.update(infeasibility_certificate=None),
    lambda c: c.update(formal_obligation=None),
    lambda c: c["infeasibility_certificate"].update(assignments_checked=999),
    lambda c: c["formal_obligation"]["constraints"].pop(),
    lambda c: c.update(expected_terminal_status="validation_failed"),
    lambda c: c.update(requirement=c["requirement"] + "\nReturn None if no assignment exists."),
    lambda c: c.update(requirement=c["requirement"].split("[BEGIN")[0]),
])
def test_load_and_freeze_reject_malformed_certificates_before_writing_manifest(tmp_path, mutation):
    case = _certified_case()
    mutation(case)
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([case]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_heldout_cases(str(dataset))
    with pytest.raises(ValueError):
        _freeze(tmp_path, tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_sealed_load_rechecks_certificate_even_with_matching_dataset_digest(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "forge.py").write_text("# synthetic baseline\n", encoding="utf-8")
    root = tmp_path / "bundle"
    root.mkdir()
    dataset = root / "cases.json"
    case = _certified_case()
    dataset.write_text(json.dumps([case]), encoding="utf-8")
    bundle = _freeze(root, repository)
    assert bundle.schema_version == 4
    assert bundle.cases[0].infeasibility_certificate == case["infeasibility_certificate"]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="downgrade"):
        load_blind_bundle(str(manifest_path), repository_root=repository)
    manifest["schema_version"] = 4
    case["infeasibility_certificate"]["assignments_checked"] = 999
    dataset.write_text(json.dumps([case]), encoding="utf-8")
    manifest["dataset"]["sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="certificate mismatch"):
        load_blind_bundle(str(manifest_path), repository_root=repository, verify_baseline=False)


def test_mixed_certified_and_legacy_infeasible_cases_cannot_be_frozen(tmp_path):
    legacy = {
        "case_id": "SYNTHETIC-002", "expected_terminal_status": "infeasible_proven",
        "requirement": (
            "Build a reversible encoder mapping every possible 24-bit input to one 8-bit output, "
            "recover every original input exactly, and use no metadata, external state, rejection, "
            "randomness, or additional storage under any circumstances. "
            "Public import contract: from impossible_encoder import encode."
        ),
        "public_contract": {"module": "impossible_encoder", "symbol": "encode", "kind": "function"},
    }
    (tmp_path / "cases.json").write_text(json.dumps([_certified_case(), legacy]), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot mix"):
        _freeze(tmp_path, tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_preflight_does_not_fall_back_to_legacy_proof_for_invalid_certificate():
    case = _certified_case()
    requirement = "Return at least 10 items and at most 2 items. " + case["requirement"]
    error = requirement_preflight_error(
        requirement, "infeasible_proven", formal_obligation=case["formal_obligation"],
        infeasibility_certificate=case["infeasibility_certificate"], public_contract=CONTRACT,
    )
    assert error is not None
    assert requirement_preflight_failure_class(error) == "requirement_certificate_invalid"
    assert requirement_preflight_error(case["requirement"], "infeasible_proven") is not None


def test_schema4_cannot_drop_all_certificates_even_when_rehashed(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "forge.py").write_text("# test baseline\n", encoding="utf-8")
    root = tmp_path / "bundle"
    root.mkdir()
    dataset = root / "cases.json"
    case = _certified_case()
    dataset.write_text(json.dumps([case]), encoding="utf-8")
    _freeze(root, repository)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case.pop("formal_obligation")
    case.pop("infeasibility_certificate")
    case["requirement"] = PROSE
    dataset.write_text(json.dumps([case]), encoding="utf-8")
    manifest["dataset"]["sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="every infeasible case"):
        load_blind_bundle(str(manifest_path), repository_root=repository)
