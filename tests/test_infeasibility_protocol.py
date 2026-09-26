import copy
import itertools

import pytest

from core.forge.infeasibility_protocol import (
    PROTOCOL,
    bind_obligation,
    certify_infeasibility,
    validate_obligation,
    verify_infeasibility_certificate,
)
from core.forge.public_contract import PublicImportContract


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
