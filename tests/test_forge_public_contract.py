from core.forge.public_contract import (
    PublicImportContract,
    extract_public_import_contract,
    oracle_public_import_error,
    requirement_public_import_error,
)


def test_public_import_contract_is_extracted_from_canonical_requirement():
    contract = extract_public_import_contract(
        "Build the transformer. Public import contract: from package.rules import apply."
    )

    assert contract == PublicImportContract(
        module="package.rules",
        symbol="apply",
        kind="callable",
    )


def test_requirement_contract_rejects_structured_target_mismatch():
    error = requirement_public_import_error(
        "Public import contract: from package.rules import apply.",
        PublicImportContract("package.other", "apply", "function"),
    )

    assert error is not None
    assert "does not match" in error


def test_oracle_must_import_exact_declared_module_and_symbol():
    contract = PublicImportContract("package.rules", "apply", "function")

    assert (
        oracle_public_import_error(
            "from package.rules import apply as target\n",
            contract,
        )
        is None
    )
    assert "from package.rules import apply" in oracle_public_import_error(
        "from package.other import apply\n",
        contract,
    )
