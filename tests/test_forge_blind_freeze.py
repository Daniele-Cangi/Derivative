import json
from pathlib import Path

import pytest

from core.forge.blind_benchmark import load_blind_bundle
from core.forge.blind_freeze import BlindFreezeProvenance, freeze_blind_bundle


def _write_external_bundle(root: Path) -> Path:
    oracle_root = root / "oracles" / "V3001"
    oracle_root.mkdir(parents=True)
    (oracle_root / "oracle.py").write_text(
        "from library.core import identity\n\n"
        "def test_identity_contract():\n"
        "    assert identity('external') == 'external'\n",
        encoding="utf-8",
    )
    dataset = root / "cases.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "case_id": "V3001",
                    "requirement": (
                        "Build a Python library exposing identity(value). Include tests. "
                        "Public import contract: from library.core import identity."
                    ),
                    "expected_terminal_status": "verified",
                    "public_contract": {
                        "module": "library.core",
                        "symbol": "identity",
                        "kind": "function",
                    },
                    "tags": ["blind-v3", "externally-authored"],
                    "oracle": {
                        "path": "oracles/V3001/oracle.py",
                        "timeout_seconds": 20,
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset


def _write_baseline(root: Path) -> Path:
    root.mkdir()
    (root / "forge.py").write_text("BASELINE = 'sealed'\n", encoding="utf-8")
    return root


def _freeze(bundle_root: Path, repository_root: Path):
    return freeze_blind_bundle(
        bundle_root=bundle_root,
        bundle_id="forge-blind-v3-external-001",
        provenance=BlindFreezeProvenance(
            producer="Independent benchmark author",
            requirements_origin="Private requirements authored without Forge access",
            oracle_origin="Independent black-box acceptance suite",
            declaration="Requirements and oracles were finalized before Forge execution.",
        ),
        source_urls=["https://example.com/external-benchmark-spec"],
        repository_root=repository_root,
    )


def test_freeze_writes_loadable_schema_v3_manifest_with_provenance_and_digests(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    _write_external_bundle(bundle_root)
    repository_root = _write_baseline(tmp_path / "repository")

    bundle = _freeze(bundle_root, repository_root)
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))

    assert bundle.schema_version == 3
    assert bundle.baseline_verified is True
    assert bundle.provenance is not None
    assert bundle.provenance.independent_of_forge is True
    assert bundle.provenance.sealed_before_first_execution is True
    assert manifest["dataset"]["sha256"] == bundle.dataset_sha256
    assert manifest["oracle_sha256"]["V3001"] == bundle.oracle_sha256["V3001"]
    assert manifest["forge_baseline"]["sha256"] == bundle.baseline_sha256
    assert manifest["frozen_at"].endswith("Z")


def test_freeze_is_one_shot_and_refuses_manifest_overwrite(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    _write_external_bundle(bundle_root)
    repository_root = _write_baseline(tmp_path / "repository")
    _freeze(bundle_root, repository_root)

    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        _freeze(bundle_root, repository_root)


def test_freeze_requires_public_import_contract_for_new_schema(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    dataset = _write_external_bundle(bundle_root)
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    payload[0].pop("public_contract")
    dataset.write_text(json.dumps(payload), encoding="utf-8")
    repository_root = _write_baseline(tmp_path / "repository")

    with pytest.raises(ValueError, match="requires a public_contract object"):
        _freeze(bundle_root, repository_root)


def test_freeze_rejects_oracle_import_outside_declared_contract(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    _write_external_bundle(bundle_root)
    oracle = bundle_root / "oracles" / "V3001" / "oracle.py"
    oracle.write_text(
        "from another_module import identity\n\n"
        "def test_identity_contract():\n"
        "    assert identity('external') == 'external'\n",
        encoding="utf-8",
    )
    repository_root = _write_baseline(tmp_path / "repository")

    with pytest.raises(ValueError, match="rejection_classes=public_import_mismatch"):
        _freeze(bundle_root, repository_root)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("producer", "producer"),
        ("requirements_origin", "requirements_origin"),
        ("oracle_origin", "oracle_origin"),
        ("declaration", "declaration"),
    ],
)
def test_freeze_refuses_incomplete_independent_provenance(tmp_path, field, message):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    _write_external_bundle(bundle_root)
    repository_root = _write_baseline(tmp_path / "repository")
    values = {
        "producer": "External author",
        "requirements_origin": "Independent requirements",
        "oracle_origin": "Independent oracles",
        "declaration": "Frozen before execution",
    }
    values[field] = ""

    with pytest.raises(ValueError, match=message):
        freeze_blind_bundle(
            bundle_root=bundle_root,
            bundle_id="forge-blind-v3-external-001",
            provenance=BlindFreezeProvenance(**values),
            source_urls=["https://example.com/spec"],
            repository_root=repository_root,
        )


def test_freeze_accepts_private_provenance_without_public_source(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    _write_external_bundle(bundle_root)
    repository_root = _write_baseline(tmp_path / "repository")
    provenance = BlindFreezeProvenance("author", "requirements", "oracles", "declaration")

    bundle = freeze_blind_bundle(
        bundle_root=bundle_root,
        bundle_id="blind-v3",
        provenance=provenance,
        source_urls=[],
        repository_root=repository_root,
    )

    assert bundle.source_urls == []


def test_freeze_refuses_non_https_sources(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    _write_external_bundle(bundle_root)
    repository_root = _write_baseline(tmp_path / "repository")

    with pytest.raises(ValueError, match="only HTTPS URLs"):
        freeze_blind_bundle(
            bundle_root=bundle_root,
            bundle_id="blind-v3",
            provenance=BlindFreezeProvenance(
                "author", "requirements", "oracles", "declaration"
            ),
            source_urls=["http://example.com/spec"],
            repository_root=repository_root,
        )


def test_frozen_bundle_rejects_dataset_or_oracle_tampering(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    dataset = _write_external_bundle(bundle_root)
    repository_root = _write_baseline(tmp_path / "repository")
    bundle = _freeze(bundle_root, repository_root)
    original_dataset = dataset.read_text(encoding="utf-8")

    dataset.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset digest mismatch"):
        load_blind_bundle(bundle.manifest_path, repository_root=repository_root)

    dataset.write_text(original_dataset, encoding="utf-8")
    oracle = bundle_root / "oracles" / "V3001" / "oracle.py"
    oracle.write_text("def test_changed():\n    assert False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="oracle digest mismatch"):
        load_blind_bundle(bundle.manifest_path, repository_root=repository_root)


def test_frozen_bundle_rejects_forge_baseline_change(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    _write_external_bundle(bundle_root)
    repository_root = _write_baseline(tmp_path / "repository")
    bundle = _freeze(bundle_root, repository_root)

    (repository_root / "forge.py").write_text("BASELINE = 'changed'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="baseline digest mismatch"):
        load_blind_bundle(bundle.manifest_path, repository_root=repository_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(source_urls=["http://example.com"]), "HTTPS URLs"),
        (
            lambda payload: payload["provenance"].update(independent_of_forge=False),
            "independent_of_forge=true",
        ),
        (lambda payload: payload.update(frozen_at="not-utc"), "UTC timestamp"),
    ],
)
def test_current_schema_loader_enforces_freeze_attestations(tmp_path, mutation, message):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    _write_external_bundle(bundle_root)
    repository_root = _write_baseline(tmp_path / "repository")
    bundle = _freeze(bundle_root, repository_root)
    manifest_path = Path(bundle.manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(payload)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_blind_bundle(str(manifest_path), repository_root=repository_root)


def test_freeze_refuses_dataset_outside_bundle(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    outside = tmp_path / "cases.json"
    outside.write_text("[]", encoding="utf-8")
    repository_root = _write_baseline(tmp_path / "repository")

    with pytest.raises(ValueError, match="escapes the bundle directory"):
        freeze_blind_bundle(
            bundle_root=bundle_root,
            bundle_id="blind-v3",
            provenance=BlindFreezeProvenance(
                "author", "requirements", "oracles", "declaration"
            ),
            source_urls=["https://example.com/spec"],
            repository_root=repository_root,
            dataset_path="../cases.json",
        )


def test_freeze_refuses_external_oracle_with_fixture_contradiction(tmp_path):
    bundle_root = tmp_path / "invalid-oracle-bundle"
    oracle_root = bundle_root / "oracles" / "V5-001"
    oracle_root.mkdir(parents=True)
    (oracle_root / "oracle.py").write_text(
        "from text_reverse import transform\n\n"
        "def test_stdin_to_file():\n"
        "    input_content = 'x yz\\n'\n"
        "    expected = 'x z y\\n'\n",
        encoding="utf-8",
    )
    (bundle_root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "V5-001",
                    "requirement": (
                        "Reverse every word defined as a sequence of non-whitespace "
                        "characters separated by ASCII whitespace, with word order preserved. "
                        "Public import contract: from text_reverse import transform."
                    ),
                    "expected_terminal_status": "verified",
                    "public_contract": {
                        "module": "text_reverse",
                        "symbol": "transform",
                        "kind": "function",
                    },
                    "tags": ["blind-v5", "text-processing"],
                    "oracle": {
                        "path": "oracles/V5-001/oracle.py",
                        "timeout_seconds": 20,
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    repository_root = _write_baseline(tmp_path / "repository")

    with pytest.raises(ValueError, match="rejection_classes=fixture_oracle_mismatch"):
        _freeze(bundle_root, repository_root)

    assert (bundle_root / "manifest.json").exists() is False


def test_freeze_refuses_external_oracle_with_main_argv_contract_mismatch(tmp_path):
    bundle_root = tmp_path / "invalid-argv-oracle-bundle"
    oracle_root = bundle_root / "oracles" / "V5-001"
    oracle_root.mkdir(parents=True)
    (oracle_root / "oracle.py").write_text(
        "from pycolmask import main\n\n"
        "def test_nominal():\n"
        "    argv = ['pycolmask', 'input.csv', '--mask=1']\n"
        "    rc = main(argv)\n"
        "    assert rc == 0\n",
        encoding="utf-8",
    )
    (bundle_root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "V5-001",
                    "requirement": (
                        "Implement a verified CLI utility named 'pycolmask'. "
                        "The main(argv: list[str] | None = None) -> int contract "
                        "must be importable. Public import contract: from pycolmask import main."
                    ),
                    "expected_terminal_status": "verified",
                    "public_contract": {
                        "module": "pycolmask",
                        "symbol": "main",
                        "kind": "cli_entrypoint",
                    },
                    "tags": ["blind-v5", "cli"],
                    "oracle": {
                        "path": "oracles/V5-001/oracle.py",
                        "timeout_seconds": 20,
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    repository_root = _write_baseline(tmp_path / "repository")

    with pytest.raises(ValueError, match="rejection_classes=oracle_contract_mismatch"):
        _freeze(bundle_root, repository_root)

    assert (bundle_root / "manifest.json").exists() is False


def test_freeze_refuses_verified_unicode_case_cardinality_conflict(tmp_path):
    bundle_root = tmp_path / "invalid-unicode-requirement-bundle"
    oracle_root = bundle_root / "oracles" / "V5-003"
    oracle_root.mkdir(parents=True)
    (oracle_root / "oracle.py").write_text(
        "from pyutfinvert import invert_case_preserve_nonletters\n\n"
        "def test_case():\n"
        "    assert invert_case_preserve_nonletters('A') == 'a'\n",
        encoding="utf-8",
    )
    (bundle_root / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "V5-003",
                    "requirement": (
                        "Return a string of the same length as input where each "
                        "Unicode letter has its case inverted. Public import contract: "
                        "from pyutfinvert import invert_case_preserve_nonletters."
                    ),
                    "expected_terminal_status": "verified",
                    "public_contract": {
                        "module": "pyutfinvert",
                        "symbol": "invert_case_preserve_nonletters",
                        "kind": "function",
                    },
                    "tags": ["blind-v5", "unicode"],
                    "oracle": {
                        "path": "oracles/V5-003/oracle.py",
                        "timeout_seconds": 20,
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    repository_root = _write_baseline(tmp_path / "repository")

    with pytest.raises(
        ValueError,
        match="rejection_classes=requirement_finite_witness",
    ):
        _freeze(bundle_root, repository_root)

    assert (bundle_root / "manifest.json").exists() is False
