import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from core.forge.blind_benchmark import (
    BLIND_BENCHMARK_SCHEMA_VERSION,
    BlindBenchmarkBundle,
    compute_forge_baseline_digest,
    load_blind_bundle,
)
from core.forge.heldout_benchmark import load_heldout_cases


@dataclass(frozen=True)
class BlindFreezeProvenance:
    producer: str
    requirements_origin: str
    oracle_origin: str
    declaration: str


def freeze_blind_bundle(
    bundle_root: str | Path,
    bundle_id: str,
    provenance: BlindFreezeProvenance,
    source_urls: List[str],
    repository_root: str | Path,
    dataset_path: str = "cases.json",
    manifest_name: str = "manifest.json",
) -> BlindBenchmarkBundle:
    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Blind benchmark bundle directory does not exist: {root}")

    identifier = bundle_id.strip()
    if not identifier:
        raise ValueError("Blind benchmark bundle_id must not be empty.")
    _validate_provenance(provenance)
    _validate_sources(source_urls)

    manifest_path = _resolve_output_file(root, manifest_name, "manifest")
    if manifest_path.exists():
        raise FileExistsError(
            f"Blind benchmark manifest already exists and cannot be overwritten: {manifest_path}"
        )
    dataset = _resolve_input_file(root, dataset_path, "dataset")
    cases = load_heldout_cases(str(dataset))

    oracle_digests = {
        case.case_id: _sha256_file(Path(case.oracle.path))
        for case in cases
        if case.oracle is not None
    }
    baseline_digest, baseline_file_count = compute_forge_baseline_digest(repository_root)
    payload = {
        "schema_version": BLIND_BENCHMARK_SCHEMA_VERSION,
        "bundle_id": identifier,
        "frozen_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        "dataset": {
            "path": dataset.relative_to(root).as_posix(),
            "sha256": _sha256_file(dataset),
        },
        "forge_baseline": {
            "sha256": baseline_digest,
            "file_count": baseline_file_count,
        },
        "oracle_sha256": dict(sorted(oracle_digests.items())),
        "source_urls": list(source_urls),
        "provenance": {
            "producer": provenance.producer.strip(),
            "requirements_origin": provenance.requirements_origin.strip(),
            "oracle_origin": provenance.oracle_origin.strip(),
            "independent_of_forge": True,
            "sealed_before_first_execution": True,
            "declaration": provenance.declaration.strip(),
        },
    }
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return load_blind_bundle(
        str(manifest_path),
        repository_root=repository_root,
        verify_baseline=True,
    )


def _validate_provenance(provenance: BlindFreezeProvenance) -> None:
    fields = {
        "producer": provenance.producer,
        "requirements_origin": provenance.requirements_origin,
        "oracle_origin": provenance.oracle_origin,
        "declaration": provenance.declaration,
    }
    missing = [key for key, value in fields.items() if not value.strip()]
    if missing:
        raise ValueError(
            "Blind benchmark provenance requires non-empty fields: "
            + ", ".join(missing)
            + "."
        )


def _validate_sources(source_urls: List[str]) -> None:
    if not all(
        isinstance(item, str) and item.startswith("https://") for item in source_urls
    ):
        raise ValueError("Blind benchmark source_urls must contain only HTTPS URLs.")


def _resolve_input_file(root: Path, raw_path: str, label: str) -> Path:
    path = _resolve_bundle_path(root, raw_path, label)
    if not path.is_file():
        raise ValueError(f"Blind benchmark {label} does not exist: {path}")
    return path


def _resolve_output_file(root: Path, raw_path: str, label: str) -> Path:
    path = _resolve_bundle_path(root, raw_path, label)
    if path.parent != root:
        raise ValueError(f"Blind benchmark {label} must be written at bundle root.")
    return path


def _resolve_bundle_path(root: Path, raw_path: str, label: str) -> Path:
    relative = raw_path.strip()
    if not relative:
        raise ValueError(f"Blind benchmark {label} path is empty.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Blind benchmark {label} escapes the bundle directory.")
    return path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
