import importlib.util
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from core.forge.contracts import BuildSpec, CodeArtifact


class QualityContractChecker:
    def check(
        self,
        materialized: Dict[str, Path],
        code_artifact: CodeArtifact,
        build_spec: BuildSpec,
    ) -> Tuple[List[str], Dict[str, object]]:
        quality = build_spec.quality_contract
        source_paths = sorted(
            path
            for path in materialized
            if path.startswith("src/") and path.endswith(".py") and materialized[path].exists()
        )
        source_by_path = {
            path: materialized[path].read_text(encoding="utf-8")
            for path in source_paths
        }
        combined_source = "\n".join(source_by_path.values())
        combined_lower = combined_source.lower()
        failures: List[str] = []
        checks: Dict[str, bool] = {}

        expected_manifest_contract = asdict(quality)
        manifest_contract = (
            code_artifact.artifact_manifest.get("quality_contract")
            if isinstance(code_artifact.artifact_manifest, dict)
            else None
        )
        checks["manifest_contract_matches"] = manifest_contract == expected_manifest_contract
        if not checks["manifest_contract_matches"]:
            failures.append(
                "quality_contract_violation: artifact manifest does not match BuildSpec quality contract"
            )

        if quality.auth_level == "hashed":
            checks["hashed_auth_uses_bcrypt"] = "bcrypt" in combined_lower
            if not checks["hashed_auth_uses_bcrypt"]:
                failures.append(
                    "quality_contract_violation: auth_level=hashed but bcrypt not found"
                )
            checks["hashed_auth_has_no_fallback"] = (
                "sha256$" not in combined_source and "FORGE_USE_BCRYPT" not in combined_source
            )
            if not checks["hashed_auth_has_no_fallback"]:
                failures.append(
                    "quality_contract_violation: auth_level=hashed must not include sha256/env-gated fallback"
                )
            checks["bcrypt_available"] = importlib.util.find_spec("bcrypt") is not None
            if not checks["bcrypt_available"]:
                failures.append(
                    "quality_contract_violation: bcrypt required but not available"
                )
        if quality.auth_level == "jwt":
            checks["jwt_verification_present"] = "_verify_jwt_token" in combined_source
            if not checks["jwt_verification_present"]:
                failures.append(
                    "quality_contract_violation: auth_level=jwt but jwt verification path not found"
                )
        if not quality.secrets_in_plaintext:
            checks["no_plaintext_api_key_storage"] = "api_key text" not in combined_lower
            if not checks["no_plaintext_api_key_storage"]:
                failures.append(
                    "quality_contract_violation: secrets_in_plaintext=False but API key stored as plaintext TEXT"
                )

        if quality.rate_limit_persistent:
            checks["persistent_rate_limit_not_memory_only"] = "_rate_limit_buckets" not in combined_lower
            if not checks["persistent_rate_limit_not_memory_only"]:
                failures.append(
                    "quality_contract_violation: rate_limit_persistent=True but in-memory dict found"
                )
            checks["persistent_rate_limit_storage_present"] = (
                "rate_limit_hits" in combined_lower or "redis" in combined_lower
            )
            if not checks["persistent_rate_limit_storage_present"]:
                failures.append(
                    "quality_contract_violation: rate_limit_persistent=True but no persistent limiter storage found"
                )
        if quality.rate_limit_scope == "distributed":
            checks["distributed_rate_limit_uses_redis"] = "redis" in combined_lower
            if not checks["distributed_rate_limit_uses_redis"]:
                failures.append(
                    "quality_contract_violation: rate_limit_scope=distributed but no Redis found"
                )

        if quality.audit_trail:
            checks["audit_schema_present"] = any(
                token in combined_lower
                for token in (
                    "create table if not exists events",
                    "create table if not exists audit_events",
                )
            )
            checks["audit_insert_present"] = any(
                token in combined_lower
                for token in ("insert into events", "insert into audit_events")
            )
            if not checks["audit_schema_present"] or not checks["audit_insert_present"]:
                failures.append(
                    "quality_contract_violation: audit_trail=True but executable audit schema/INSERT evidence is incomplete"
                )
        if quality.schema_versioned:
            checks["schema_versioning_present"] = any(
                token in combined_lower
                for token in ("schema_meta", "alembic", "migration")
            )
            if not checks["schema_versioning_present"]:
                failures.append(
                    "quality_contract_violation: schema_versioned=True but schema metadata/migrations not found"
                )
        if quality.health_endpoint:
            checks["health_endpoint_present"] = "/health" in combined_source
            if not checks["health_endpoint_present"]:
                failures.append(
                    "quality_contract_violation: health_endpoint=True but /health route not found"
                )
        if quality.structured_logging:
            checks["structured_logging_present"] = "json.dumps" in combined_source
            if not checks["structured_logging_present"]:
                failures.append(
                    "quality_contract_violation: structured_logging=True but JSON logging not found"
                )

        if quality.integration_tests:
            test_sources = "\n".join(
                materialized[path].read_text(encoding="utf-8")
                for path in code_artifact.test_paths
                if path in materialized and materialized[path].exists()
            ).lower()
            checks["integration_tests_present"] = (
                "test_integration" in test_sources or "integration" in test_sources
            )
            if not checks["integration_tests_present"]:
                failures.append(
                    "quality_contract_violation: integration_tests=True but no integration test found"
                )

        evidence: Dict[str, object] = {
            "target_artifact_type": build_spec.target_artifact_type.value,
            "checked_source_files": source_paths,
            "checks": checks,
            "failures": list(failures),
            "passed": not failures,
        }
        return failures, evidence
