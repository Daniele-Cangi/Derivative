import re
import hashlib
from typing import Dict, List, Set, Tuple

from core.forge.contracts import ArtifactTargetType, CodeArtifact, FeasiblePlan
from core.forge.domains.registry import DomainAdapterRegistry


class AdapterCapabilityContractChecker:
    def __init__(self, registry: DomainAdapterRegistry | None = None):
        self.registry = registry or DomainAdapterRegistry()

    def check(
        self,
        code_artifact: CodeArtifact,
        plan: FeasiblePlan,
    ) -> Tuple[List[str], List[str], Dict[str, object]]:
        required = self.required_capabilities(plan)
        metadata = (
            code_artifact.artifact_manifest.get("metadata", {})
            if isinstance(code_artifact.artifact_manifest, dict)
            else {}
        )
        if not isinstance(metadata, dict):
            metadata = {}
        declared_adapter = str(metadata.get("domain_adapter", ""))
        declared_raw = metadata.get("adapter_capabilities", [])
        declared = {
            str(item)
            for item in declared_raw
            if isinstance(declared_raw, list)
        }

        if metadata.get("generator") == "forge_candidate_compiler":
            return self._check_candidate_compiler(
                code_artifact=code_artifact,
                plan=plan,
                required=required,
                metadata=metadata,
                declared_adapter=declared_adapter,
                declared=declared,
            )

        adapter = self.registry.select(plan)
        provided = set(adapter.provided_capabilities(plan))

        missing = sorted(required - provided)
        unexpected_declared = sorted(declared - provided)
        undeclared_provided = sorted(provided - declared)
        adapter_matches = declared_adapter == adapter.name
        failures: List[str] = []
        signatures: List[str] = []

        if not adapter_matches:
            failures.append(
                "Artifact manifest domain adapter does not match deterministic registry selection: "
                f"declared={declared_adapter!r}, selected={adapter.name!r}."
            )
            self._append_unique(signatures, "adapter_capability_mismatch")
        if unexpected_declared or undeclared_provided:
            failures.append(
                "Artifact manifest adapter capabilities do not match the selected adapter: "
                f"unexpected={unexpected_declared}, undeclared={undeclared_provided}."
            )
            self._append_unique(signatures, "adapter_capability_manifest_mismatch")
        if missing:
            failures.append(
                f"Selected domain adapter '{adapter.name}' does not implement required capabilities: "
                f"{missing}."
            )
            self._append_unique(signatures, "adapter_capability_mismatch")

        evidence: Dict[str, object] = {
            "selected_adapter": adapter.name,
            "declared_adapter": declared_adapter,
            "adapter_matches": adapter_matches,
            "required_capabilities": sorted(required),
            "provided_capabilities": sorted(provided),
            "declared_capabilities": sorted(declared),
            "missing_capabilities": missing,
            "unexpected_declared_capabilities": unexpected_declared,
            "undeclared_provided_capabilities": undeclared_provided,
            "passed": not failures,
        }
        return failures, signatures, evidence

    def _check_candidate_compiler(
        self,
        code_artifact: CodeArtifact,
        plan: FeasiblePlan,
        required: Set[str],
        metadata: Dict[str, object],
        declared_adapter: str,
        declared: Set[str],
    ) -> Tuple[List[str], List[str], Dict[str, object]]:
        compilation = metadata.get("candidate_compilation", {})
        if not isinstance(compilation, dict):
            compilation = {}
        planned_paths = {
            self._normalize_path(plan_file.path)
            for plan_file in plan.file_tree_plan
        }
        planned_paths.update(
            f"tests/{planned_test.test_name}.py"
            for planned_test in plan.required_tests
            if planned_test.required
        )
        manifest_paths = {
            self._normalize_path(path)
            for path in code_artifact.manifest_paths
        }
        planned_paths -= manifest_paths
        transaction_paths = {
            self._normalize_path(path)
            for path in compilation.get("transaction_paths", [])
            if isinstance(path, str)
        }
        files_by_path = {
            self._normalize_path(generated.path): generated
            for generated in code_artifact.files
        }
        declared_digests = compilation.get("file_sha256", {})
        if not isinstance(declared_digests, dict):
            declared_digests = {}
        required_tokens = {
            "candidate_compiler:openai",
            *(f"candidate_capability:{capability}" for capability in required),
        }
        provenance_mismatches: List[str] = []
        digest_mismatches: List[str] = []
        for path in sorted(planned_paths):
            generated = files_by_path.get(path)
            if generated is None:
                provenance_mismatches.append(path)
                digest_mismatches.append(path)
                continue
            provenance = set(generated.generated_from_plan_sections)
            has_transaction_token = any(
                token.startswith("candidate_transaction:")
                for token in provenance
            )
            if not required_tokens.issubset(provenance) or not has_transaction_token:
                provenance_mismatches.append(path)
            digest = hashlib.sha256(generated.content.encode("utf-8")).hexdigest()
            if declared_digests.get(path) != digest:
                digest_mismatches.append(path)

        adapter_matches = declared_adapter == "candidate"
        capabilities_match = declared == required
        paths_match = transaction_paths == planned_paths
        preflight_passed = compilation.get("preflight_passed") is True
        compiler_contract_valid = all(
            (
                adapter_matches,
                capabilities_match,
                paths_match,
                preflight_passed,
                not provenance_mismatches,
                not digest_mismatches,
            )
        )
        failures: List[str] = []
        signatures: List[str] = []
        if not compiler_contract_valid:
            failures.append(
                "Candidate compiler manifest/provenance contract is invalid: "
                f"adapter_matches={adapter_matches}, capabilities_match={capabilities_match}, "
                f"paths_match={paths_match}, preflight_passed={preflight_passed}, "
                f"provenance_mismatches={provenance_mismatches}, "
                f"digest_mismatches={digest_mismatches}."
            )
            self._append_unique(signatures, "adapter_capability_manifest_mismatch")
            self._append_unique(signatures, "adapter_capability_mismatch")

        provided = set(required) if compiler_contract_valid else set()
        missing = sorted(required - provided)
        evidence: Dict[str, object] = {
            "selected_adapter": "candidate",
            "declared_adapter": declared_adapter,
            "adapter_matches": adapter_matches,
            "required_capabilities": sorted(required),
            "provided_capabilities": sorted(provided),
            "declared_capabilities": sorted(declared),
            "missing_capabilities": missing,
            "unexpected_declared_capabilities": sorted(declared - required),
            "undeclared_provided_capabilities": sorted(provided - declared),
            "candidate_transaction_paths": sorted(transaction_paths),
            "planned_candidate_paths": sorted(planned_paths),
            "paths_match": paths_match,
            "preflight_passed": preflight_passed,
            "provenance_mismatches": provenance_mismatches,
            "digest_mismatches": digest_mismatches,
            "compiler_contract_valid": compiler_contract_valid,
            "passed": not failures,
        }
        return failures, signatures, evidence

    def required_capabilities(self, plan: FeasiblePlan) -> Set[str]:
        atoms = [
            atom.text.lower()
            for atom in plan.build_spec.requirement_atoms
            if atom.category != "ambiguity" and atom.strength in {"hard", "universal"}
        ]
        text = " ".join(atoms)
        target = plan.build_spec.target_artifact_type
        required: Set[str] = set()

        if target == ArtifactTargetType.CLI:
            required.add("cli_entrypoint")
        elif target == ArtifactTargetType.SERVICE:
            required.add("rest_service")
        elif target == ArtifactTargetType.PIPELINE:
            required.add("pipeline_entrypoint")
        elif target == ArtifactTargetType.LIBRARY:
            required.add("library_public_api")
        elif target in {ArtifactTargetType.SCRIPT, ArtifactTargetType.UNKNOWN}:
            required.add("planned_entrypoint")

        if any(
            "csv" in atom and re.search(r"\b(read|reads|input|loads?)\b", atom)
            for atom in atoms
        ):
            required.add("csv_input")
        if "summary csv" in text:
            required.add("summary_csv_output")
        if any(token in text for token in ("expiration", "due_date", "invalid dates", "date format")):
            required.add("date_parsing")
        if "expir" in text and any(token in text for token in ("flag", "within", "less than")):
            required.add("expiration_flagging")
        if "overdue" in text:
            required.add("overdue_flagging")
        if "malformed row" in text and target == ArtifactTargetType.CLI:
            required.add("malformed_csv_handling")
        if "totals and counts" in text and "invoice" in text:
            required.add("invoice_totals_counts")

        if "json lines" in text or "jsonl" in text:
            if "telemetry" in text:
                required.add("jsonl_telemetry_input")
            elif "sales event" in text:
                required.add("jsonl_sales_input")
            else:
                required.add("jsonl_log_input")
        if "summary json" in text or "json report" in text:
            required.add("summary_json_output")
        if "counts_by_level" in text or "counts by level" in text:
            required.add("log_level_counts")
        if "malformed" in text and "quarantine" in text:
            required.add("malformed_record_quarantine")
        if "per-device" in text or "per device" in text:
            required.add("per_device_aggregation")
        if "per-customer" in text or "per customer" in text:
            required.add("per_customer_sales_aggregation")

        if "merge" in text and "json" in text:
            required.add("recursive_json_merge")
        if "replaces lists" in text or "replace lists" in text:
            required.add("json_list_replacement")
        if "non-object root" in text or "non object root" in text:
            required.add("json_object_root_validation")

        if "duplicate file" in text:
            required.add("duplicate_file_grouping")
        if "sha-256" in text or "sha256" in text:
            required.add("sha256_hashing")
        if "recursively scans" in text or "recursive scan" in text:
            required.add("recursive_file_scan")

        if "api-key authentication" in text or "api key authentication" in text:
            required.add("api_key_authentication")
        if "sqlite" in text:
            required.add("sqlite_persistence")
        if "rate limit" in text:
            required.add("per_user_rate_limiting")
        if "idempotent" in text and "event_id" in text:
            required.add("idempotent_event_creation")

        if "watched directory" in text:
            required.add("watched_directory")
        if "schema validation" in text or "validates each row" in text:
            required.add("schema_validation")
        if "audit trail" in text:
            required.add("audit_trail")
        if "health endpoint" in text:
            required.add("health_endpoint")
        if "structured" in text and "logging" in text:
            required.add("structured_logging")
        return required

    @staticmethod
    def _append_unique(collection: List[str], value: str) -> None:
        if value not in collection:
            collection.append(value)

    @staticmethod
    def _normalize_path(path: str) -> str:
        return str(path).strip().replace("\\", "/").lstrip("./")
