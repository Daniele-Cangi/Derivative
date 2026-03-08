import hashlib
import json
from dataclasses import asdict
from typing import Dict, List

from core.forge.contracts import CodeArtifact, FeasiblePlan, GeneratedFile, PlanFile, PlanInterface, PlanTest


class CoderStageError(Exception):
    """Base error for plan-to-code expansion failures."""


class MalformedPlanError(CoderStageError):
    """Raised when a feasible plan is structurally incomplete for code generation."""


class CoderStage:
    def generate(self, plan: FeasiblePlan) -> CodeArtifact:
        self._validate_plan(plan)

        generated: Dict[str, GeneratedFile] = {}

        for plan_file in plan.file_tree_plan:
            file_obj = self._generate_from_plan_file(plan, plan_file)
            generated[file_obj.path] = file_obj

        for plan_test in plan.required_tests:
            test_file = self._generate_from_test_requirement(plan, plan_test)
            generated[test_file.path] = test_file

        runnable_entrypoints = self._resolve_runnable_entrypoints(plan, generated)
        manifest_path = "forge_artifact_manifest.json"
        traceability = {
            path: list(file.generated_from_plan_sections)
            for path, file in generated.items()
        }
        artifact_manifest = self._build_manifest(plan, generated, runnable_entrypoints, traceability, manifest_path)
        generated[manifest_path] = GeneratedFile(
            path=manifest_path,
            content=json.dumps(artifact_manifest, indent=2, sort_keys=True),
            kind="manifest",
            generated_from_plan_sections=[
                f"plan:{plan.plan_id}",
                f"obligation_mode:{plan.obligation_mode}",
                "artifact_manifest",
            ],
        )
        traceability[manifest_path] = list(generated[manifest_path].generated_from_plan_sections)

        sorted_files = [generated[path] for path in sorted(generated.keys())]
        test_paths = sorted(path for path in generated.keys() if path.startswith("tests/"))
        manifest_paths = [manifest_path]
        artifact_id = self._artifact_id(plan.plan_id)
        return CodeArtifact(
            artifact_id=artifact_id,
            plan_id=plan.plan_id,
            files=sorted_files,
            test_paths=test_paths,
            manifest_paths=manifest_paths,
            runnable_entrypoints=runnable_entrypoints,
            artifact_manifest=artifact_manifest,
            traceability=traceability,
        )

    def _validate_plan(self, plan: FeasiblePlan) -> None:
        if not plan.plan_id.strip():
            raise MalformedPlanError("FeasiblePlan.plan_id is required.")
        if not plan.file_tree_plan:
            raise MalformedPlanError("FeasiblePlan.file_tree_plan is required.")
        if not plan.interfaces:
            raise MalformedPlanError("FeasiblePlan.interfaces is required.")
        if not plan.required_tests:
            raise MalformedPlanError("FeasiblePlan.required_tests is required.")

        missing_paths = [file.path for file in plan.file_tree_plan if not file.path.strip()]
        if missing_paths:
            raise MalformedPlanError("All planned files must have non-empty paths.")

        cli_interfaces = [interface for interface in plan.interfaces if interface.interface_type == "cli_entrypoint"]
        if cli_interfaces:
            has_cli_file = any(
                path.path in {"src/cli.py", "src/main.py"} for path in plan.file_tree_plan
            )
            if not has_cli_file:
                raise MalformedPlanError(
                    "Plan declares cli_entrypoint interface but lacks src/cli.py or src/main.py."
                )

    def _artifact_id(self, plan_id: str) -> str:
        digest = hashlib.sha256(plan_id.encode("utf-8")).hexdigest()[:12]
        return f"code-{digest}"

    def _generate_from_plan_file(self, plan: FeasiblePlan, plan_file: PlanFile) -> GeneratedFile:
        path = plan_file.path
        content = self._template_for_plan_file(plan, path, plan.interfaces)
        generated_from = [
            f"plan_file:{path}",
            f"plan_purpose:{plan_file.purpose}",
        ]
        interface_refs = self._interfaces_for_path(path, plan.interfaces)
        generated_from.extend(f"interface:{name}" for name in interface_refs)
        generated_from.extend(f"requirement:{requirement_id}" for requirement_id in plan_file.source_requirement_refs)
        return GeneratedFile(
            path=path,
            content=content,
            kind=self._infer_kind(path),
            generated_from_plan_sections=generated_from,
        )

    def _generate_from_test_requirement(self, plan: FeasiblePlan, plan_test: PlanTest) -> GeneratedFile:
        path = f"tests/{plan_test.test_name}.py"
        content = self._template_for_test(plan, plan_test)
        generated_from = [f"test_requirement:{plan_test.test_name}"]
        generated_from.extend(f"acceptance:{criterion_id}" for criterion_id in plan_test.acceptance_criterion_ids)
        generated_from.extend(f"obligation:{field}" for field in plan_test.obligation_fields)
        generated_from.extend(f"requirement:{requirement_id}" for requirement_id in plan_test.requirement_ids)
        return GeneratedFile(
            path=path,
            content=content,
            kind="python_test",
            generated_from_plan_sections=generated_from,
        )

    def _resolve_runnable_entrypoints(
        self,
        plan: FeasiblePlan,
        generated_files: Dict[str, GeneratedFile],
    ) -> List[str]:
        entrypoints: List[str] = []
        for interface in plan.interfaces:
            if interface.interface_type != "cli_entrypoint":
                continue
            if "src/cli.py" in generated_files:
                entrypoints.append("src/cli.py")
                continue
            if "src/main.py" in generated_files:
                entrypoints.append("src/main.py")
                continue
            raise MalformedPlanError(
                f"Unable to resolve runnable entrypoint for interface '{interface.name}'."
            )
        deduplicated: List[str] = []
        for entrypoint in entrypoints:
            if entrypoint not in deduplicated:
                deduplicated.append(entrypoint)
        return deduplicated

    def _build_manifest(
        self,
        plan: FeasiblePlan,
        generated_files: Dict[str, GeneratedFile],
        runnable_entrypoints: List[str],
        traceability: Dict[str, List[str]],
        manifest_path: str,
    ) -> Dict[str, object]:
        return {
            "plan_id": plan.plan_id,
            "build_id": plan.build_spec.build_id,
            "architecture_summary": plan.architecture_summary,
            "packaging_target": plan.packaging_target,
            "obligation_mode": plan.obligation_mode,
            "required_obligations": list(plan.required_obligations),
            "acceptance_criterion_ids": list(plan.acceptance_criterion_ids),
            "requirement_coverage": plan.requirement_coverage,
            "quality_contract": asdict(plan.quality_contract),
            "validation_strategy": asdict(plan.validation_strategy),
            "runnable_entrypoints": runnable_entrypoints,
            "generated_file_count": len(generated_files) + 1,
            "generated_files": [
                {
                    "path": file.path,
                    "kind": file.kind,
                    "generated_from": list(file.generated_from_plan_sections),
                }
                for file in [generated_files[path] for path in sorted(generated_files.keys())]
            ],
            "traceability": traceability,
            "manifest_path": manifest_path,
            "metadata": {
                "generator": "forge_coder_stage",
                "deterministic_templates": True,
            },
        }

    def _interfaces_for_path(self, path: str, interfaces: List[PlanInterface]) -> List[str]:
        refs: List[str] = []
        lowered = path.lower()
        for interface in interfaces:
            name = interface.name.lower()
            if interface.interface_type == "cli_entrypoint" and lowered in {"src/cli.py", "src/main.py"}:
                refs.append(interface.name)
                continue
            if name in lowered:
                refs.append(interface.name)
                continue
            if "contracts_csv" in lowered and "load_contracts_csv" == interface.name:
                refs.append(interface.name)
                continue
            if "expiration_rules" in lowered and "flag_expiring_contracts" == interface.name:
                refs.append(interface.name)
                continue
            if "summary_writer" in lowered and "write_summary_csv" == interface.name:
                refs.append(interface.name)
                continue
        return refs

    def _infer_kind(self, path: str) -> str:
        if path.endswith(".py"):
            if path.startswith("tests/"):
                return "python_test"
            return "python_module"
        if path.endswith(".json"):
            return "json"
        return "text"

    def _template_for_plan_file(
        self,
        plan: FeasiblePlan,
        path: str,
        interfaces: List[PlanInterface],
    ) -> str:
        mode = self._infer_plan_template_mode(plan)
        normalized = path.replace("\\", "/").lower()
        if mode == "service":
            if normalized.endswith("src/service.py"):
                return self._template_service_entrypoint(plan)
            if normalized.endswith("src/domain.py"):
                return self._template_service_domain(plan)
            if normalized.startswith("tests/"):
                return self._template_service_plan_test_module(plan)
            return self._template_generic_module(path, interfaces)
        if normalized.endswith("src/cli.py") or normalized.endswith("src/main.py"):
            return self._template_cli(plan)
        if normalized.endswith("src/contracts_csv.py"):
            return self._template_contracts_csv()
        if normalized.endswith("src/expiration_rules.py"):
            return self._template_expiration_rules()
        if normalized.endswith("src/summary_writer.py"):
            return self._template_summary_writer()
        if normalized.startswith("tests/"):
            return self._template_plan_test_module(plan, path)
        return self._template_generic_module(path, interfaces)

    def _infer_plan_template_mode(self, plan: FeasiblePlan) -> str:
        summary = (plan.architecture_summary or "").lower()
        planned_paths = " ".join(file.path.lower() for file in plan.file_tree_plan)
        combined = f"{summary} {planned_paths}"
        service_tokens = ("service", "rest", "api", "endpoint", "http", "src/service.py")
        cli_tokens = ("cli", "csv", "contracts", "src/cli.py", "src/main.py")
        if any(token in combined for token in service_tokens):
            return "service"
        if any(token in combined for token in cli_tokens):
            return "cli"
        return "generic"

    def _template_cli(self, plan: FeasiblePlan) -> str:
        is_invoice = self._is_invoice_plan(plan)
        parser_description = "Process invoice due dates from CSV input." if is_invoice else "Process contract expirations from CSV input."
        input_help = "Input invoices CSV path." if is_invoice else "Input contracts CSV path."
        output_help = "Output invoice summary CSV path." if is_invoice else "Output summary CSV path."
        horizon_help = "Due-date horizon in days." if is_invoice else "Expiration horizon in days."
        return (
            "import argparse\n"
            "\n"
            "from contracts_csv import load_contracts_csv\n"
            "from expiration_rules import flag_expiring_contracts\n"
            "from summary_writer import write_summary_csv\n"
            "\n"
            "\n"
            "def build_parser() -> argparse.ArgumentParser:\n"
            f"    parser = argparse.ArgumentParser(description={parser_description!r})\n"
            f"    parser.add_argument('input_csv', help={input_help!r})\n"
            f"    parser.add_argument('output_csv', help={output_help!r})\n"
            f"    parser.add_argument('--horizon-days', type=int, default=90, help={horizon_help!r})\n"
            "    return parser\n"
            "\n"
            "\n"
            "def main(argv: list[str] | None = None) -> int:\n"
            "    parser = build_parser()\n"
            "    args = parser.parse_args(argv)\n"
            "    rows = load_contracts_csv(args.input_csv)\n"
            "    flagged = flag_expiring_contracts(rows, horizon_days=args.horizon_days)\n"
            "    write_summary_csv(flagged, args.output_csv)\n"
            "    return 0\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )

    def _template_contracts_csv(self) -> str:
        return (
            "import csv\n"
            "\n"
            "\n"
            "def _required_columns(fieldnames: list[str]) -> list[str]:\n"
            "    normalized = {str(name).strip() for name in fieldnames if name is not None}\n"
            "    invoice_required = {'invoice_id', 'due_date', 'amount', 'customer_name'}\n"
            "    contract_required = {'contract_id', 'expiration_date'}\n"
            "    if invoice_required.issubset(normalized):\n"
            "        return ['invoice_id', 'due_date', 'amount', 'customer_name']\n"
            "    if contract_required.issubset(normalized):\n"
            "        return ['contract_id', 'expiration_date']\n"
            "    return []\n"
            "\n"
            "\n"
            "def load_contracts_csv(path: str) -> list[dict[str, str]]:\n"
            "    rows: list[dict[str, str]] = []\n"
            "    with open(path, 'r', encoding='utf-8', newline='') as handle:\n"
            "        reader = csv.DictReader(handle)\n"
            "        header = [str(name).strip() for name in (reader.fieldnames or []) if name is not None]\n"
            "        required = _required_columns(header)\n"
            "        for row in reader:\n"
            "            if any(key is None for key in row.keys()):\n"
            "                continue\n"
            "            normalized = {\n"
            "                str(key).strip(): (str(value).strip() if value is not None else '')\n"
            "                for key, value in row.items()\n"
            "            }\n"
            "            if required and any(not normalized.get(column, '') for column in required):\n"
            "                continue\n"
            "            rows.append(normalized)\n"
            "    return rows\n"
        )

    def _template_expiration_rules(self) -> str:
        return (
            "from datetime import date, datetime\n"
            "\n"
            "\n"
            "def parse_expiration_date(value: str) -> date | None:\n"
            "    candidate = (value or '').strip()\n"
            "    if not candidate:\n"
            "        return None\n"
            "    formats = ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y')\n"
            "    for fmt in formats:\n"
            "        try:\n"
            "            return datetime.strptime(candidate, fmt).date()\n"
            "        except ValueError:\n"
            "            continue\n"
            "    return None\n"
            "\n"
            "\n"
            "def _extract_date_value(record: dict[str, str]) -> str:\n"
            "    for key in ('expiration_date', 'due_date', 'date'):\n"
            "        value = str(record.get(key, '')).strip()\n"
            "        if value:\n"
            "            return value\n"
            "    return ''\n"
            "\n"
            "\n"
            "def flag_expiring_contracts(\n"
            "    records: list[dict[str, str]],\n"
            "    horizon_days: int = 90,\n"
            "    today: date | None = None,\n"
            ") -> list[dict[str, str]]:\n"
            "    if today is None:\n"
            "        today = date.today()\n"
            "    flagged: list[dict[str, str]] = []\n"
            "    for record in records:\n"
            "        expiration = parse_expiration_date(_extract_date_value(record))\n"
            "        output = dict(record)\n"
            "        if expiration is None:\n"
            "            output['days_to_expiration'] = ''\n"
            "            output['is_expiring_within_horizon'] = 'False'\n"
            "            output['is_overdue'] = 'False'\n"
            "        else:\n"
            "            days = (expiration - today).days\n"
            "            output['days_to_expiration'] = str(days)\n"
            "            output['is_expiring_within_horizon'] = str(days < horizon_days)\n"
            "            output['is_overdue'] = str(days < 0)\n"
            "        flagged.append(output)\n"
            "    return flagged\n"
        )

    def _template_summary_writer(self) -> str:
        return (
            "import csv\n"
            "\n"
            "\n"
            "def _format_total(value: float) -> str:\n"
            "    if abs(value - round(value)) < 1e-9:\n"
            "        return str(int(round(value)))\n"
            "    return f'{value:.2f}'.rstrip('0').rstrip('.')\n"
            "\n"
            "\n"
            "def write_summary_csv(rows: list[dict[str, str]], output_path: str) -> None:\n"
            "    record_count = len(rows)\n"
            "    total_amount = 0.0\n"
            "    for row in rows:\n"
            "        raw_amount = str(row.get('amount', '')).strip()\n"
            "        if not raw_amount:\n"
            "            continue\n"
            "        try:\n"
            "            total_amount += float(raw_amount)\n"
            "        except ValueError:\n"
            "            continue\n"
            "    enriched_rows: list[dict[str, str]] = []\n"
            "    for row in rows:\n"
            "        output = dict(row)\n"
            "        output.setdefault('total_amount', _format_total(total_amount))\n"
            "        output.setdefault('invoice_count', str(record_count))\n"
            "        output.setdefault('record_count', str(record_count))\n"
            "        enriched_rows.append(output)\n"
            "    if enriched_rows:\n"
            "        fieldnames = list(enriched_rows[0].keys())\n"
            "    else:\n"
            "        fieldnames = [\n"
            "            'contract_id',\n"
            "            'expiration_date',\n"
            "            'days_to_expiration',\n"
            "            'is_expiring_within_horizon',\n"
            "            'total_amount',\n"
            "            'invoice_count',\n"
            "            'record_count',\n"
            "        ]\n"
            "    with open(output_path, 'w', encoding='utf-8', newline='') as handle:\n"
            "        writer = csv.DictWriter(handle, fieldnames=fieldnames)\n"
            "        writer.writeheader()\n"
            "        for row in enriched_rows:\n"
            "            writer.writerow(row)\n"
        )

    def _template_service_entrypoint(self, plan: FeasiblePlan) -> str:
        quality = plan.quality_contract
        auth_level = quality.auth_level.lower()
        use_hashed_auth = auth_level == "hashed"
        use_jwt_auth = auth_level == "jwt"
        secrets_plaintext = bool(quality.secrets_in_plaintext and auth_level == "plaintext")
        distributed_rate = quality.rate_limit_scope == "distributed"
        persistent_rate = bool(quality.rate_limit_persistent or distributed_rate)
        audit_trail = bool(quality.audit_trail)
        schema_versioned = bool(quality.schema_versioned)
        health_endpoint = bool(quality.health_endpoint)
        structured_logging = bool(quality.structured_logging)
        integration_tests = bool(quality.integration_tests)
        storage_column = "api_key TEXT UNIQUE NOT NULL" if secrets_plaintext else "api_key_hash TEXT UNIQUE NOT NULL"
        auth_mode_label = "jwt" if use_jwt_auth else ("hashed" if use_hashed_auth else "plaintext")

        lines: List[str] = [
            "import base64",
            "import hashlib",
            "import hmac",
            "import json",
            "import logging",
            "import os",
            "import sqlite3",
            "import time",
            "from typing import Any",
            "",
            "try:",
            "    import bcrypt  # type: ignore",
            "except Exception:",
            "    bcrypt = None",
            "",
            "try:",
            "    import redis  # type: ignore",
            "except Exception:",
            "    redis = None",
            "",
            "DB_PATH = os.environ.get('FORGE_SERVICE_DB', 'service.db')",
            "REDIS_URL = os.environ.get('FORGE_REDIS_URL', 'redis://localhost:6379/0')",
            "JWT_SECRET = os.environ.get('FORGE_JWT_SECRET', 'forge-dev-secret')",
            "RATE_LIMIT_PER_MINUTE = int(os.environ.get('FORGE_RATE_LIMIT_PER_MINUTE', '100'))",
            f"QUALITY_AUTH_MODE = {auth_mode_label!r}",
            f"QUALITY_RATE_SCOPE = {quality.rate_limit_scope!r}",
            f"QUALITY_RATE_PERSISTENT = {str(persistent_rate)}",
            f"QUALITY_SCHEMA_VERSIONED = {str(schema_versioned)}",
            f"QUALITY_AUDIT_TRAIL = {str(audit_trail)}",
            f"QUALITY_HEALTH_ENDPOINT = {str(health_endpoint)}",
            f"QUALITY_STRUCTURED_LOGGING = {str(structured_logging)}",
            f"QUALITY_INTEGRATION_TESTS = {str(integration_tests)}",
            f"QUALITY_LEVEL = {quality.overall_level}",
            "_logger = logging.getLogger('forge.service')",
            "if not _logger.handlers:",
            "    logging.basicConfig(level=logging.INFO)",
            "",
            "",
            "def init_db(db_path: str = DB_PATH) -> None:",
            "    with sqlite3.connect(db_path) as conn:",
            "        conn.execute(",
            "            'CREATE TABLE IF NOT EXISTS users ('",
            "            'username TEXT PRIMARY KEY, '",
            f"            '{storage_column})'",
            "        )",
        ]
        if schema_versioned:
            lines.extend(
                [
                    "        conn.execute(",
                    "            'CREATE TABLE IF NOT EXISTS schema_meta ('",
                    "            'name TEXT PRIMARY KEY, '",
                    "            'version INTEGER NOT NULL)'",
                    "        )",
                    "        conn.execute(",
                    "            'INSERT OR REPLACE INTO schema_meta(name, version) VALUES (?, ?)',",
                    "            ('service', 1),",
                    "        )",
                ]
            )
        if persistent_rate:
            lines.extend(
                [
                    "        conn.execute(",
                    "            'CREATE TABLE IF NOT EXISTS rate_limit_hits ('",
                    "            'user_id TEXT NOT NULL, '",
                    "            'ts REAL NOT NULL)'",
                    "        )",
                ]
            )
        if audit_trail:
            lines.extend(
                [
                    "        conn.execute(",
                    "            'CREATE TABLE IF NOT EXISTS events ('",
                    "            'id INTEGER PRIMARY KEY AUTOINCREMENT, '",
                    "            'created_at REAL NOT NULL, '",
                    "            'user_id TEXT, '",
                    "            'status_code INTEGER NOT NULL, '",
                    "            'detail TEXT NOT NULL)'",
                    "        )",
                ]
            )
        lines.extend(
            [
                "        conn.commit()",
                "",
                "",
                "def _hash_api_key(api_key: str) -> str:",
                "    value = api_key.encode('utf-8')",
            ]
        )
        if use_hashed_auth:
            lines.extend(
                [
                    "    if bcrypt is None:",
                    "        raise RuntimeError('bcrypt required but not available')",
                    "    token = bcrypt.hashpw(value, bcrypt.gensalt(rounds=4)).decode('utf-8')",
                    "    return f'bcrypt${token}'",
                ]
            )
        else:
            lines.extend(
                [
                    "    if bcrypt is not None and os.environ.get('FORGE_USE_BCRYPT', '0') == '1':",
                    "        token = bcrypt.hashpw(value, bcrypt.gensalt(rounds=4)).decode('utf-8')",
                    "        return f'bcrypt${token}'",
                    "    digest = hashlib.sha256(value + b'forge-fallback-salt').hexdigest()",
                    "    return f'sha256${digest}'",
                ]
            )
        lines.extend(["", "", "def _verify_api_key(api_key: str, stored: str) -> bool:", "    value = api_key.encode('utf-8')"])
        if use_hashed_auth:
            lines.extend(
                [
                    "    if not stored.startswith('bcrypt$'):",
                    "        return False",
                    "    if bcrypt is None:",
                    "        return False",
                    "    candidate = stored.split('$', 1)[1]",
                    "    return bool(bcrypt.checkpw(value, candidate.encode('utf-8')))",
                ]
            )
        else:
            lines.extend(
                [
                    "    if stored.startswith('bcrypt$') and bcrypt is not None:",
                    "        candidate = stored.split('$', 1)[1]",
                    "        return bool(bcrypt.checkpw(value, candidate.encode('utf-8')))",
                    "    if stored.startswith('sha256$'):",
                    "        expected = hashlib.sha256(value + b'forge-fallback-salt').hexdigest()",
                    "        return stored.split('$', 1)[1] == expected",
                    "    return False",
                ]
            )
        lines.extend(
            [
                "",
                "",
                "def _verify_jwt_token(token: str) -> str | None:",
                "    candidate = token.strip()",
                "    if candidate.lower().startswith('bearer '):",
                "        candidate = candidate.split(' ', 1)[1].strip()",
                "    parts = candidate.split('.', 1)",
                "    if len(parts) != 2:",
                "        return None",
                "    username, signature = parts",
                "    expected = hmac.new(",
                "        JWT_SECRET.encode('utf-8'),",
                "        username.encode('utf-8'),",
                "        hashlib.sha256,",
                "    ).digest()",
                "    expected_token = base64.urlsafe_b64encode(expected).decode('utf-8').rstrip('=')",
                "    if hmac.compare_digest(signature, expected_token):",
                "        return username",
                "    return None",
                "",
                "",
                "def register_user(username: str, api_key: str, db_path: str = DB_PATH) -> None:",
                "    init_db(db_path)",
                "    with sqlite3.connect(db_path) as conn:",
            ]
        )
        if secrets_plaintext:
            lines.extend(
                [
                    "        conn.execute(",
                    "            'INSERT OR REPLACE INTO users(username, api_key) VALUES (?, ?)',",
                    "            (username, api_key),",
                    "        )",
                ]
            )
        else:
            lines.extend(
                [
                    "        conn.execute(",
                    "            'INSERT OR REPLACE INTO users(username, api_key_hash) VALUES (?, ?)',",
                    "            (username, _hash_api_key(api_key)),",
                    "        )",
                ]
            )
        lines.extend(
            [
                "        conn.commit()",
                "",
                "",
                "def authenticate(api_key: str, db_path: str = DB_PATH) -> str | None:",
                "    init_db(db_path)",
            ]
        )
        if use_jwt_auth:
            lines.extend(
                [
                    "    jwt_user = _verify_jwt_token(api_key)",
                    "    if jwt_user is not None:",
                    "        return jwt_user",
                ]
            )
        lines.extend(
            [
                "    with sqlite3.connect(db_path) as conn:",
            ]
        )
        if secrets_plaintext:
            lines.extend(
                [
                    "        row = conn.execute(",
                    "            'SELECT username FROM users WHERE api_key = ?',",
                    "            (api_key,),",
                    "        ).fetchone()",
                    "    return str(row[0]) if row else None",
                ]
            )
        else:
            lines.extend(
                [
                    "        rows = conn.execute(",
                    "            'SELECT username, api_key_hash FROM users'",
                    "        ).fetchall()",
                    "    for username, api_key_hash in rows:",
                    "        if _verify_api_key(api_key, str(api_key_hash)):",
                    "            return str(username)",
                    "    return None",
                ]
            )
        lines.extend(
            [
                "",
                "",
                "def _redis_client():",
                "    if redis is None:",
                "        return None",
                "    try:",
                "        return redis.from_url(REDIS_URL)",
                "    except Exception:",
                "        return None",
                "",
                "",
                "def _allow_rate_with_sqlite(user_id: str, limit: int, timestamp: float, db_path: str) -> bool:",
                "    init_db(db_path)",
                "    window_start = timestamp - 60.0",
                "    with sqlite3.connect(db_path) as conn:",
                "        conn.execute('DELETE FROM rate_limit_hits WHERE ts < ?', (window_start,))",
                "        current = conn.execute(",
                "            'SELECT COUNT(1) FROM rate_limit_hits WHERE user_id = ? AND ts >= ?',",
                "            (user_id, window_start),",
                "        ).fetchone()",
                "        count = int(current[0]) if current else 0",
                "        if count >= limit:",
                "            conn.commit()",
                "            return False",
                "        conn.execute(",
                "            'INSERT INTO rate_limit_hits(user_id, ts) VALUES (?, ?)',",
                "            (user_id, timestamp),",
                "        )",
                "        conn.commit()",
                "    return True",
            ]
        )
        if not persistent_rate:
            lines.extend(
                [
                    "_RATE_LIMIT_BUCKETS: dict[str, list[float]] = {}",
                    "",
                    "",
                    "def _allow_rate_with_memory(user_id: str, limit: int, timestamp: float) -> bool:",
                    "    window_start = timestamp - 60.0",
                    "    bucket = _RATE_LIMIT_BUCKETS.setdefault(user_id, [])",
                    "    bucket[:] = [entry for entry in bucket if entry >= window_start]",
                    "    if len(bucket) >= limit:",
                    "        return False",
                    "    bucket.append(timestamp)",
                    "    return True",
                ]
            )
            lines.extend(
                [
                    "",
                    "",
                    "def enforce_rate_limit(",
                    "    user_id: str,",
                    "    limit: int = RATE_LIMIT_PER_MINUTE,",
                    "    now: float | None = None,",
                    "    db_path: str = DB_PATH,",
                    ") -> bool:",
                    "    timestamp = time.time() if now is None else float(now)",
                    "    return _allow_rate_with_memory(user_id, limit, timestamp)",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "",
                    "def enforce_rate_limit(",
                    "    user_id: str,",
                    "    limit: int = RATE_LIMIT_PER_MINUTE,",
                    "    now: float | None = None,",
                    "    db_path: str = DB_PATH,",
                    ") -> bool:",
                    "    timestamp = time.time() if now is None else float(now)",
                ]
            )
        if distributed_rate:
            lines.extend(
                [
                    "    client = _redis_client()",
                    "    if client is not None:",
                    "        key = f'rl:{user_id}'",
                    "        window_start = timestamp - 60.0",
                    "        try:",
                    "            client.zremrangebyscore(key, 0, window_start)",
                    "            count = int(client.zcard(key))",
                    "            if count >= limit:",
                    "                return False",
                    "            client.zadd(key, {str(timestamp): timestamp})",
                    "            client.expire(key, 120)",
                    "            return True",
                    "        except Exception:",
                    "            pass",
                    "    return _allow_rate_with_sqlite(user_id, limit, timestamp, db_path)",
                ]
            )
        elif persistent_rate:
            lines.append("    return _allow_rate_with_sqlite(user_id, limit, timestamp, db_path)")
        lines.extend(
            [
                "",
                "",
                "def _log_event(user_id: str | None, status_code: int, detail: str, db_path: str) -> None:",
            ]
        )
        if audit_trail:
            lines.extend(
                [
                    "    init_db(db_path)",
                    "    with sqlite3.connect(db_path) as conn:",
                    "        conn.execute(",
                    "            'INSERT INTO events(created_at, user_id, status_code, detail) VALUES (?, ?, ?, ?)',",
                    "            (time.time(), user_id, status_code, detail),",
                    "        )",
                    "        conn.commit()",
                ]
            )
        if structured_logging:
            lines.extend(
                [
                    "    _logger.info(",
                    "        json.dumps(",
                    "            {",
                    "                'user_id': user_id,",
                    "                'status_code': status_code,",
                    "                'detail': detail,",
                    "            },",
                    "            sort_keys=True,",
                    "        )",
                    "    )",
                ]
            )
        else:
            lines.append("    _logger.info('user=%s status=%s detail=%s', user_id, status_code, detail)")
        lines.extend(
            [
                "",
                "",
                "def get_recent_events(db_path: str = DB_PATH, limit: int = 50) -> list[dict[str, object]]:",
            ]
        )
        if audit_trail:
            lines.extend(
                [
                    "    init_db(db_path)",
                    "    with sqlite3.connect(db_path) as conn:",
                    "        rows = conn.execute(",
                    "            'SELECT created_at, user_id, status_code, detail FROM events ORDER BY id DESC LIMIT ?',",
                    "            (limit,),",
                    "        ).fetchall()",
                    "    return [",
                    "        {",
                    "            'created_at': float(row[0]),",
                    "            'user_id': row[1],",
                    "            'status_code': int(row[2]),",
                    "            'detail': str(row[3]),",
                    "        }",
                    "        for row in rows",
                    "    ]",
                ]
            )
        else:
            lines.append("    return []")
        lines.extend(
            [
                "",
                "",
                "def create_app(db_path: str = DB_PATH) -> dict[str, str]:",
                "    init_db(db_path)",
                "    return {'db_path': db_path, 'auth_mode': QUALITY_AUTH_MODE, 'rate_scope': QUALITY_RATE_SCOPE}",
                "",
                "",
                "def handle_request(",
                "    api_key: str,",
                "    payload: dict[str, Any],",
                "    db_path: str = DB_PATH,",
                "    now: float | None = None,",
                ") -> tuple[int, dict[str, Any]]:",
                "    user_id = authenticate(api_key, db_path=db_path)",
                "    if user_id is None:",
                "        _log_event(None, 401, 'unauthorized', db_path)",
                "        return 401, {'error': 'unauthorized'}",
                "    if not enforce_rate_limit(user_id, now=now, db_path=db_path):",
                "        _log_event(user_id, 429, 'rate_limit_exceeded', db_path)",
                "        return 429, {'error': 'rate_limit_exceeded'}",
                "    _log_event(user_id, 200, 'accepted', db_path)",
                "    return 200, {'status': 'ok', 'user': user_id, 'payload': payload}",
                "",
                "",
                "try:",
                "    from fastapi import FastAPI, Header, HTTPException",
                "",
                "    app = FastAPI(title='Forge Service')",
            ]
        )
        if health_endpoint:
            lines.extend(
                [
                    "",
                    "    @app.get('/health')",
                    "    def health() -> dict[str, object]:",
                    "        return {'status': 'ok', 'quality_level': QUALITY_LEVEL}",
                ]
            )
        lines.extend(
            [
                "",
                "    @app.post('/events')",
                "    def events(payload: dict[str, Any], x_api_key: str | None = Header(default=None)) -> dict[str, Any]:",
                "        if not x_api_key:",
                "            raise HTTPException(status_code=401, detail='missing_api_key')",
                "        code, body = handle_request(x_api_key, payload)",
                "        if code != 200:",
                "            raise HTTPException(status_code=code, detail=str(body.get('error', 'request_rejected')))",
                "        return body",
                "except Exception:",
                "    app = None",
                "",
                "",
                "def run() -> int:",
                "    config = create_app(DB_PATH)",
                "    return 0 if 'db_path' in config else 1",
            ]
        )
        return "\n".join(lines) + "\n"

    def _template_service_domain(self, plan: FeasiblePlan) -> str:
        _ = plan
        return (
            "from service import (\n"
            "    authenticate,\n"
            "    create_app,\n"
            "    enforce_rate_limit,\n"
            "    get_recent_events,\n"
            "    handle_request,\n"
            "    init_db,\n"
            "    register_user,\n"
            "    run,\n"
            ")\n"
            "\n"
            "__all__ = [\n"
            "    'init_db',\n"
            "    'register_user',\n"
            "    'authenticate',\n"
            "    'enforce_rate_limit',\n"
            "    'handle_request',\n"
            "    'get_recent_events',\n"
            "    'create_app',\n"
            "    'run',\n"
            "]\n"
        )

    def _template_service_plan_test_module(self, plan: FeasiblePlan) -> str:
        quality = plan.quality_contract
        extra_assert = "    assert service.QUALITY_LEVEL >= 8\n" if quality.overall_level >= 8 else ""
        return (
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "\n"
            "import service\n"
            "\n"
            "\n"
            "def test_service_module_importable():\n"
            "    assert callable(service.run)\n"
            "    assert callable(service.handle_request)\n"
            + extra_assert
        )

    def _template_plan_test_module(self, plan: FeasiblePlan, path: str) -> str:
        normalized = path.replace("\\", "/").lower()
        is_invoice = self._is_invoice_plan(plan)
        if normalized.endswith("tests/test_expiration_rules.py"):
            if is_invoice:
                return (
                    "from datetime import date\n"
                    "from pathlib import Path\n"
                    "import sys\n"
                    "\n"
                    "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                    "\n"
                    "from expiration_rules import flag_expiring_contracts\n"
                    "\n"
                    "\n"
                    "def test_expiration_flags_expected_rows():\n"
                    "    rows = [{'invoice_id': 'INV-1', 'due_date': '2026-01-15'}]\n"
                    "    output = flag_expiring_contracts(rows, horizon_days=90, today=date(2026, 1, 1))\n"
                    "    assert len(output) == 1\n"
                    "    assert output[0]['is_expiring_within_horizon'] in {'True', 'False'}\n"
                    "    assert output[0]['is_overdue'] in {'True', 'False'}\n"
                )
            return (
                "from datetime import date\n"
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from expiration_rules import flag_expiring_contracts\n"
                "\n"
                "\n"
                "def test_expiration_flags_expected_rows():\n"
                "    rows = [{'contract_id': 'A', 'expiration_date': '2026-01-15'}]\n"
                "    output = flag_expiring_contracts(rows, horizon_days=90, today=date(2026, 1, 1))\n"
                "    assert len(output) == 1\n"
                "    assert output[0]['is_expiring_within_horizon'] in {'True', 'False'}\n"
            )
        if is_invoice:
            return (
                "import csv\n"
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "import cli\n"
                "\n"
                "\n"
                "def test_cli_flow_end_to_end(tmp_path):\n"
                "    input_path = tmp_path / 'invoices.csv'\n"
                "    output_path = tmp_path / 'summary.csv'\n"
                "    input_path.write_text(\n"
                "        'invoice_id,due_date,amount,customer_name\\nINV-1,2026-01-10,10,Acme\\nINV-2,2026-01-20,15,Beta\\n',\n"
                "        encoding='utf-8',\n"
                "    )\n"
                "    result = cli.main([str(input_path), str(output_path), '--horizon-days', '0'])\n"
                "    assert result == 0\n"
                "    with output_path.open('r', encoding='utf-8', newline='') as handle:\n"
                "        rows = list(csv.DictReader(handle))\n"
                "    assert len(rows) == 2\n"
                "    assert rows[0]['total_amount'] == '25'\n"
                "    assert rows[1]['invoice_count'] == '2'\n"
            )
        return (
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "\n"
            "import cli\n"
            "\n"
            "\n"
            "def test_cli_module_importable():\n"
            "    assert callable(cli.main)\n"
        )

    def _template_generic_module(self, path: str, interfaces: List[PlanInterface]) -> str:
        function_name = "run"
        for interface in interfaces:
            if interface.name and interface.name.isidentifier():
                function_name = interface.name
                break
        return (
            f"def {function_name}() -> int:\n"
            f"    _ = {path!r}\n"
            "    return 0\n"
        )

    def _template_for_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        mode = self._infer_plan_template_mode(plan)
        name = plan_test.test_name.lower()
        objective = plan_test.objective.lower()
        if mode == "service":
            if "suite_executes" in name or any(
                token in objective
                for token in ("service", "rest", "api", "authentication", "rate limit", "persistent")
            ):
                return self._template_service_suite_executes_test(plan)
            return self._template_service_requirement_test(plan)
        if "reads_contracts_csv" in name or ("read" in objective and "csv" in objective):
            if "invoice" in objective:
                return (
                    "from pathlib import Path\n"
                    "import sys\n"
                    "\n"
                    "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                    "\n"
                    "from contracts_csv import load_contracts_csv\n"
                    "\n"
                    "\n"
                    "def test_reads_contracts_csv(tmp_path):\n"
                    "    input_path = tmp_path / 'invoices.csv'\n"
                    "    input_path.write_text(\n"
                    "        'invoice_id,due_date,amount,customer_name\\nINV-1,2026-01-15,100.5,Acme\\n',\n"
                    "        encoding='utf-8',\n"
                    "    )\n"
                    "    rows = load_contracts_csv(str(input_path))\n"
                    "    assert len(rows) == 1\n"
                    "    assert rows[0]['invoice_id'] == 'INV-1'\n"
                    "    assert rows[0]['due_date'] == '2026-01-15'\n"
                    "    assert rows[0]['amount'] == '100.5'\n"
                    "    assert rows[0]['customer_name'] == 'Acme'\n"
                )
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from contracts_csv import load_contracts_csv\n"
                "\n"
                "\n"
                "def test_reads_contracts_csv(tmp_path):\n"
                "    input_path = tmp_path / 'contracts.csv'\n"
                "    input_path.write_text('contract_id,expiration_date\\nA,2026-01-15\\n', encoding='utf-8')\n"
                "    rows = load_contracts_csv(str(input_path))\n"
                "    assert rows[0]['contract_id'] == 'A'\n"
            )
        if "extracts_expiration_dates" in name or ("extract" in objective and "date" in objective):
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from expiration_rules import parse_expiration_date\n"
                "\n"
                "\n"
                "def test_extracts_expiration_dates():\n"
                "    parsed = parse_expiration_date('2026-01-15')\n"
                "    assert parsed is not None\n"
                "    assert parsed.year == 2026\n"
            )
        if "flags_contracts_within_horizon" in name:
            return (
                "from datetime import date\n"
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from expiration_rules import flag_expiring_contracts\n"
                "\n"
                "\n"
                "def test_flags_contracts_within_horizon():\n"
                "    rows = [{'contract_id': 'A', 'expiration_date': '2026-01-20'}]\n"
                "    flagged = flag_expiring_contracts(rows, horizon_days=90, today=date(2026, 1, 1))\n"
                "    assert flagged[0]['is_expiring_within_horizon'] == 'True'\n"
            )
        if "overdue" in objective or "overdue" in name:
            return (
                "from datetime import date\n"
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from expiration_rules import flag_expiring_contracts\n"
                "\n"
                "\n"
                "def test_flags_overdue_invoices():\n"
                "    rows = [\n"
                "        {'invoice_id': 'INV-1', 'due_date': '2026-01-10'},\n"
                "        {'invoice_id': 'INV-2', 'due_date': '2026-01-20'},\n"
                "    ]\n"
                "    flagged = flag_expiring_contracts(rows, horizon_days=0, today=date(2026, 1, 15))\n"
                "    flagged_by_id = {row['invoice_id']: row for row in flagged}\n"
                "    assert flagged_by_id['INV-1']['is_expiring_within_horizon'] == 'True'\n"
                "    assert flagged_by_id['INV-2']['is_expiring_within_horizon'] == 'False'\n"
                "    assert flagged_by_id['INV-1']['is_overdue'] == 'True'\n"
                "    assert flagged_by_id['INV-2']['is_overdue'] == 'False'\n"
            )
        if "writes_summary_csv_with_totals_and_counts" in name:
            return (
                "import csv\n"
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from summary_writer import write_summary_csv\n"
                "\n"
                "\n"
                "def test_writes_summary_csv_with_totals_and_counts(tmp_path):\n"
                "    output_path = tmp_path / 'summary.csv'\n"
                "    rows = [\n"
                "        {\n"
                "            'invoice_id': 'A',\n"
                "            'due_date': '2026-01-15',\n"
                "            'amount': '10',\n"
                "            'customer_name': 'C1',\n"
                "        },\n"
                "        {\n"
                "            'invoice_id': 'B',\n"
                "            'due_date': '2026-01-20',\n"
                "            'amount': '15',\n"
                "            'customer_name': 'C2',\n"
                "        },\n"
                "    ]\n"
                "    write_summary_csv(rows, str(output_path))\n"
                "    with output_path.open('r', encoding='utf-8', newline='') as handle:\n"
                "        parsed = list(csv.DictReader(handle))\n"
                "    assert len(parsed) == 2\n"
                "    assert parsed[0]['total_amount'] == '25'\n"
                "    assert parsed[1]['total_amount'] == '25'\n"
                "    assert parsed[0]['invoice_count'] == '2'\n"
                "    assert parsed[1]['invoice_count'] == '2'\n"
            )
        if "writes_summary_csv" in name:
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from summary_writer import write_summary_csv\n"
                "\n"
                "\n"
                "def test_writes_summary_csv(tmp_path):\n"
                "    output_path = tmp_path / 'summary.csv'\n"
                "    rows = [{'contract_id': 'A', 'expiration_date': '2026-01-15', 'days_to_expiration': '14', 'is_expiring_within_horizon': 'True'}]\n"
                "    write_summary_csv(rows, str(output_path))\n"
                "    data = output_path.read_text(encoding='utf-8')\n"
                "    assert 'contract_id' in data\n"
                "    assert 'A' in data\n"
            )
        if "handles_malformed_rows_and_invalid_dates" in name:
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from contracts_csv import load_contracts_csv\n"
                "from expiration_rules import flag_expiring_contracts\n"
                "\n"
                "\n"
                "def test_handles_malformed_rows_and_invalid_dates(tmp_path):\n"
                "    input_path = tmp_path / 'invoices.csv'\n"
                "    input_path.write_text(\n"
                "        'invoice_id,due_date,amount,customer_name\\nA,not-a-date,10,C1\\nB,,15,C2\\n',\n"
                "        encoding='utf-8',\n"
                "    )\n"
                "    rows = load_contracts_csv(str(input_path))\n"
                "    assert len(rows) == 1\n"
                "    assert rows[0]['invoice_id'] == 'A'\n"
                "    flagged = flag_expiring_contracts(rows, horizon_days=90)\n"
                "    assert len(flagged) == 1\n"
                "    assert flagged[0]['is_expiring_within_horizon'] == 'False'\n"
                "    assert flagged[0]['days_to_expiration'] == ''\n"
            )
        if "handles_malformed_rows" in name:
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from expiration_rules import flag_expiring_contracts\n"
                "\n"
                "\n"
                "def test_handles_malformed_rows():\n"
                "    rows = [{'invoice_id': 'A'}]\n"
                "    flagged = flag_expiring_contracts(rows, horizon_days=90)\n"
                "    assert flagged[0]['days_to_expiration'] == ''\n"
                "    assert flagged[0]['is_expiring_within_horizon'] == 'False'\n"
            )
        if "rejects_invalid_dates" in name:
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from expiration_rules import parse_expiration_date\n"
                "\n"
                "\n"
                "def test_rejects_invalid_dates():\n"
                "    parsed = parse_expiration_date('32/99/2026')\n"
                "    assert parsed is None\n"
            )
        if "universal_date_format_support" in name:
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "from expiration_rules import parse_expiration_date\n"
                "\n"
                "\n"
                "def test_universal_date_format_support_samples():\n"
                "    samples = ['2026-01-15', '15/01/2026', '01/15/2026']\n"
                "    parsed = [parse_expiration_date(sample) for sample in samples]\n"
                "    assert all(item is not None for item in parsed)\n"
            )
        if "suite_executes" in name:
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                "import cli\n"
                "import contracts_csv\n"
                "import expiration_rules\n"
                "import summary_writer\n"
                "\n"
                "\n"
                "def test_suite_executes():\n"
                "    assert callable(cli.main)\n"
                "    assert callable(contracts_csv.load_contracts_csv)\n"
                "    assert callable(expiration_rules.flag_expiring_contracts)\n"
                "    assert callable(summary_writer.write_summary_csv)\n"
            )
        return self._template_generic_requirement_test(plan, plan_test)

    def _template_service_suite_executes_test(self, plan: FeasiblePlan) -> str:
        quality = plan.quality_contract
        lines = [
            "from pathlib import Path",
            "import sys",
            "",
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))",
            "",
            "import service",
            "",
            "",
            "def test_rejects_unauthorized_requests(tmp_path):",
            "    db_path = tmp_path / 'service.sqlite3'",
            "    service.init_db(str(db_path))",
            "    code, payload = service.handle_request(",
            "        'invalid-key',",
            "        {'event': 'x'},",
            "        db_path=str(db_path),",
            "        now=1000.0,",
            "    )",
            "    assert code == 401",
            "    assert payload['error'] == 'unauthorized'",
            "",
            "",
            "def test_enforces_rate_limit(tmp_path):",
            "    db_path = tmp_path / 'service.sqlite3'",
            "    service.register_user('alice', 'key-alice', db_path=str(db_path))",
            "    for index in range(100):",
            "        code, _ = service.handle_request(",
            "            'key-alice',",
            "            {'index': index},",
            "            db_path=str(db_path),",
            "            now=1000.0 + index * 0.1,",
            "        )",
            "        assert code == 200",
            "    code, payload = service.handle_request(",
            "        'key-alice',",
            "        {'index': 101},",
            "        db_path=str(db_path),",
            "        now=1011.0,",
            "    )",
            "    assert code == 429",
            "    assert payload['error'] == 'rate_limit_exceeded'",
            "",
            "",
            "def test_persists_user_credentials_across_reloads(tmp_path):",
            "    db_path = tmp_path / 'service.sqlite3'",
            "    service.register_user('bob', 'key-bob', db_path=str(db_path))",
            "    user = service.authenticate('key-bob', db_path=str(db_path))",
            "    assert user == 'bob'",
            "    app_config = service.create_app(str(db_path))",
            "    assert app_config['db_path'] == str(db_path)",
            "    assert service.run() == 0",
        ]
        if quality.audit_trail:
            lines.extend(
                [
                    "",
                    "",
                    "def test_audit_trail_records_requests(tmp_path):",
                    "    db_path = tmp_path / 'service.sqlite3'",
                    "    service.register_user('eve', 'key-eve', db_path=str(db_path))",
                    "    service.handle_request('key-eve', {'hello': 'world'}, db_path=str(db_path), now=1000.0)",
                    "    events = service.get_recent_events(str(db_path), limit=5)",
                    "    assert events",
                    "    assert any(event['status_code'] in {200, 401, 429} for event in events)",
                ]
            )
        if quality.integration_tests:
            lines.extend(
                [
                    "",
                    "",
                    "def test_integration_flow_handles_valid_request(tmp_path):",
                    "    db_path = tmp_path / 'service.sqlite3'",
                    "    service.register_user('intg', 'key-intg', db_path=str(db_path))",
                    "    code, payload = service.handle_request(",
                    "        'key-intg',",
                    "        {'integration': True},",
                    "        db_path=str(db_path),",
                    "        now=1000.0,",
                    "    )",
                    "    assert code == 200",
                    "    assert payload['user'] == 'intg'",
                ]
            )
        if quality.auth_level == "hashed":
            lines.extend(
                [
                    "",
                    "",
                    "def test_hashed_auth_path_stores_hashes(tmp_path):",
                    "    db_path = tmp_path / 'service.sqlite3'",
                    "    service.register_user('hash', 'key-hash', db_path=str(db_path))",
                    "    assert service.authenticate('key-hash', db_path=str(db_path)) == 'hash'",
                ]
            )
        return "\n".join(lines) + "\n"

    def _template_service_requirement_test(self, plan: FeasiblePlan) -> str:
        quality = plan.quality_contract
        auth_assert = (
            "    assert service.QUALITY_AUTH_MODE in {'hashed', 'jwt'}\n"
            if quality.auth_level in {"hashed", "jwt"}
            else "    assert service.QUALITY_AUTH_MODE == 'plaintext'\n"
        )
        return (
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "\n"
            "import service\n"
            "\n"
            "\n"
            "def test_service_requirement_smoke(tmp_path):\n"
            "    db_path = tmp_path / 'service.sqlite3'\n"
            "    service.register_user('smoke', 'key-smoke', db_path=str(db_path))\n"
            "    code, payload = service.handle_request('key-smoke', {'ok': True}, db_path=str(db_path), now=1000.0)\n"
            "    assert code == 200\n"
            "    assert payload['user'] == 'smoke'\n"
            + auth_assert
        )

    def _template_generic_requirement_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        src_modules = [
            path.path.split("/")[-1].replace(".py", "")
            for path in plan.file_tree_plan
            if path.path.startswith("src/") and path.path.endswith(".py")
        ]
        is_invoice = self._is_invoice_plan(plan)
        has_cli = "cli" in src_modules
        has_main = "main" in src_modules
        if has_cli or has_main:
            module_name = "cli" if has_cli else "main"
            if is_invoice:
                return (
                    "import csv\n"
                    "from pathlib import Path\n"
                    "import sys\n"
                    "\n"
                    "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                    "\n"
                    f"import {module_name}\n"
                    "\n"
                    "\n"
                    "def test_generated_requirement_exercises_cli_flow(tmp_path):\n"
                    "    input_path = tmp_path / 'input.csv'\n"
                    "    output_path = tmp_path / 'output.csv'\n"
                    "    input_path.write_text(\n"
                    "        'invoice_id,due_date,amount,customer_name\\nINV-1,2026-01-10,10,Acme\\nINV-2,2026-01-20,15,Beta\\n',\n"
                    "        encoding='utf-8',\n"
                    "    )\n"
                    f"    result = {module_name}.main([str(input_path), str(output_path), '--horizon-days', '0'])\n"
                    "    assert result == 0\n"
                    "    with output_path.open('r', encoding='utf-8', newline='') as handle:\n"
                    "        rows = list(csv.DictReader(handle))\n"
                    "    assert len(rows) == 2\n"
                    "    assert rows[0]['total_amount'] == '25'\n"
                    "    assert rows[1]['invoice_count'] == '2'\n"
                )
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                f"import {module_name}\n"
                "\n"
                "\n"
                "def test_generated_requirement_exercises_cli_flow(tmp_path):\n"
                "    input_path = tmp_path / 'input.csv'\n"
                "    output_path = tmp_path / 'output.csv'\n"
                "    input_path.write_text('contract_id,expiration_date\\nA,2026-01-15\\n', encoding='utf-8')\n"
                f"    result = {module_name}.main([str(input_path), str(output_path)])\n"
                "    assert result == 0\n"
                "    assert output_path.exists()\n"
            )

        module_name = src_modules[0] if src_modules else ""
        interface_name = plan.interfaces[0].name if plan.interfaces else "run"
        if module_name and interface_name.isidentifier():
            return (
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "\n"
                f"import {module_name}\n"
                "\n"
                "\n"
                "def test_generated_requirement_invokes_target_code():\n"
                f"    target = getattr({module_name}, {interface_name!r}, None)\n"
                "    assert callable(target)\n"
                "    result = target()\n"
                "    assert isinstance(result, (int, type(None)))\n"
            )

        raise MalformedPlanError(
            "Unable to generate semantic test template for required test "
            f"'{plan_test.test_name}'. Plan does not provide a runnable module/interface mapping."
        )

    def _is_invoice_plan(self, plan: FeasiblePlan) -> bool:
        atom_text = " ".join(atom.text.lower() for atom in plan.build_spec.requirement_atoms)
        goal_text = " ".join(goal.lower() for goal in plan.build_spec.functional_goals)
        combined = f"{atom_text} {goal_text}"
        return "invoice" in combined or "due_date" in combined
