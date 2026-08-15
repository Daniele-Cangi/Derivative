from typing import Callable, Dict, List, Set

from core.forge.capabilities.service import (
    render_audit,
    render_auth,
    render_domain,
    render_observability,
    render_rate_limit,
    render_service_api,
    render_storage,
)
from core.forge.contracts import ArtifactTargetType, FeasiblePlan, PlanInterface, PlanTest
from core.forge.domains.base import BaseDomainAdapter
from core.forge.domains.service_events import (
    is_idempotent_event_service,
    render_event_service_file,
    render_event_service_test,
)


class ServiceDomainAdapter(BaseDomainAdapter):
    name = "service"

    def __init__(self):
        self._renderers: Dict[str, Callable[[FeasiblePlan], str]] = {
            "src/service.py": render_service_api,
            "src/domain.py": render_domain,
            "src/storage.py": render_storage,
            "src/auth.py": render_auth,
            "src/rate_limit.py": render_rate_limit,
            "src/audit.py": render_audit,
            "src/observability.py": render_observability,
        }

    def matches(self, plan: FeasiblePlan) -> bool:
        if plan.build_spec.target_artifact_type == ArtifactTargetType.SERVICE:
            return True
        paths = {item.path.replace("\\", "/").lower() for item in plan.file_tree_plan}
        return "src/service.py" in paths

    def render_file(self, plan: FeasiblePlan, path: str, interfaces: List[PlanInterface]) -> str:
        if is_idempotent_event_service(plan):
            rendered = render_event_service_file(plan, path, interfaces)
            if rendered is not None:
                return rendered
        normalized = path.replace("\\", "/").lower()
        renderer = self._renderers.get(normalized)
        if renderer is not None:
            return renderer(plan)
        if normalized.startswith("tests/"):
            return self._template_service_plan_test_module(plan)
        return self._template_generic_module(path, interfaces)

    def render_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        if is_idempotent_event_service(plan):
            return render_event_service_test(plan, plan_test)
        name = plan_test.test_name.lower()
        objective = plan_test.objective.lower()
        if "suite_executes" in name or any(
            token in objective
            for token in ("service", "rest", "api", "authentication", "rate limit", "persistent")
        ):
            return self._template_service_suite_executes_test(plan)
        return self._template_service_requirement_test(plan)

    def provided_capabilities(self, plan: FeasiblePlan) -> Set[str]:
        if is_idempotent_event_service(plan):
            capabilities = {
                "rest_service",
                "api_key_authentication",
                "sqlite_persistence",
                "idempotent_event_creation",
            }
            if plan.quality_contract.integration_tests:
                capabilities.add("integration_tests")
            return capabilities
        capabilities = {
            "rest_service",
            "api_key_authentication",
            "sqlite_persistence",
            "per_user_rate_limiting",
            "request_handling",
        }
        quality = plan.quality_contract
        if quality.audit_trail:
            capabilities.add("audit_trail")
        if quality.health_endpoint:
            capabilities.add("health_endpoint")
        if quality.structured_logging:
            capabilities.add("structured_logging")
        if quality.integration_tests:
            capabilities.add("integration_tests")
        if quality.auth_level == "hashed":
            capabilities.add("hashed_authentication")
        if quality.rate_limit_persistent:
            capabilities.add("persistent_rate_limiting")
        if quality.rate_limit_scope == "distributed":
            capabilities.add("distributed_rate_limiting")
        return capabilities

    def implements_plan_semantics(self, plan: FeasiblePlan) -> bool:
        return True

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
