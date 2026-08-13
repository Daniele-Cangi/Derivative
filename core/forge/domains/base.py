from typing import List, Set

from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest


class DomainAdapterError(Exception):
    """Raised when a domain adapter cannot expand a plan deterministically."""


class BaseDomainAdapter:
    name = "base"

    def matches(self, plan: FeasiblePlan) -> bool:
        return False

    def render_file(self, plan: FeasiblePlan, path: str, interfaces: List[PlanInterface]) -> str:
        raise NotImplementedError

    def render_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        raise NotImplementedError

    def provided_capabilities(self, plan: FeasiblePlan) -> Set[str]:
        return set()

    def _entrypoint_name(self, plan: FeasiblePlan, default: str = "run") -> str:
        for interface in plan.interfaces:
            if interface.interface_type in {"entrypoint", "cli_entrypoint"} and interface.name.isidentifier():
                return interface.name
        return default

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

    def _template_generic_requirement_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        src_modules = [
            path.path.split("/")[-1].replace(".py", "")
            for path in plan.file_tree_plan
            if path.path.startswith("src/") and path.path.endswith(".py")
        ]
        is_invoice = self._is_invoice_plan(plan)
        has_cli = "cli" in src_modules
        has_main = "main" in src_modules and any(
            interface.interface_type == "cli_entrypoint"
            or interface.name == "main"
            for interface in plan.interfaces
        )
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

        raise DomainAdapterError(
            "Unable to generate semantic test template for required test "
            f"'{plan_test.test_name}'. Plan does not provide a runnable module/interface mapping."
        )

    def _template_generic_planned_test(self, plan: FeasiblePlan) -> str:
        source_modules = [
            item.path.split("/")[-1].removesuffix(".py")
            for item in plan.file_tree_plan
            if item.path.startswith("src/") and item.path.endswith(".py")
        ]
        if not source_modules:
            raise DomainAdapterError("Generic planned test requires at least one Python source module.")
        module_name = source_modules[0]
        entrypoint_name = self._entrypoint_name(plan)
        return (
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "\n"
            f"import {module_name}\n"
            "\n"
            "\n"
            "def test_planned_entrypoint_smoke(tmp_path, monkeypatch):\n"
            "    monkeypatch.chdir(tmp_path)\n"
            f"    result = {module_name}.{entrypoint_name}()\n"
            "    assert result == 0\n"
        )

    def _is_invoice_plan(self, plan: FeasiblePlan) -> bool:
        atom_text = " ".join(atom.text.lower() for atom in plan.build_spec.requirement_atoms)
        goal_text = " ".join(goal.lower() for goal in plan.build_spec.functional_goals)
        combined = f"{atom_text} {goal_text}"
        return "invoice" in combined or "due_date" in combined



class GenericDomainAdapter(BaseDomainAdapter):
    name = "generic"

    def matches(self, plan: FeasiblePlan) -> bool:
        return True

    def render_file(self, plan: FeasiblePlan, path: str, interfaces: List[PlanInterface]) -> str:
        if path.replace("\\", "/").lower().startswith("tests/"):
            return self._template_generic_planned_test(plan)
        return self._template_generic_module(path, interfaces)

    def render_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        return self._template_generic_requirement_test(plan, plan_test)

    def provided_capabilities(self, plan: FeasiblePlan) -> Set[str]:
        return {"python_module", "planned_entrypoint"}
