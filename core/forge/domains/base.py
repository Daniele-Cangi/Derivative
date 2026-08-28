import ast
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

    def implements_plan_semantics(self, plan: FeasiblePlan) -> bool:
        return False

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

    def _template_plan_contract_module(
        self,
        plan: FeasiblePlan,
        path: str,
        interfaces: List[PlanInterface],
    ) -> str:
        interface = self._interface_for_path(plan, path, interfaces)
        function_name = interface.name if interface is not None else "run"
        signature = interface.signature.strip() if interface is not None else ""
        if not self._valid_function_signature(function_name, signature):
            signature = f"{function_name}() -> int"
        return (
            "from __future__ import annotations\n"
            "\n"
            "\n"
            f"def {signature}:\n"
            "    raise NotImplementedError(\n"
            f"        'Uncompiled plan contract for {plan.plan_id}: {path}'\n"
            "    )\n"
        )

    def _template_generic_requirement_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        return self._template_contract_test(
            plan,
            requirement_ids=plan_test.requirement_ids,
        )

    def _template_generic_planned_test(self, plan: FeasiblePlan, path: str) -> str:
        plan_file = next(
            (
                item
                for item in plan.file_tree_plan
                if item.path.replace("\\", "/") == path.replace("\\", "/")
            ),
            None,
        )
        requirement_ids = plan_file.source_requirement_refs if plan_file is not None else []
        return self._template_contract_test(plan, requirement_ids=requirement_ids)

    def _template_contract_test(
        self,
        plan: FeasiblePlan,
        requirement_ids: List[str],
    ) -> str:
        interface = self._interface_for_path(
            plan,
            plan.implementation_blueprint.entrypoint_path,
            plan.interfaces,
        )
        if interface is None or not interface.name.isidentifier():
            raise DomainAdapterError(
                "Unable to generate a contract-bound test scaffold without a declared entrypoint."
            )
        module_name = interface.module_path or self._module_name_from_path(
            plan.implementation_blueprint.entrypoint_path
        )
        source_modules = [
            self._module_name_from_path(item.path)
            for item in plan.file_tree_plan
            if item.path.replace("\\", "/").startswith("src/")
            and item.path.lower().endswith(".py")
        ]
        if not module_name and len(source_modules) == 1:
            module_name = source_modules[0]
        if not module_name:
            raise DomainAdapterError(
                "Unable to generate semantic test template without a declared module."
            )
        module_path = f"src/{module_name.replace('.', '/')}.py"
        planned_paths = {
            item.path.replace("\\", "/")
            for item in plan.file_tree_plan
        }
        if module_path not in planned_paths:
            raise DomainAdapterError(
                "Unable to generate semantic test template without a planned source module."
            )
        atoms_by_id = {
            atom.requirement_id: atom.text
            for atom in plan.build_spec.requirement_atoms
        }
        normalized_ids = [
            requirement_id
            for requirement_id in requirement_ids
            if requirement_id in atoms_by_id
        ]
        if not normalized_ids:
            normalized_ids = [atom.requirement_id for atom in plan.build_spec.requirement_atoms]
        requirement_text = tuple(atoms_by_id[item] for item in normalized_ids)
        return (
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "\n"
            f"import {module_name} as target_module\n"
            "\n"
            f"REQUIREMENT_IDS = {tuple(normalized_ids)!r}\n"
            f"REQUIREMENT_TEXT = {requirement_text!r}\n"
            "\n"
            "\n"
            "def test_requirement_contract_requires_candidate_implementation():\n"
            f"    target = getattr(target_module, {interface.name!r}, None)\n"
            "    assert callable(target)\n"
            "    raise AssertionError(\n"
            "        'Candidate compiler must replace contract scaffold: ' + ','.join(REQUIREMENT_IDS)\n"
            "    )\n"
        )

    def _interface_for_path(
        self,
        plan: FeasiblePlan,
        path: str,
        interfaces: List[PlanInterface],
    ) -> PlanInterface | None:
        normalized_path = path.replace("\\", "/").removeprefix("src/").removesuffix(".py")
        normalized_path = normalized_path.replace("/", ".")
        for interface in interfaces:
            if interface.module_path and interface.module_path == normalized_path:
                return interface
        entrypoint_path = plan.implementation_blueprint.entrypoint_path.replace("\\", "/")
        if path.replace("\\", "/") == entrypoint_path:
            for interface in interfaces:
                if interface.interface_type in {"entrypoint", "cli_entrypoint"}:
                    return interface
        return next(
            (
                interface
                for interface in interfaces
                if interface.interface_type in {"entrypoint", "cli_entrypoint"}
            ),
            interfaces[0] if interfaces else None,
        )

    @staticmethod
    def _valid_function_signature(function_name: str, signature: str) -> bool:
        if not signature.startswith(f"{function_name}("):
            return False
        try:
            ast.parse(f"def {signature}:\n    pass\n")
        except SyntaxError:
            return False
        return True

    @staticmethod
    def _module_name_from_path(path: str) -> str:
        return path.replace("\\", "/").removeprefix("src/").removesuffix(".py").replace("/", ".")

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
            return self._template_generic_planned_test(plan, path)
        return self._template_generic_module(path, interfaces)

    def render_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        return self._template_generic_requirement_test(plan, plan_test)

    def provided_capabilities(self, plan: FeasiblePlan) -> Set[str]:
        return {"python_module", "planned_entrypoint"}
