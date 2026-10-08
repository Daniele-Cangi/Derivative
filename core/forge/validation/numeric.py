import json
from pathlib import Path
from typing import Any

from core.forge.contracts import BuildSpec, FeasiblePlan
from core.forge.execution import ProcessExecutor, SandboxProcessRequest
from core.forge.expiration_contract import expiration_horizon_contract


class NumericContractChecker:
    """Execute validator-owned boundary probes independently of candidate tests."""

    def __init__(self, executor: ProcessExecutor, timeout_seconds: int):
        self.executor = executor
        self.timeout_seconds = timeout_seconds

    def check(
        self, spec: BuildSpec, plan: FeasiblePlan, workspace: Path,
    ) -> tuple[list[str], list[str], dict[str, Any]]:
        contract = expiration_horizon_contract(spec)
        if contract is None:
            return [], [], {"contracts": []}
        horizon = contract.get("threshold_days")
        interface = next((item for item in plan.interfaces if item.interface_type == "cli_entrypoint"), None)
        module_name = interface.module_path if interface is not None else ""
        if not module_name and plan.implementation_blueprint.entrypoint_path == "src/cli.py":
            module_name = "cli"
        if type(horizon) is not int or interface is None or not module_name:
            return ["Numeric expiration obligation lacks a derivable probe."], ["numeric_contract_unproven"], {
                "contracts": [{"contract": contract, "passed": False, "reason": "probe_unavailable"}],
            }
        # Both the function's default and the public CLI's default must implement
        # the compiled bound; passing --horizon-days would hide a wrong default.
        offsets = sorted({-31, -1, 0, 1, horizon // 2, horizon - 31, horizon - 1, horizon, horizon + 1})
        script = self._expiration_probe(offsets, module_name, interface.name)
        result = self.executor.run(SandboxProcessRequest(
            command=["python", "-B", "-c", script], workspace=workspace,
            timeout_seconds=self.timeout_seconds,
            environment={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
        ))
        try:
            observed = json.loads(result.stdout.strip())
        except (ValueError, TypeError):
            observed = None
        expected = {str(offset): str(offset < horizon) for offset in offsets}
        passed = (
            result.returncode == 0
            and isinstance(observed, dict)
            and observed.get("function_flags") == expected
            and observed.get("cli_flags") == expected
            and observed.get("function_row_count") == len(offsets)
            and observed.get("cli_row_count") == len(offsets)
            and observed.get("cli_status") == 0
        )
        evidence = {"contracts": [{
            "contract": contract, "boundary_offsets": [horizon - 1, horizon, horizon + 1],
            "probe_offsets": offsets,
            "correlation_key": "contract_id",
            "expected_flags": expected, "observed": observed, "passed": passed,
            "returncode": result.returncode, "stderr": result.stderr,
            "backend": result.backend, "isolation": result.isolation,
            "launch_error": result.launch_error,
        }]}
        if passed:
            return [], [], evidence
        signature = "numeric_contract_unproven" if observed is None else "numeric_contract_mismatch"
        return [f"Expiration horizon is not independently evidenced: {contract}."], [
            signature, "semantic_content_mismatch",
        ], evidence

    @staticmethod
    def _expiration_probe(offsets: list[int], module_name: str, symbol: str) -> str:
        return (
            "import csv, importlib, json\n"
            "from datetime import date, timedelta\n"
            "from pathlib import Path\n"
            "rules = importlib.import_module('expiration_rules')\n"
            f"entrypoint = getattr(importlib.import_module({module_name!r}), {symbol!r})\n"
            f"offsets = {offsets!r}\n"
            "def rows_for(today):\n"
            "    return [{'contract_id': str(n), 'expiration_date': (today + timedelta(days=n)).isoformat()} for n in offsets]\n"
            "rows = rows_for(date(2026, 1, 1))\n"
            "result = rules.flag_expiring_contracts(rows, today=date(2026, 1, 1))\n"
            "function_flags = {row['contract_id']: row['is_expiring_within_horizon'] for row in result}\n"
            "input_path = Path('.forge_numeric_contract_input.csv')\n"
            "output_path = Path('.forge_numeric_contract_output.csv')\n"
            "with input_path.open('w', encoding='utf-8', newline='') as handle:\n"
            "    writer = csv.DictWriter(handle, fieldnames=['contract_id', 'expiration_date'])\n"
            "    writer.writeheader()\n"
            "    writer.writerows(rows_for(date.today()))\n"
            "status = entrypoint([str(input_path), str(output_path)])\n"
            "with output_path.open(encoding='utf-8', newline='') as handle:\n"
            "    cli_rows = list(csv.DictReader(handle))\n"
            "cli_flags = {row['contract_id']: row['is_expiring_within_horizon'] for row in cli_rows}\n"
            "print(json.dumps({'function_flags': function_flags, 'cli_flags': cli_flags, 'cli_status': status, "
            "'function_row_count': len(result), 'cli_row_count': len(cli_rows)}))\n"
        )
