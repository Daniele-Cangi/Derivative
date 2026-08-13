import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.forge.contracts import CodeArtifact, FeasiblePlan


def preflight_has_source_failure(preflight: dict[str, Any]) -> bool:
    if preflight.get("source_failed_paths"):
        return True
    output = "\n".join(
        str(preflight.get(field, ""))
        for field in ("stdout", "stderr")
    )
    return bool(
        re.search(
            r"(?:^|[\s'\"])(?:[A-Za-z]:)?[^\n]*?[\\/]src[\\/][^\s:'\"]+\.py(?::\d+)?",
            output,
            re.IGNORECASE | re.MULTILINE,
        )
        or re.search(r"\bsrc[\\/][^\s:'\"]+\.py(?::\d+)?", output, re.IGNORECASE)
    )


def preflight_failed_paths(
    preflight: dict[str, Any],
    *,
    prefix: str | None = None,
) -> list[str]:
    paths = [
        str(path).replace("\\", "/")
        for path in preflight.get("failed_paths", [])
        if str(path).strip()
    ]
    if not paths:
        output = "\n".join(
            str(preflight.get(field, ""))
            for field in ("stdout", "stderr")
        )
        paths.extend(
            match.replace("\\", "/")
            for match in re.findall(
                r"(?:FAILED|ERROR)\s+([^\s:]+\.py)(?:::|\s|$)",
                output,
                re.IGNORECASE,
            )
        )
        paths.extend(
            match.replace("\\", "/")
            for match in re.findall(
                r"\b((?:src|tests)[\\/][^\s:'\"]+\.py)(?::\d+)?",
                output,
                re.IGNORECASE,
            )
        )
    unique: list[str] = []
    for path in paths:
        normalized = path.lstrip("./")
        if prefix and not normalized.startswith(prefix):
            continue
        if normalized not in unique:
            unique.append(normalized)
    return unique


def source_api_contracts(source_files: dict[str, str]) -> dict[str, Any]:
    contracts: dict[str, Any] = {}
    for path, content in sorted(source_files.items()):
        contract: dict[str, Any] = {
            "functions": [],
            "classes": [],
            "cli_options": sorted(
                set(re.findall(r"add_argument\(\s*['\"](--?[^'\"]+)", content))
            ),
        }
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            contract["parse_error"] = f"line {exc.lineno}: {exc.msg}"
            contracts[path] = contract
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                positional = [
                    argument.arg
                    for argument in (*node.args.posonlyargs, *node.args.args)
                ]
                keyword_only = [argument.arg for argument in node.args.kwonlyargs]
                contract["functions"].append(
                    {
                        "name": node.name,
                        "decorators": [
                            ast.unparse(decorator)
                            for decorator in node.decorator_list
                        ],
                        "positional_parameters": positional,
                        "keyword_only_parameters": keyword_only,
                        "vararg": node.args.vararg.arg if node.args.vararg else None,
                        "kwarg": node.args.kwarg.arg if node.args.kwarg else None,
                    }
                )
            elif isinstance(node, ast.ClassDef):
                contract["classes"].append(node.name)
        contracts[path] = contract
    return contracts


def test_generation_contracts(
    test_paths: list[str],
    plan: FeasiblePlan,
    artifact: CodeArtifact,
) -> dict[str, Any]:
    tests_by_name = {test.test_name: test for test in plan.required_tests}
    atoms_by_id = {
        atom.requirement_id: atom
        for atom in plan.build_spec.requirement_atoms
    }
    requirement_corpus = " ".join(
        atom.text.lower()
        for atom in plan.build_spec.requirement_atoms
        if atom.category != "ambiguity"
    )
    forbidden_assumptions: list[str] = []
    if not any(token in requirement_corpus for token in ("sqlite", "database", "persist")):
        forbidden_assumptions.extend(
            [
                "SQLite or database persistence assertions",
                "records, audit, or schema table assertions",
            ]
        )
    if (
        any(token in requirement_corpus for token in ("json lines", "jsonl"))
        and "watched directory" not in requirement_corpus
    ):
        forbidden_assumptions.extend(
            [
                "watched-directory polling assertions",
                "CSV input fixtures; summary CSV output remains allowed",
            ]
        )
    contracts: dict[str, Any] = {}
    for path in sorted(test_paths):
        test_name = Path(path).stem
        planned = tests_by_name.get(test_name)
        traceability = artifact.traceability.get(path, [])
        requirement_ids = list(planned.requirement_ids) if planned else []
        for entry in traceability:
            if entry.startswith("requirement:"):
                requirement_id = entry.split(":", 1)[1]
                if requirement_id not in requirement_ids:
                    requirement_ids.append(requirement_id)
        contracts[path] = {
            "test_name": test_name,
            "objective": (
                planned.objective
                if planned
                else "Preserve mapped requirement behavior."
            ),
            "test_type": planned.test_type if planned else "behavioral",
            "acceptance_criterion_ids": (
                list(planned.acceptance_criterion_ids) if planned else []
            ),
            "requirements": [
                {
                    "id": requirement_id,
                    "text": atoms_by_id[requirement_id].text,
                    "evidence_terms": list(
                        atoms_by_id[requirement_id].evidence_terms
                    ),
                }
                for requirement_id in requirement_ids
                if requirement_id in atoms_by_id
            ],
            "traceability": list(traceability),
            "declared_plan_interfaces": [
                {
                    "name": interface.name,
                    "type": interface.interface_type,
                    "signature": interface.signature,
                }
                for interface in plan.interfaces
            ],
            "forbidden_unrequested_behaviors": list(forbidden_assumptions),
        }
    return contracts


def run_test_preflight(
    candidate_files: dict[str, str],
    test_paths: list[str],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "phase": "materialization",
        "ran": False,
        "passed": False,
        "returncode": None,
        "tests": list(test_paths),
        "stdout": "",
        "stderr": "",
        "failed_paths": [],
        "source_failed_paths": [],
        "test_failed_paths": [],
        "failures": [],
    }
    try:
        with tempfile.TemporaryDirectory(
            prefix="forge_repair_preflight_",
            ignore_cleanup_errors=True,
        ) as tmp_dir:
            workspace = Path(tmp_dir)
            workspace_root = workspace.resolve()
            for path, content in candidate_files.items():
                target = (workspace / path).resolve()
                if target != workspace_root and workspace_root not in target.parents:
                    raise ValueError(
                        f"Artifact path escapes preflight workspace: {path}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            syntax_failures: list[dict[str, Any]] = []
            for path, content in sorted(candidate_files.items()):
                if not path.endswith(".py"):
                    continue
                try:
                    ast.parse(content, filename=path)
                except SyntaxError as exc:
                    syntax_failures.append(
                        {
                            "path": path.replace("\\", "/"),
                            "kind": "syntax_error",
                            "line": exc.lineno,
                            "message": exc.msg,
                        }
                    )
            if syntax_failures:
                failed_paths = [failure["path"] for failure in syntax_failures]
                result.update(
                    {
                        "phase": "syntax",
                        "failures": syntax_failures,
                        "failed_paths": failed_paths,
                        "source_failed_paths": [
                            path for path in failed_paths if path.startswith("src/")
                        ],
                        "test_failed_paths": [
                            path for path in failed_paths if path.startswith("tests/")
                        ],
                    }
                )
                return result

            import_script = (
                "import importlib, pathlib, sys\n"
                "root = pathlib.Path(sys.argv[1]).resolve()\n"
                "sys.path[:0] = [str(root), str(root / 'src')]\n"
                "path = pathlib.PurePosixPath(sys.argv[2])\n"
                "module_name = '.'.join(path.with_suffix('').parts)\n"
                "importlib.import_module(module_name)\n"
            )
            import_failures: list[dict[str, Any]] = []
            for path in sorted(
                path
                for path in candidate_files
                if path.startswith("src/") and path.endswith(".py")
            ):
                imported = subprocess.run(
                    [sys.executable, "-B", "-c", import_script, str(workspace), path],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    check=False,
                )
                if imported.returncode != 0:
                    import_failures.append(
                        {
                            "path": path.replace("\\", "/"),
                            "kind": "import_failure",
                            "line": None,
                            "message": (imported.stderr or imported.stdout)[-4000:],
                        }
                    )
            if import_failures:
                failed_paths = [failure["path"] for failure in import_failures]
                result.update(
                    {
                        "phase": "import",
                        "returncode": 1,
                        "stderr": "\n".join(
                            failure["message"] for failure in import_failures
                        )[-12000:],
                        "failures": import_failures,
                        "failed_paths": failed_paths,
                        "source_failed_paths": failed_paths,
                    }
                )
                return result

            command = [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "-q",
                *test_paths,
                "-p",
                "no:cacheprovider",
                f"--basetemp={workspace / '.pytest_tmp'}",
            ]
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            completed = subprocess.run(
                command,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
                check=False,
            )
            result.update(
                {
                    "phase": "tests",
                    "ran": True,
                    "passed": completed.returncode == 0,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-12000:],
                    "stderr": completed.stderr[-12000:],
                }
            )
            if completed.returncode != 0:
                failed_paths = preflight_failed_paths(result)
                result.update(
                    {
                        "failed_paths": failed_paths,
                        "source_failed_paths": [
                            path for path in failed_paths if path.startswith("src/")
                        ],
                        "test_failed_paths": [
                            path for path in failed_paths if path.startswith("tests/")
                        ],
                        "failures": [
                            {
                                "path": path,
                                "kind": "test_failure",
                                "line": None,
                                "message": "pytest reported a failure for this file",
                            }
                            for path in failed_paths
                        ],
                    }
                )
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "phase": result.get("phase", "tests"),
                "ran": True,
                "error_type": "TimeoutExpired",
                "stdout": str(exc.stdout or "")[-12000:],
                "stderr": str(exc.stderr or "")[-12000:],
            }
        )
    except Exception as exc:
        result.update(
            {
                "phase": result.get("phase", "materialization"),
                "error_type": type(exc).__name__,
                "stderr": str(exc),
            }
        )
    return result
