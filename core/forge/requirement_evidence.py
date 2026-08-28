from typing import Callable, Iterable, Mapping

from core.forge.test_evidence import analyze_test_functions


def requirement_assertion_evidence(
    requirement_terms: Mapping[str, Iterable[str]],
    requirement_test_paths: Mapping[str, Iterable[str]],
    file_contents: Mapping[str, str],
    target_names: Iterable[str] = (),
    target_modules: Iterable[str] = (),
    term_matcher: Callable[[str, str], bool] | None = None,
) -> dict[str, dict[str, object]]:
    matcher = term_matcher or _default_term_matcher
    expected_target_names = tuple(target_names)
    expected_target_modules = tuple(target_modules)
    report: dict[str, dict[str, object]] = {}

    for requirement_id in sorted(requirement_terms):
        terms = list(dict.fromkeys(str(term) for term in requirement_terms[requirement_id]))
        mapped_paths = sorted(set(requirement_test_paths.get(requirement_id, ())))
        existing_paths = [path for path in mapped_paths if path in file_contents]
        covered_terms: set[str] = set()
        assertions: list[dict[str, object]] = []
        causal_functions: list[dict[str, object]] = []

        for path in existing_paths:
            for function in analyze_test_functions(
                file_contents[path],
                target_names=expected_target_names,
                target_modules=expected_target_modules,
            ):
                if not function["semantic"]:
                    continue
                function_source = str(function["source"])
                matched_terms = [
                    term
                    for term in terms
                    if matcher(term, function_source.lower())
                    or _target_contract_term_covered(
                        term,
                        function,
                        expected_target_names,
                        expected_target_modules,
                    )
                ]
                causal_functions.append(
                    {
                        "path": path,
                        "function": function["function"],
                        "matched_terms": matched_terms,
                    }
                )
                if terms and not matched_terms:
                    continue
                covered_terms.update(matched_terms)
                for assertion in function["assertions"]:
                    assertions.append(
                        {
                            "path": path,
                            "function": function["function"],
                            "line": assertion["line"],
                            "kind": assertion["kind"],
                            "expression": assertion["expression"],
                            "evidence_terms": list(matched_terms),
                        }
                    )

        missing_terms = [term for term in terms if term not in covered_terms]
        passed = bool(assertions) and not missing_terms
        report[requirement_id] = {
            "mapped_test_paths": mapped_paths,
            "existing_test_paths": existing_paths,
            "required_terms": terms,
            "covered_terms": sorted(covered_terms),
            "missing_terms": missing_terms,
            "causal_functions": causal_functions,
            "assertions": assertions,
            "passed": passed,
            "failure_reason": _failure_reason(
                existing_paths,
                causal_functions,
                missing_terms,
                assertions,
            ),
        }
    return report


def _target_contract_term_covered(
    term: str,
    function: Mapping[str, object],
    target_names: tuple[str, ...],
    target_modules: tuple[str, ...],
) -> bool:
    if not function.get("target_invoked", False):
        return False
    normalized = term.lower().replace("-", "_").replace(" ", "_")
    normalized_modules = {
        module.lower().replace("-", "_").replace(" ", "_")
        for module in target_modules
    }
    if normalized in normalized_modules:
        return True
    return normalized in {"cli_entrypoint", "cli_flow"} and "main" in target_names


def _default_term_matcher(term: str, content: str) -> bool:
    normalized_term = term.lower().replace("-", "_").replace(" ", "_")
    normalized_content = content.lower().replace("-", "_").replace(" ", "_")
    candidates = {normalized_term}
    if normalized_term.endswith("s"):
        candidates.add(normalized_term[:-1])
    return any(candidate and candidate in normalized_content for candidate in candidates)


def _failure_reason(
    existing_paths: list[str],
    causal_functions: list[dict[str, object]],
    missing_terms: list[str],
    assertions: list[dict[str, object]],
) -> str:
    if not existing_paths:
        return "missing_mapped_test"
    if not causal_functions:
        return "missing_causal_assertion"
    if missing_terms:
        return "missing_requirement_assertion_evidence"
    if not assertions:
        return "requirement_assertion_disconnected"
    return ""
