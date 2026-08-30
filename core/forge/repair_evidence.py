from typing import Any, Dict, Iterable, List

from core.forge.contracts import ValidationArtifact
from core.forge.evidence_integrity import canonical_json_bytes


def compile_requirement_assertion_targets(
    validation: ValidationArtifact,
) -> Dict[str, Any]:
    if "missing_requirement_assertion_evidence" not in validation.failure_signatures:
        return {}

    evidence = validation.evidence if isinstance(validation.evidence, dict) else {}
    layer2 = _mapping(evidence.get("layer2"))
    layer3 = _mapping(evidence.get("layer3"))
    targets: Dict[str, Any] = {}

    semantic_checks = _mapping(layer2.get("requirement_semantic_checks"))
    for mismatch in _list(semantic_checks.get("requirement_assertion_mismatches")):
        if not isinstance(mismatch, dict):
            continue
        requirement_id = mismatch.get("requirement_id")
        if not isinstance(requirement_id, str):
            continue
        _merge_target(
            targets,
            requirement_id,
            _mapping(mismatch.get("assertion_evidence")),
            _list(mismatch.get("test_paths")),
            (
                "layer2.requirement_semantic_checks."
                f"requirements:{requirement_id}.assertion_evidence"
            ),
        )

    adversarial = _mapping(layer3.get("semantic_requirement_test_coverage"))
    missing_ids = {
        value
        for value in _list(
            adversarial.get("missing_requirement_assertion_evidence")
        )
        if isinstance(value, str)
    }
    requirements = _mapping(adversarial.get("requirements"))
    for requirement_id in sorted(missing_ids):
        item = _mapping(requirements.get(requirement_id))
        _merge_target(
            targets,
            requirement_id,
            _mapping(item.get("assertion_evidence")),
            _list(item.get("mapped_tests")),
            (
                "layer3.semantic_requirement_test_coverage."
                f"requirements:{requirement_id}.assertion_evidence"
            ),
        )

    return {
        requirement_id: targets[requirement_id]
        for requirement_id in sorted(targets)
    }


def _merge_target(
    targets: Dict[str, Any],
    requirement_id: str,
    assertion_evidence: Dict[str, Any],
    fallback_paths: List[Any],
    evidence_ref: str,
) -> None:
    target = targets.setdefault(
        requirement_id,
        {
            "test_paths": [],
            "causal_functions": [],
            "assertions": [],
            "required_terms": [],
            "covered_terms": [],
            "missing_terms": [],
            "failure_reason": "",
            "evidence_refs": [],
        },
    )
    paths = [
        *_list(assertion_evidence.get("existing_test_paths")),
        *_list(assertion_evidence.get("mapped_test_paths")),
        *fallback_paths,
    ]
    target["test_paths"] = _dedupe_strings(
        [*target["test_paths"], *(path for path in paths if isinstance(path, str))]
    )
    for key in ("required_terms", "covered_terms", "missing_terms"):
        values = [
            value
            for value in _list(assertion_evidence.get(key))
            if isinstance(value, str)
        ]
        target[key] = _dedupe_strings([*target[key], *values])
    for key in ("causal_functions", "assertions"):
        values = [
            value
            for value in _list(assertion_evidence.get(key))
            if isinstance(value, dict)
        ]
        target[key] = _dedupe_mappings([*target[key], *values])
    if not target["failure_reason"]:
        target["failure_reason"] = str(assertion_evidence.get("failure_reason", ""))
    target["evidence_refs"] = _dedupe_strings(
        [*target["evidence_refs"], evidence_ref]
    )


def _mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _dedupe_mappings(values: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        marker = canonical_json_bytes(value).decode("utf-8")
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result
