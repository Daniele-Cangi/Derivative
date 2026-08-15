import ast
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.forge.benchmark import (
    TERMINAL_INFEASIBLE_PROVEN,
    TERMINAL_VALIDATION_FAILED,
    TERMINAL_VERIFIED,
)
from core.forge.blind_benchmark import BlindBenchmarkBundle, load_blind_bundle
from core.forge.blind_freeze import BlindFreezeProvenance, freeze_blind_bundle
from core.model_provider import (
    create_openai_client,
    generate_text,
    is_live_openai_key,
    resolve_openai_api_key,
    resolve_openai_model,
)


TextGenerator = Callable[..., str]


@dataclass(frozen=True)
class BlindProducerConfig:
    bundle_id: str
    verified_cases: int = 6
    validation_failed_cases: int = 3
    infeasible_cases: int = 3
    max_generation_attempts: int = 3

    @property
    def total_cases(self) -> int:
        return self.verified_cases + self.validation_failed_cases + self.infeasible_cases


def produce_and_freeze_blind_bundle(
    *,
    output_root: str | Path,
    repository_root: str | Path,
    config: BlindProducerConfig,
    text_generator: TextGenerator | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> BlindBenchmarkBundle:
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Blind producer output already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if config.total_cases < 3:
        raise ValueError("Blind producer requires at least three cases.")
    if min(
        config.verified_cases,
        config.validation_failed_cases,
        config.infeasible_cases,
        config.max_generation_attempts,
    ) < 1:
        raise ValueError("Blind producer case counts and attempts must be positive.")

    resolved_model = resolve_openai_model(model)
    generator = text_generator or _live_generator(api_key)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".blind-v3-stage-",
            dir=str(destination.parent),
        )
    )
    try:
        raw_cases = _generate_requirement_cases(generator, resolved_model, config)
        dataset = _materialize_cases_and_oracles(
            generator=generator,
            model=resolved_model,
            staging=staging,
            raw_cases=raw_cases,
            config=config,
        )
        (staging / "cases.json").write_bytes(
            (json.dumps(dataset, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        freeze_blind_bundle(
            bundle_root=staging,
            bundle_id=config.bundle_id,
            provenance=BlindFreezeProvenance(
                producer=f"OpenAI Responses API isolated producer ({resolved_model})",
                requirements_origin=(
                    "Fresh one-shot generation from a domain-neutral benchmark brief; "
                    "no Forge source, generated artifact, or prior blind case was supplied."
                ),
                oracle_origin=(
                    "Separate stateless generation request per verified requirement; "
                    "no Forge source or candidate implementation was supplied."
                ),
                declaration=(
                    "Requirements and black-box oracles were generated, statically checked, "
                    "and sealed in one transaction before any case was executed by Forge."
                ),
            ),
            source_urls=[],
            repository_root=repository_root,
        )
        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return load_blind_bundle(
        str(destination / "manifest.json"),
        repository_root=repository_root,
        verify_baseline=True,
    )


def _live_generator(api_key: str | None) -> TextGenerator:
    resolved_key = resolve_openai_api_key(api_key)
    if not is_live_openai_key(resolved_key):
        raise RuntimeError("Blind production requires a live OPENAI_API_KEY.")
    client = create_openai_client(resolved_key)

    def invoke(**kwargs: Any) -> str:
        return generate_text(client, **kwargs)

    return invoke


def _generate_requirement_cases(
    generator: TextGenerator,
    model: str,
    config: BlindProducerConfig,
) -> list[dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "cases": {
                "type": "array",
                "minItems": config.total_cases,
                "maxItems": config.total_cases,
                "items": {
                    "type": "object",
                    "properties": {
                        "requirement": {"type": "string"},
                        "expected_terminal_status": {
                            "type": "string",
                            "enum": [
                                TERMINAL_VERIFIED,
                                TERMINAL_VALIDATION_FAILED,
                                TERMINAL_INFEASIBLE_PROVEN,
                            ],
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 6,
                        },
                    },
                    "required": ["requirement", "expected_terminal_status", "tags"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["cases"],
        "additionalProperties": False,
    }
    feedback = ""
    for _ in range(config.max_generation_attempts):
        response = generator(
            model=model,
            max_output_tokens=7000,
            instructions=_requirement_producer_instructions(config),
            input_text=(
                "Create the frozen benchmark case definitions now. Do not include oracle code."
                + feedback
            ),
            output_schema=schema,
            output_schema_name="forge_blind_v3_requirements",
        )
        payload = _parse_json_object(response, "requirement producer")
        cases = payload.get("cases")
        error = _case_set_error(cases, config)
        if error is None:
            return list(cases)
        feedback = f"\nThe previous response was rejected: {error}. Return a complete replacement."
    raise ValueError(f"Requirement producer failed validation: {feedback.strip()}")


def _materialize_cases_and_oracles(
    *,
    generator: TextGenerator,
    model: str,
    staging: Path,
    raw_cases: list[dict[str, Any]],
    config: BlindProducerConfig,
) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for index, item in enumerate(raw_cases, start=1):
        case_id = f"V3-{index:03d}"
        status = str(item["expected_terminal_status"])
        case = {
            "case_id": case_id,
            "requirement": str(item["requirement"]).strip(),
            "expected_terminal_status": status,
            "tags": ["blind-v3", *[str(tag).strip() for tag in item["tags"]]],
        }
        if status == TERMINAL_VERIFIED:
            relative_oracle = Path("oracles") / case_id / "oracle.py"
            oracle_source = _generate_oracle(
                generator=generator,
                model=model,
                case_id=case_id,
                requirement=case["requirement"],
                max_attempts=config.max_generation_attempts,
            )
            oracle_path = staging / relative_oracle
            oracle_path.parent.mkdir(parents=True, exist_ok=True)
            oracle_path.write_bytes((oracle_source.rstrip() + "\n").encode("utf-8"))
            case["oracle"] = {
                "path": relative_oracle.as_posix(),
                "timeout_seconds": 30,
            }
        dataset.append(case)
    return dataset


def _generate_oracle(
    *,
    generator: TextGenerator,
    model: str,
    case_id: str,
    requirement: str,
    max_attempts: int,
) -> str:
    schema = {
        "type": "object",
        "properties": {"oracle_py": {"type": "string"}},
        "required": ["oracle_py"],
        "additionalProperties": False,
    }
    feedback = ""
    for _ in range(max_attempts):
        response = generator(
            model=model,
            max_output_tokens=6000,
            instructions=_oracle_producer_instructions(),
            input_text=(
                f"Case id: {case_id}\nRequirement:\n{requirement}\n"
                "Produce the complete independent oracle now."
                + feedback
            ),
            output_schema=schema,
            output_schema_name="forge_blind_v3_oracle",
        )
        payload = _parse_json_object(response, f"oracle producer {case_id}")
        source = str(payload.get("oracle_py", ""))
        error = _oracle_error(source)
        if error is None:
            return source
        feedback = f"\nThe previous oracle was rejected: {error}. Return a complete replacement."
    raise ValueError(f"Oracle producer failed validation for {case_id}: {feedback.strip()}")


def _case_set_error(cases: object, config: BlindProducerConfig) -> str | None:
    if not isinstance(cases, list) or len(cases) != config.total_cases:
        return f"expected exactly {config.total_cases} cases"
    expected_counts = {
        TERMINAL_VERIFIED: config.verified_cases,
        TERMINAL_VALIDATION_FAILED: config.validation_failed_cases,
        TERMINAL_INFEASIBLE_PROVEN: config.infeasible_cases,
    }
    observed_counts = {status: 0 for status in expected_counts}
    normalized_requirements: set[str] = set()
    for item in cases:
        if not isinstance(item, dict):
            return "every case must be an object"
        requirement = str(item.get("requirement", "")).strip()
        if len(requirement) < 80:
            return "every requirement must be explicit and at least 80 characters"
        normalized = " ".join(requirement.lower().split())
        if normalized in normalized_requirements:
            return "requirements must be unique"
        normalized_requirements.add(normalized)
        status = str(item.get("expected_terminal_status", ""))
        if status not in observed_counts:
            return f"unsupported terminal status {status!r}"
        observed_counts[status] += 1
        tags = item.get("tags")
        if not isinstance(tags, list) or len(tags) < 2:
            return "every case requires at least two tags"
    if observed_counts != expected_counts:
        return f"terminal status distribution {observed_counts} != {expected_counts}"
    return None


def _oracle_error(source: str) -> str | None:
    if not source.strip():
        return "oracle source is empty"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"oracle syntax error at line {exc.lineno}: {exc.msg}"

    test_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    if len(test_functions) < 3:
        return "oracle must define at least three independent pytest tests"
    if any(
        isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Constant)
        and node.test.value is True
        for node in ast.walk(tree)
    ):
        return "oracle contains trivial assert True"
    semantic_checks = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert) or _is_pytest_raises(node)
    )
    if semantic_checks < 3:
        return "oracle must contain at least three semantic assertions or exception checks"
    if any(isinstance(node, ast.Pass) for test in test_functions for node in ast.walk(test)):
        return "oracle test contains pass"

    forbidden_imports = {"core", "forge", "subprocess", "socket", "requests", "httpx"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".", 1)[0]}
        else:
            continue
        overlap = names & forbidden_imports
        if overlap:
            return "oracle imports forbidden module(s): " + ", ".join(sorted(overlap))
    return None


def _is_pytest_raises(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "raises"
    )


def _parse_json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} returned invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must return a JSON object.")
    return payload


def _requirement_producer_instructions(config: BlindProducerConfig) -> str:
    return f"""You are an independent software benchmark producer operating without access to Forge source code.
Create exactly {config.total_cases} new greenfield Python requirements within CLI, library, service-module, and data-pipeline surfaces.
Use only the Python standard library in required implementations. Every verified requirement must declare exact public interfaces,
input and output behavior, edge cases, deterministic failure behavior, and behavioral tests. Avoid familiar benchmark tasks involving
semantic versions, JSON Pointer, RFC 3339 parsing, interval merging, generic record sorting, email canonicalization, largest-remainder
allocation, sensor aggregation, or idempotent event creation. Prefer less common but production-plausible transformations and policies.
Create exactly {config.verified_cases} feasible verified cases, {config.validation_failed_cases} cases whose claims are materially
ambiguous or universally unprovable, and {config.infeasible_cases} cases with precise formal contradictions. Do not weaken impossible
constraints. Do not include solutions, implementation hints, test code, oracle details, Markdown, or references to Forge internals.
Return only the requested structured object."""


def _oracle_producer_instructions() -> str:
    return """You are an independent black-box acceptance-test producer. You receive one frozen natural-language requirement and no
Forge source code or generated implementation. Return a complete pytest module testing only the public contract. The package under
test is the current working directory and its src directory is already on PYTHONPATH. Import the exact public module/function named
by the requirement. Define at least three tests with non-trivial fixtures, direct target invocation, concrete semantic assertions,
edge cases, and required exception checks. Tests must be deterministic, cross-platform, offline, and standard-library-only except
for pytest. Do not use subprocesses, sockets, HTTP clients, timing assumptions, skip/xfail, assert True, source inspection, manifest
inspection, Forge modules, generated tests, or implementation-private names. Return only the requested structured object."""
