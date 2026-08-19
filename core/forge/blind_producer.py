import json
import re
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
from core.forge.blind_oracle import (
    oracle_preflight_error,
    oracle_preflight_failure_class,
    oracle_producer_instructions,
    oracle_review_error,
    oracle_review_failure_class,
    oracle_reviewer_instructions,
)
from core.forge.blind_requirement import (
    requirement_review_error,
    requirement_reviewer_instructions,
)
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
    benchmark_version: str = "v3"
    verified_cases: int = 6
    validation_failed_cases: int = 3
    infeasible_cases: int = 3
    max_generation_attempts: int = 5

    def __post_init__(self) -> None:
        if re.fullmatch(r"v[1-9][0-9]*", self.benchmark_version) is None:
            raise ValueError("Blind benchmark_version must match vN, for example 'v4'.")

    @property
    def total_cases(self) -> int:
        return self.verified_cases + self.validation_failed_cases + self.infeasible_cases

    @property
    def case_prefix(self) -> str:
        return self.benchmark_version.upper()

    @property
    def benchmark_tag(self) -> str:
        return f"blind-{self.benchmark_version}"

    @property
    def schema_namespace(self) -> str:
        return f"forge_blind_{self.benchmark_version}"


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
            prefix=f".{config.benchmark_tag}-stage-",
            dir=str(destination.parent),
        )
    )
    try:
        raw_cases, requirement_validation = _generate_requirement_cases(
            generator,
            resolved_model,
            config,
        )
        dataset = _materialize_cases_and_oracles(
            generator=generator,
            model=resolved_model,
            staging=staging,
            raw_cases=raw_cases,
            config=config,
            requirement_validation=requirement_validation,
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
                    "Separate stateless generation and review requests per verified "
                    "requirement; no Forge source or candidate implementation was supplied."
                ),
                declaration=(
                    "Requirements and black-box oracles were generated, causally checked, "
                    "independently reviewed, and sealed in one transaction before any case "
                    "was executed by Forge."
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    rejection_classes: list[str] = []
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
            output_schema_name=f"{config.schema_namespace}_requirements",
        )
        payload = _parse_json_object(response, "requirement producer")
        cases = payload.get("cases")
        error = _case_set_error(cases, config)
        if error is None:
            review = _review_requirement_cases(
                generator=generator,
                model=model,
                cases=list(cases),
                schema_namespace=config.schema_namespace,
            )
            review_error = requirement_review_error(review)
            if review_error is None:
                return list(cases), {
                    "static_checks_passed": True,
                    "independent_review_passed": True,
                    "review_model": model,
                    "findings": [],
                }
            rejection_classes.append("independent_review")
            error = review_error
        else:
            rejection_classes.append("static_case_set")
        feedback = f"\nThe previous response was rejected: {error}. Return a complete replacement."
    classes = ",".join(sorted(set(rejection_classes))) or "unknown"
    raise ValueError(
        "Requirement producer failed validation; "
        f"rejection_classes={classes}"
    )


def _review_requirement_cases(
    *,
    generator: TextGenerator,
    model: str,
    cases: list[dict[str, Any]],
    schema_namespace: str,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "findings": {
                "type": "array",
                "items": {"type": "string", "maxLength": 300},
                "maxItems": 12,
            },
        },
        "required": ["approved", "findings"],
        "additionalProperties": False,
    }
    indexed_cases = [
        {"index": index, **case}
        for index, case in enumerate(cases, start=1)
    ]
    response = generator(
        model=model,
        max_output_tokens=2400,
        instructions=requirement_reviewer_instructions(),
        input_text=(
            "Review this complete candidate set before any oracle is authored:\n"
            + json.dumps(indexed_cases, indent=2, sort_keys=True)
        ),
        output_schema=schema,
        output_schema_name=f"{schema_namespace}_requirements_review",
    )
    return _parse_json_object(response, "requirement reviewer")


def _materialize_cases_and_oracles(
    *,
    generator: TextGenerator,
    model: str,
    staging: Path,
    raw_cases: list[dict[str, Any]],
    config: BlindProducerConfig,
    requirement_validation: dict[str, Any],
) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for index, item in enumerate(raw_cases, start=1):
        case_id = f"{config.case_prefix}-{index:03d}"
        status = str(item["expected_terminal_status"])
        item_tags = [
            str(tag).strip()
            for tag in item["tags"]
            if str(tag).strip() and str(tag).strip() != config.benchmark_tag
        ]
        case = {
            "case_id": case_id,
            "requirement": str(item["requirement"]).strip(),
            "expected_terminal_status": status,
            "tags": [config.benchmark_tag, *item_tags],
            "requirement_validation": requirement_validation,
        }
        if status == TERMINAL_VERIFIED:
            relative_oracle = Path("oracles") / case_id / "oracle.py"
            oracle_source, oracle_validation = _generate_oracle(
                generator=generator,
                model=model,
                case_id=case_id,
                requirement=case["requirement"],
                max_attempts=config.max_generation_attempts,
                schema_namespace=config.schema_namespace,
            )
            oracle_path = staging / relative_oracle
            oracle_path.parent.mkdir(parents=True, exist_ok=True)
            oracle_path.write_bytes((oracle_source.rstrip() + "\n").encode("utf-8"))
            case["oracle"] = {
                "path": relative_oracle.as_posix(),
                "timeout_seconds": 30,
            }
            case["oracle_validation"] = oracle_validation
        dataset.append(case)
    return dataset


def _generate_oracle(
    *,
    generator: TextGenerator,
    model: str,
    case_id: str,
    requirement: str,
    max_attempts: int,
    schema_namespace: str,
) -> tuple[str, dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {"oracle_py": {"type": "string"}},
        "required": ["oracle_py"],
        "additionalProperties": False,
    }
    feedback = ""
    rejection_classes: list[str] = []
    for _ in range(max_attempts):
        response = generator(
            model=model,
            max_output_tokens=6000,
            instructions=oracle_producer_instructions(),
            input_text=(
                f"Case id: {case_id}\nRequirement:\n{requirement}\n"
                "Produce the complete independent oracle now."
                + feedback
            ),
            output_schema=schema,
            output_schema_name=f"{schema_namespace}_oracle",
        )
        payload = _parse_json_object(response, f"oracle producer {case_id}")
        source = str(payload.get("oracle_py", ""))
        error = oracle_preflight_error(source)
        if error is None:
            review = _review_oracle(
                generator=generator,
                model=model,
                case_id=case_id,
                requirement=requirement,
                source=source,
                schema_namespace=schema_namespace,
            )
            review_error = oracle_review_error(review)
            if review_error is None:
                return source, {
                    "static_checks_passed": True,
                    "independent_review_passed": True,
                    "review_model": model,
                    "findings": [],
                }
            rejection_classes.append(oracle_review_failure_class(review_error))
            error = review_error
        else:
            rejection_classes.append(oracle_preflight_failure_class(error))
        feedback = f"\nThe previous oracle was rejected: {error}. Return a complete replacement."
    classes = ",".join(sorted(set(rejection_classes))) or "unknown"
    raise ValueError(
        f"Oracle producer failed validation for {case_id}; "
        f"rejection_classes={classes}"
    )


def _review_oracle(
    *,
    generator: TextGenerator,
    model: str,
    case_id: str,
    requirement: str,
    source: str,
    schema_namespace: str,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "findings": {
                "type": "array",
                "items": {"type": "string", "maxLength": 300},
                "maxItems": 12,
            },
        },
        "required": ["approved", "findings"],
        "additionalProperties": False,
    }
    response = generator(
        model=model,
        max_output_tokens=1800,
        instructions=oracle_reviewer_instructions(),
        input_text=(
            f"Case id: {case_id}\n"
            f"Frozen requirement:\n{requirement}\n\n"
            f"Candidate oracle.py:\n{source}"
        ),
        output_schema=schema,
        output_schema_name=f"{schema_namespace}_oracle_review",
    )
    return _parse_json_object(response, f"oracle reviewer {case_id}")


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
ambiguous or universally unprovable while remaining logically satisfiable in principle, and {config.infeasible_cases} cases with precise
mathematical or finite constraint contradictions independent of platform behavior, unavailable dependencies, or implementation difficulty.
Do not label an environmental limitation as formal infeasibility and do not weaken impossible constraints. Do not include solutions,
implementation hints, test code, oracle details, Markdown, or references to Forge internals.
Return only the requested structured object."""
