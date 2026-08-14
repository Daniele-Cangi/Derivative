import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ModelUsage:
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    models: dict[str, int] = field(default_factory=dict)

    def record_request(self, model: str) -> None:
        self.request_count += 1
        self.models[model] = self.models.get(model, 0) + 1

    def record_tokens(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> None:
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        self.total_tokens += max(0, total_tokens)

    def estimated_cost(self) -> tuple[float | None, str]:
        if self.request_count == 0:
            return 0.0, "no_model_calls"

        input_rate = _optional_non_negative_float("OPENAI_INPUT_COST_PER_1M_TOKENS")
        output_rate = _optional_non_negative_float("OPENAI_OUTPUT_COST_PER_1M_TOKENS")
        if input_rate is None or output_rate is None:
            return None, "unconfigured"

        cost = (
            (self.input_tokens / 1_000_000) * input_rate
            + (self.output_tokens / 1_000_000) * output_rate
        )
        return round(cost, 8), "environment"


_ACTIVE_MODEL_USAGE: ContextVar[ModelUsage | None] = ContextVar(
    "forge_active_model_usage",
    default=None,
)


@contextmanager
def track_model_usage() -> Iterator[ModelUsage]:
    usage = ModelUsage()
    token = _ACTIVE_MODEL_USAGE.set(usage)
    try:
        yield usage
    finally:
        _ACTIVE_MODEL_USAGE.reset(token)


def record_model_response(response: object) -> None:
    active = _ACTIVE_MODEL_USAGE.get()
    if active is None:
        return

    usage = getattr(response, "usage", None)
    input_tokens = _non_negative_int(getattr(usage, "input_tokens", 0))
    output_tokens = _non_negative_int(getattr(usage, "output_tokens", 0))
    total_tokens = _non_negative_int(
        getattr(usage, "total_tokens", input_tokens + output_tokens)
    )
    active.record_tokens(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def record_model_request(model: str) -> None:
    active = _ACTIVE_MODEL_USAGE.get()
    if active is not None:
        active.record_request(model)


def _non_negative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _optional_non_negative_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None
