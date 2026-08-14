from typing import Iterable, List

from core.forge.contracts import FeasiblePlan
from core.forge.domains.base import BaseDomainAdapter, GenericDomainAdapter
from core.forge.domains.cli import CliDomainAdapter
from core.forge.domains.library import LibraryDomainAdapter
from core.forge.domains.pipeline import PipelineDomainAdapter
from core.forge.domains.service import ServiceDomainAdapter


class DomainAdapterRegistry:
    """Selects one deterministic generator from typed plan structure."""

    def __init__(self, adapters: Iterable[BaseDomainAdapter] | None = None):
        configured = list(adapters) if adapters is not None else [
            PipelineDomainAdapter(),
            ServiceDomainAdapter(),
            CliDomainAdapter(),
            LibraryDomainAdapter(),
            GenericDomainAdapter(),
        ]
        if not configured:
            raise ValueError("At least one domain adapter is required.")
        self._adapters: List[BaseDomainAdapter] = configured

    @property
    def adapter_names(self) -> List[str]:
        return [adapter.name for adapter in self._adapters]

    def select(self, plan: FeasiblePlan) -> BaseDomainAdapter:
        for adapter in self._adapters:
            if adapter.matches(plan):
                return adapter
        raise ValueError("No domain adapter accepted the feasible plan.")
