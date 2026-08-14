from typing import List, Set

from core.forge.contracts import FeasiblePlan, PlanInterface, PlanTest
from core.forge.domains.base import BaseDomainAdapter
from core.forge.domains.library_email import (
    is_email_normalization_library,
    render_email_library_file,
    render_email_library_test,
)
from core.forge.domains.library_allocation import (
    is_largest_remainder_library,
    render_allocation_library_file,
    render_allocation_library_test,
)


class LibraryDomainAdapter(BaseDomainAdapter):
    name = "library"

    def matches(self, plan: FeasiblePlan) -> bool:
        return is_email_normalization_library(plan) or is_largest_remainder_library(plan)

    def render_file(self, plan: FeasiblePlan, path: str, interfaces: List[PlanInterface]) -> str:
        if is_largest_remainder_library(plan):
            rendered = render_allocation_library_file(plan, path, interfaces)
            if rendered is not None:
                return rendered
        rendered = render_email_library_file(plan, path, interfaces)
        if rendered is not None:
            return rendered
        return self._template_generic_module(path, interfaces)

    def render_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        if is_largest_remainder_library(plan):
            return render_allocation_library_test(plan, plan_test)
        return render_email_library_test(plan, plan_test)

    def provided_capabilities(self, plan: FeasiblePlan) -> Set[str]:
        return {"library_public_api"}
