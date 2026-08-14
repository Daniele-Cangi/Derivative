from typing import List, Set

from core.forge.contracts import ArtifactTargetType, FeasiblePlan, PlanInterface, PlanTest
from core.forge.domains.base import BaseDomainAdapter
from core.forge.domains.library_allocation import (
    is_largest_remainder_library,
    render_allocation_library_file,
    render_allocation_library_test,
)
from core.forge.domains.library_email import (
    is_email_normalization_library,
    render_email_library_file,
    render_email_library_test,
)
from core.forge.domains.library_intervals import (
    is_interval_merge_library,
    render_interval_library_file,
    render_interval_library_test,
)
from core.forge.domains.library_semver import (
    is_semver_library,
    render_semver_library_file,
    render_semver_library_test,
)


class LibraryDomainAdapter(BaseDomainAdapter):
    name = "library"

    def matches(self, plan: FeasiblePlan) -> bool:
        if plan.build_spec.target_artifact_type == ArtifactTargetType.LIBRARY:
            return True
        paths = {item.path.replace("\\", "/").lower() for item in plan.file_tree_plan}
        return {"src/library/__init__.py", "src/library/core.py"}.issubset(paths)

    def render_file(self, plan: FeasiblePlan, path: str, interfaces: List[PlanInterface]) -> str:
        renderers = []
        if is_largest_remainder_library(plan):
            renderers.append(render_allocation_library_file)
        if is_semver_library(plan):
            renderers.append(render_semver_library_file)
        if is_interval_merge_library(plan):
            renderers.append(render_interval_library_file)
        if is_email_normalization_library(plan):
            renderers.append(render_email_library_file)
        for renderer in renderers:
            rendered = renderer(plan, path, interfaces)
            if rendered is not None:
                return rendered
        return self._template_generic_module(path, interfaces)

    def render_test(self, plan: FeasiblePlan, plan_test: PlanTest) -> str:
        if is_largest_remainder_library(plan):
            return render_allocation_library_test(plan, plan_test)
        if is_semver_library(plan):
            return render_semver_library_test(plan, plan_test)
        if is_interval_merge_library(plan):
            return render_interval_library_test(plan, plan_test)
        if is_email_normalization_library(plan):
            return render_email_library_test(plan, plan_test)
        return self._template_generic_requirement_test(plan, plan_test)

    def provided_capabilities(self, plan: FeasiblePlan) -> Set[str]:
        supported = any(
            matcher(plan)
            for matcher in (
                is_email_normalization_library,
                is_largest_remainder_library,
                is_semver_library,
                is_interval_merge_library,
            )
        )
        return {"library_public_api"} if supported else set()
