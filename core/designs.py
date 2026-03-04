from dataclasses import dataclass
from typing import List

from core.artifacts import DesignArtifact


@dataclass
class EmergentDesign:
    design_id: str
    title: str
    premise: str
    composition_tags: List[str]
    component_primitives: List[str]
    implementation_outline: List[str]
    governing_constraints: List[str]
    novelty_score: float
    feasibility_score: float
    composite_score: float
    artifacts: List[DesignArtifact]
    lineage_titles: List[str]
    mutation_strategy: str
