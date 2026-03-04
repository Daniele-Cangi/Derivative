import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from core.kernel import ReasoningResult


@dataclass
class VerifiedGenome:
    timestamp: str
    problem_hash: str
    hypothesis: str
    conclusion_snapshot: str
    cycles_used: int
    was_verified: bool


class DesignGenePool:
    def __init__(self, storage_file: str = "verified_gene_pool.json"):
        self.storage_file = storage_file
        self.genomes: List[VerifiedGenome] = self._load()

    def _load(self) -> List[VerifiedGenome]:
        try:
            with open(self.storage_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        if not isinstance(data, list):
            return []

        genomes: List[VerifiedGenome] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                genomes.append(VerifiedGenome(**item))
            except TypeError:
                continue
        return genomes

    def _save(self) -> None:
        with open(self.storage_file, "w", encoding="utf-8") as handle:
            json.dump([dataclasses.asdict(genome) for genome in self.genomes], handle, indent=2)

    def record_execution(self, result: ReasoningResult, problem: str) -> List[VerifiedGenome]:
        execution_result = result.execution_result
        if execution_result is None or not execution_result.converged:
            return []

        problem_hash = hashlib.sha256(problem.encode("utf-8")).hexdigest()
        existing_hypotheses = {
            (genome.problem_hash, genome.hypothesis)
            for genome in self.genomes
        }
        recorded: List[VerifiedGenome] = []

        for cycle in execution_result.history:
            if not cycle.converged:
                continue
            key = (problem_hash, cycle.hypothesis)
            if key in existing_hypotheses:
                continue
            genome = VerifiedGenome(
                timestamp=datetime.now(timezone.utc).isoformat(),
                problem_hash=problem_hash,
                hypothesis=cycle.hypothesis,
                conclusion_snapshot=result.conclusion,
                cycles_used=execution_result.cycles_used,
                was_verified=True,
            )
            self.genomes.append(genome)
            existing_hypotheses.add(key)
            recorded.append(genome)

        if recorded:
            self._save()
        return recorded
