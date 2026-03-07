import hashlib
import re
from typing import List

from core.forge.contracts import (
    AcceptanceContract,
    AcceptanceCriterion,
    ArtifactTargetType,
    BuildSpec,
    ObligationContract,
    RequirementAtom,
)
from core.obligation_compiler import ObligationCompiler
from core.problem_classifier import ProblemClassifier


class RequirementCompiler:
    def __init__(self):
        self.problem_classifier = ProblemClassifier()
        self.obligation_compiler = ObligationCompiler()

    def compile(self, requirement: str) -> BuildSpec:
        normalized = self._normalize_requirement(requirement)
        if not normalized:
            raise ValueError("Requirement cannot be empty.")

        requirement_atoms = self._extract_requirement_atoms(normalized)
        functional_goals = [atom.text for atom in requirement_atoms if atom.category in {"functional", "validation"}]
        if not functional_goals:
            functional_goals = self._extract_functional_goals(normalized)
        target_artifact_type = self._detect_target_artifact_type(normalized)
        non_functional_constraints = [
            atom.text
            for atom in requirement_atoms
            if atom.category in {"non_functional", "quality", "universal_constraint"}
        ]
        ambiguity_flags = self._extract_ambiguity_flags(
            normalized,
            requirement_atoms,
            functional_goals,
            non_functional_constraints,
            target_artifact_type,
        )
        acceptance_contract = self._build_acceptance_contract(
            functional_goals,
            non_functional_constraints,
            requirement_atoms,
        )
        obligation_contract = self._build_obligation_contract(
            normalized,
            target_artifact_type,
            functional_goals,
            acceptance_contract,
        )

        return BuildSpec(
            build_id=self._build_id(normalized),
            raw_requirement=requirement,
            normalized_requirement=normalized,
            functional_goals=functional_goals,
            non_functional_constraints=non_functional_constraints,
            requirement_atoms=requirement_atoms,
            acceptance_contract=acceptance_contract,
            obligation_contract=obligation_contract,
            target_artifact_type=target_artifact_type,
            risk_hints=self._derive_risk_hints(
                normalized,
                ambiguity_flags,
                non_functional_constraints,
                requirement_atoms,
            ),
            ambiguity_flags=ambiguity_flags,
            assumptions=self._derive_assumptions(normalized),
        )

    def _extract_requirement_atoms(self, requirement: str) -> List[RequirementAtom]:
        body = self._requirement_body(requirement)
        clauses = self._extract_atomic_clauses(body)

        atoms: List[RequirementAtom] = []
        seen = set()
        index = 1
        for clause in clauses:
            normalized_clause = " ".join(clause.split())
            if not normalized_clause:
                continue
            dedupe_key = normalized_clause.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            category = self._categorize_clause(normalized_clause)
            strength = self._strength_for_clause(normalized_clause, category)
            atoms.append(
                RequirementAtom(
                    requirement_id=f"R{index:03d}",
                    text=normalized_clause,
                    category=category,
                    strength=strength,
                    source_fragment=clause,
                )
            )
            index += 1

        if not atoms:
            atoms.append(
                RequirementAtom(
                    requirement_id="R001",
                    text=requirement,
                    category="ambiguity",
                    strength="ambiguous",
                    source_fragment=requirement,
                )
            )
        return atoms

    def _normalize_requirement(self, requirement: str) -> str:
        collapsed = " ".join((requirement or "").strip().split())
        return collapsed

    def _build_id(self, normalized_requirement: str) -> str:
        digest = hashlib.sha256(normalized_requirement.encode("utf-8")).hexdigest()[:12]
        return f"build-{digest}"

    def _extract_functional_goals(self, requirement: str) -> List[str]:
        lower = requirement.lower()
        seed_clauses: List[str] = []

        that_index = lower.find(" that ")
        if that_index >= 0:
            seed_clauses.append(requirement[that_index + len(" that "):])
        else:
            seed_clauses.append(requirement)

        segments: List[str] = []
        for clause in seed_clauses:
            segments.extend(self._segment_requirement(clause))

        functional_verbs = (
            "build",
            "read",
            "reads",
            "extract",
            "extracts",
            "flag",
            "flags",
            "write",
            "writes",
            "generate",
            "generates",
            "include",
            "includes",
            "parse",
            "parses",
            "validate",
            "validates",
        )
        goals: List[str] = []
        for segment in segments:
            lowered = segment.lower()
            if any(verb in lowered for verb in functional_verbs):
                goals.append(segment)

        if not goals:
            classifier_goals = self.problem_classifier.extract_explicit_objectives(requirement)
            goals = classifier_goals or [requirement]

        deduplicated: List[str] = []
        seen = set()
        for goal in goals:
            normalized_goal = " ".join(goal.lower().split())
            if normalized_goal in seen:
                continue
            seen.add(normalized_goal)
            deduplicated.append(goal)
        return deduplicated

    def _extract_non_functional_constraints(
        self,
        requirement: str,
        functional_goals: List[str],
        target_artifact_type: ArtifactTargetType,
    ) -> List[str]:
        comparator_pattern = re.compile(
            r"\b(exactly|strictly greater than|greater than|strictly less than|less than|at least|at most|"
            r"no more than|does not exceed|must|should|<=|>=|<|>)\b",
            re.IGNORECASE,
        )
        quality_pattern = re.compile(
            r"\b(latency|performance|memory|secure|security|reliable|reliability|deterministic|"
            r"scalable|availability|compliance|audit)\b",
            re.IGNORECASE,
        )
        test_pattern = re.compile(r"\btests?\b|unit test|integration test|pytest", re.IGNORECASE)
        functional_patterns = (
            "build",
            "read",
            "reads",
            "extract",
            "extracts",
            "flag",
            "flags",
            "write",
            "writes",
            "generate",
            "generates",
            "parse",
            "parses",
        )
        goal_set = {" ".join(goal.lower().split()) for goal in functional_goals}

        constraints: List[str] = []
        for cleaned in self._segment_requirement(requirement):
            if not cleaned:
                continue
            lowered = cleaned.lower()
            normalized_clause = " ".join(lowered.split())
            contains_functional_verb = any(token in lowered for token in functional_patterns)

            # Keep quality/test constraints for software builds, but avoid duplicating functional behavior.
            if target_artifact_type in {
                ArtifactTargetType.CLI,
                ArtifactTargetType.SERVICE,
                ArtifactTargetType.LIBRARY,
                ArtifactTargetType.SCRIPT,
            }:
                if test_pattern.search(cleaned):
                    constraints.append(cleaned)
                    continue
                if quality_pattern.search(cleaned):
                    constraints.append(cleaned)
                    continue
                if comparator_pattern.search(cleaned) and not contains_functional_verb:
                    constraints.append(cleaned)
                    continue
                continue

            if normalized_clause in goal_set and not quality_pattern.search(cleaned):
                continue
            if comparator_pattern.search(cleaned) or quality_pattern.search(cleaned) or test_pattern.search(cleaned):
                constraints.append(cleaned)

        deduplicated: List[str] = []
        seen = set()
        for constraint in constraints:
            normalized_constraint = " ".join(constraint.lower().split())
            if normalized_constraint in seen:
                continue
            seen.add(normalized_constraint)
            deduplicated.append(constraint)
        return deduplicated

    def _detect_target_artifact_type(self, requirement: str) -> ArtifactTargetType:
        lowered = requirement.lower()
        if re.search(r"\bcli\b|command[- ]line", lowered):
            return ArtifactTargetType.CLI
        if re.search(r"\bservice\b|\bapi\b|\bserver\b", lowered):
            return ArtifactTargetType.SERVICE
        if re.search(r"\blibrary\b|\bpackage\b|\bsdk\b", lowered):
            return ArtifactTargetType.LIBRARY
        if re.search(r"\bscript\b", lowered):
            return ArtifactTargetType.SCRIPT
        return ArtifactTargetType.UNKNOWN

    def _extract_ambiguity_flags(
        self,
        requirement: str,
        requirement_atoms: List[RequirementAtom],
        functional_goals: List[str],
        non_functional_constraints: List[str],
        target_artifact_type: ArtifactTargetType,
    ) -> List[str]:
        flags: List[str] = []
        lowered = requirement.lower()

        if target_artifact_type == ArtifactTargetType.UNKNOWN:
            flags.append("Target artifact type is not explicit.")
        if not functional_goals:
            flags.append("No explicit functional goals detected.")
        if "test" not in lowered:
            flags.append("Automated test expectations are not explicit.")
        if ("csv" in lowered or "date" in lowered) and "format" not in lowered:
            flags.append("Input date/CSV format is unspecified.")
        if re.search(r"\b(robust|efficient|scalable|fast)\b", lowered) and not non_functional_constraints:
            flags.append("Quality adjectives are present without measurable constraints.")
        universal_atoms = [atom for atom in requirement_atoms if atom.strength == "universal"]
        if universal_atoms:
            flags.append(
                "Universal/absolute constraints require explicit proof coverage and may fail if only finite tests exist."
            )
        if any(atom.category == "ambiguity" for atom in requirement_atoms):
            flags.append("One or more requirement clauses remained semantically ambiguous.")

        deduplicated: List[str] = []
        seen = set()
        for flag in flags:
            if flag.lower() in seen:
                continue
            seen.add(flag.lower())
            deduplicated.append(flag)
        return deduplicated

    def _build_acceptance_contract(
        self,
        functional_goals: List[str],
        non_functional_constraints: List[str],
        requirement_atoms: List[RequirementAtom],
    ) -> AcceptanceContract:
        criteria: List[AcceptanceCriterion] = []
        index = 1
        for atom in requirement_atoms:
            if atom.category == "ambiguity":
                continue
            if atom.category in {"functional", "validation"}:
                description = f"Implement functional goal: {atom.text}"
            elif atom.category == "universal_constraint":
                description = f"Prove universal constraint: {atom.text}"
            else:
                description = f"Satisfy constraint: {atom.text}"
            criteria.append(
                AcceptanceCriterion(
                    criterion_id=f"AC{index:03d}",
                    description=description,
                    required=True,
                    verification_hint="Validate through executable behavior and tests.",
                    requirement_ids=[atom.requirement_id],
                )
            )
            index += 1
        if not criteria:
            criteria.append(
                AcceptanceCriterion(
                    criterion_id="AC001",
                    description="Deliver the requested build artifact with executable behavior.",
                    required=True,
                    verification_hint="Validate through end-to-end run and tests.",
                    requirement_ids=["R001"],
                )
            )
        return AcceptanceContract(criteria=criteria, pass_condition="all_required", notes=[])

    def _build_obligation_contract(
        self,
        requirement: str,
        target_artifact_type: ArtifactTargetType,
        functional_goals: List[str],
        acceptance_contract: AcceptanceContract,
    ) -> ObligationContract:
        classification = self.problem_classifier.classify(requirement)
        compiled = self.obligation_compiler.compile(requirement, classification)
        required_fields = [spec.field for spec in compiled.specs if spec.required]
        if (
            compiled.mode == "none"
            and target_artifact_type
            in {
                ArtifactTargetType.CLI,
                ArtifactTargetType.SERVICE,
                ArtifactTargetType.LIBRARY,
                ArtifactTargetType.SCRIPT,
            }
        ):
            required = [
                "entrypoint_defined",
                "input_output_contract_defined",
                "core_workflow_defined",
                "tests_defined",
                "acceptance_criteria_covered",
                "validation_layers_defined",
            ]
            schema = {
                "entrypoint_defined": "bool",
                "input_output_contract_defined": "bool",
                "core_workflow_defined": "bool",
                "tests_defined": "bool",
                "acceptance_criteria_covered": "int",
                "validation_layers_defined": "bool",
            }
            return ObligationContract(
                mode="software_build",
                schema=schema,
                required_fields=required,
                context={
                    "target_artifact_type": target_artifact_type.value,
                    "functional_goal_count": len(functional_goals),
                    "acceptance_criteria_count": len(acceptance_contract.criteria),
                    "required_validation_layers": 3,
                },
            )
        return ObligationContract(
            mode=compiled.mode,
            schema=dict(compiled.schema),
            required_fields=required_fields,
            context=dict(compiled.context),
        )

    def _derive_risk_hints(
        self,
        requirement: str,
        ambiguity_flags: List[str],
        non_functional_constraints: List[str],
        requirement_atoms: List[RequirementAtom],
    ) -> List[str]:
        hints: List[str] = []
        lowered = requirement.lower()
        if "csv" in lowered and "date" in lowered:
            hints.append("Date parsing may fail if CSV date formats are heterogeneous.")
        if non_functional_constraints:
            hints.append("Constraint checks must be enforced in validation, not inferred from prose.")
        if any(atom.strength == "universal" for atom in requirement_atoms):
            hints.append("Universal constraints need proof-oriented validation; finite examples are insufficient.")
        for flag in ambiguity_flags:
            hints.append(f"Ambiguity risk: {flag}")

        deduplicated: List[str] = []
        seen = set()
        for hint in hints:
            normalized_hint = " ".join(hint.lower().split())
            if normalized_hint in seen:
                continue
            seen.add(normalized_hint)
            deduplicated.append(hint)
        return deduplicated

    def _derive_assumptions(self, requirement: str) -> List[str]:
        assumptions: List[str] = []
        lowered = requirement.lower()
        if "python" not in lowered:
            assumptions.append("Implementation language defaults to Python.")
        if "csv" in lowered and "delimiter" not in lowered:
            assumptions.append("CSV delimiter is assumed to be comma.")
        return assumptions

    def _segment_requirement(self, requirement: str) -> List[str]:
        segments: List[str] = []
        for segment in re.split(r",|;|\band\b", requirement, flags=re.IGNORECASE):
            cleaned = segment.strip(" .")
            if cleaned:
                segments.append(cleaned)
        return segments

    def _requirement_body(self, requirement: str) -> str:
        lowered = requirement.lower()
        pivot = lowered.find(" that ")
        if pivot >= 0:
            return requirement[pivot + len(" that "):].strip(" .")
        return requirement.strip(" .")

    def _extract_atomic_clauses(self, body: str) -> List[str]:
        verb_pattern = re.compile(
            r"\b(reads?|extracts?|flags?|writes?|includes?|guarantees?|supports?|validates?|"
            r"processes?|identifies?|produces?|parses?|builds?)\b",
            re.IGNORECASE,
        )
        matches = list(verb_pattern.finditer(body))
        clauses: List[str] = []
        if matches:
            for index, match in enumerate(matches):
                start = match.start()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
                clause = body[start:end].strip(" ,.;")
                clause = re.sub(r"^(and|then)\s+", "", clause, flags=re.IGNORECASE)
                if clause:
                    clauses.append(clause)
            return self._normalize_clause_boundaries(clauses)
        return self._segment_requirement(body)

    def _normalize_clause_boundaries(self, clauses: List[str]) -> List[str]:
        normalized: List[str] = []
        for clause in clauses:
            cleaned = re.sub(r"[,;]?\s+(and|or)$", "", clause.strip(), flags=re.IGNORECASE)
            cleaned = cleaned.strip(" ,.;")
            if cleaned:
                normalized.append(cleaned)

        merged: List[str] = []
        index = 0
        while index < len(normalized):
            current = normalized[index]
            if (
                current.lower() in {"guarantee", "guarantees", "support", "supports"}
                and index + 1 < len(normalized)
            ):
                merged.append(f"{current} {normalized[index + 1]}")
                index += 2
                continue
            merged.append(current)
            index += 1
        return merged

    def _categorize_clause(self, clause: str) -> str:
        lowered = clause.lower()
        universal_tokens = ("every", "all", "any", "arbitrary", "guarantee", "guarantees", "exactly")
        validation_tokens = ("test", "tests", "malformed", "invalid", "reject", "validate", "verif")
        quality_tokens = ("latency", "performance", "memory", "secure", "security", "reliable", "scalable")
        comparator_pattern = re.compile(
            r"\b(exactly|strictly|at least|at most|no more than|does not exceed|less than|greater than)\b",
            re.IGNORECASE,
        )
        functional_tokens = (
            "build",
            "read",
            "reads",
            "extract",
            "extracts",
            "flag",
            "flags",
            "write",
            "writes",
            "process",
            "identify",
            "produce",
            "parse",
            "support",
        )

        if any(token in lowered for token in universal_tokens):
            return "universal_constraint"
        if any(token in lowered for token in validation_tokens):
            return "validation"
        if re.match(
            r"^(reads?|extracts?|flags?|writes?|includes?|processes?|identifies?|produces?|parses?|builds?)\b",
            lowered,
        ):
            return "functional"
        if any(token in lowered for token in quality_tokens):
            return "quality"
        if comparator_pattern.search(clause):
            return "non_functional"
        if any(token in lowered for token in functional_tokens):
            return "functional"
        return "ambiguity"

    def _strength_for_clause(self, clause: str, category: str) -> str:
        lowered = clause.lower()
        if category == "ambiguity":
            return "ambiguous"
        universal_tokens = ("every", "all", "any", "arbitrary", "guarantee", "guarantees")
        hard_tokens = (
            "must",
            "exactly",
            "strictly",
            "at least",
            "at most",
            "no more than",
            "does not exceed",
            "less than",
            "greater than",
            "malformed",
            "invalid",
        )
        soft_tokens = ("should", "prefer", "ideally", "if possible")
        if any(token in lowered for token in universal_tokens):
            return "universal"
        if any(token in lowered for token in hard_tokens):
            return "hard"
        if any(token in lowered for token in soft_tokens):
            return "soft"
        return "hard"
