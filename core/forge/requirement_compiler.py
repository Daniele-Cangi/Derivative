import hashlib
import re
from typing import List

from core.forge.contracts import (
    AcceptanceContract,
    AcceptanceCriterion,
    ArtifactTargetType,
    BuildSpec,
    ObligationContract,
    QualityContract,
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
        public_module = self._extract_public_module(normalized, target_artifact_type)
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
        quality_contract = self._extract_quality_contract(normalized)

        return BuildSpec(
            build_id=self._build_id(normalized),
            raw_requirement=requirement,
            normalized_requirement=normalized,
            functional_goals=functional_goals,
            non_functional_constraints=non_functional_constraints,
            requirement_atoms=requirement_atoms,
            acceptance_contract=acceptance_contract,
            obligation_contract=obligation_contract,
            quality_contract=quality_contract,
            target_artifact_type=target_artifact_type,
            public_module=public_module,
            risk_hints=self._derive_risk_hints(
                normalized,
                ambiguity_flags,
                non_functional_constraints,
                requirement_atoms,
            ),
            ambiguity_flags=ambiguity_flags,
            assumptions=self._derive_assumptions(normalized),
        )

    def _extract_quality_contract(self, requirement: str) -> QualityContract:
        lowered = requirement.lower()
        quality = QualityContract(
            auth_level="plaintext",
            secrets_in_plaintext=True,
            rate_limit_scope="per_user",
            rate_limit_persistent=False,
            schema_versioned=False,
            audit_trail=False,
            health_endpoint=False,
            structured_logging=False,
            test_coverage_target=0.6,
            integration_tests=False,
        )

        # Auth quality
        if any(token in lowered for token in ("jwt", "bearer token", "oauth")):
            quality.auth_level = "jwt"
            quality.secrets_in_plaintext = False
        elif any(token in lowered for token in ("hashed", "bcrypt", "argon2")):
            quality.auth_level = "hashed"
            quality.secrets_in_plaintext = False
        elif any(token in lowered for token in ("api key", "api-key", "authentication")):
            quality.auth_level = "plaintext"
            quality.secrets_in_plaintext = True
            if "secure" in lowered:
                quality.auth_level = "hashed"
                quality.secrets_in_plaintext = False

        # Rate limiting quality
        if any(token in lowered for token in ("distributed", "redis", "across instances")):
            quality.rate_limit_scope = "distributed"
            quality.rate_limit_persistent = True
        elif any(token in lowered for token in ("per user", "per-user", "per client")):
            quality.rate_limit_scope = "per_user"
        elif "rate limit" in lowered or "rate limiting" in lowered:
            quality.rate_limit_scope = "per_user"
            quality.rate_limit_persistent = False
        if any(token in lowered for token in ("persistent", "survives restart", "survive restart", "restart")):
            quality.rate_limit_persistent = True

        # Persistence quality
        if any(token in lowered for token in ("migrations", "versioned schema", "alembic")):
            quality.schema_versioned = True
        if any(token in lowered for token in ("audit log", "audit trail", "event log", "full audit trail")):
            quality.audit_trail = True
        if any(token in lowered for token in ("production", "prod-ready", "production-grade")):
            quality.schema_versioned = True
            quality.audit_trail = True

        # Observability quality
        if any(
            token in lowered
            for token in (
                "health check",
                "monitoring",
                "observability",
                "structured json logging",
                "structured logging",
                "structured error logging",
            )
        ):
            quality.health_endpoint = True
            quality.structured_logging = True
        if any(token in lowered for token in ("production", "prod-ready", "production-grade")):
            quality.health_endpoint = True

        # Test quality
        if any(token in lowered for token in ("integration tests", "end-to-end", "e2e")):
            quality.integration_tests = True
        if any(token in lowered for token in ("production", "prod-ready", "production-grade")):
            quality.test_coverage_target = 0.8
            quality.integration_tests = True

        computed_level = quality.compute_level()
        if any(token in lowered for token in ("microservice", "service", "rest", "api")) and computed_level < 5:
            computed_level = 5
        if any(token in lowered for token in ("production", "prod-ready", "production-grade")) and computed_level > 9:
            computed_level = 9
        quality.overall_level = computed_level
        return quality

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
                    evidence_terms=self._extract_evidence_terms(normalized_clause),
                    verification_method=self._verification_method_for_clause(
                        normalized_clause,
                        category,
                    ),
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

    def _leading_requirement_clause(self, requirement: str) -> str:
        lowered = requirement.lower()
        pivot = lowered.find(" that ")
        if pivot < 0:
            return ""
        leading = requirement[:pivot].strip(" .")
        if not leading:
            return ""
        if re.match(r"^(build|create|implement|develop|deliver)\b", leading, re.IGNORECASE):
            return leading
        return ""

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
        explicit_callable = bool(
            re.search(r"\bdef\s+[a-z_][a-z0-9_]*\s*\(", lowered)
            or re.search(
                r"\b(?:function|callable)\s+(?:(?:called|named)\s+)?"
                r"['\"]?[a-z_][a-z0-9_]*['\"]?\s*\(",
                lowered,
            )
            or re.search(
                r"\b(?:implement|create|provide|define|write|develop)\b.{0,80}\bfunction\b",
                lowered,
            )
            or re.search(
                r"\b(?:function|method|component)\s+['\"]?[a-z_][a-z0-9_]*['\"]?"
                r"\s+(?:accepting|taking|that|to|with)\b",
                lowered,
            )
        )
        explicit_http_service = bool(
            re.search(r"\b(?:rest|http|api|endpoint|microservice|server)\b", lowered)
        )
        if explicit_callable and not explicit_http_service:
            return ArtifactTargetType.LIBRARY
        if re.search(r"\bdata\s+pipeline\b|\bpipeline\b", lowered):
            return ArtifactTargetType.PIPELINE
        if explicit_http_service:
            return ArtifactTargetType.SERVICE
        if re.search(r"\bservice\b", lowered):
            return ArtifactTargetType.SERVICE
        if re.search(r"\blibrary\b|\bpackage\b|\bsdk\b", lowered):
            return ArtifactTargetType.LIBRARY
        if re.search(r"\bscript\b", lowered):
            return ArtifactTargetType.SCRIPT
        return ArtifactTargetType.UNKNOWN

    def _extract_public_module(
        self,
        requirement: str,
        target_artifact_type: ArtifactTargetType,
    ) -> str:
        explicit_module_patterns = (
            r"\bmodule\s+(?:named|called)\s+['\"]?([a-z_][a-z0-9_]*)['\"]?",
            r"\bin\s+(?:an?\s+)?module\s+(?:(?:named|called)\s+)?"
            r"['\"]?([a-z_][a-z0-9_]*)['\"]?",
            r"\b(?:create|implement|provide|define|write|develop)\s+(?:an?\s+)?"
            r"(?:python\s+)?['\"]?([a-z_][a-z0-9_]*)['\"]?\s+module\b",
        )
        for pattern in explicit_module_patterns:
            match = re.search(pattern, requirement, re.IGNORECASE)
            if not match:
                continue
            module_name = match.group(1).lower()
            if module_name not in {"python", "library", "package", "public"}:
                return module_name

        if target_artifact_type == ArtifactTargetType.CLI:
            cli_match = re.search(
                r"\b(?:cli|command[- ]line)\s+(?:utility|tool|command)\s+"
                r"(?:(?:named|called)\s+)?['\"]?([a-z_][a-z0-9_]*)['\"]?",
                requirement,
                re.IGNORECASE,
            )
            return cli_match.group(1).lower() if cli_match else ""

        if target_artifact_type != ArtifactTargetType.LIBRARY:
            return ""

        signature_match = re.search(
            r"\bdef\s+([a-z_][a-z0-9_]*)\s*\(",
            requirement,
            re.IGNORECASE,
        )
        if signature_match:
            return signature_match.group(1).lower()

        named_callable_match = re.search(
            r"\b(?:library\s+function|function|method|component)\s+"
            r"['\"]([a-z_][a-z0-9_]*)['\"]",
            requirement,
            re.IGNORECASE,
        )
        if named_callable_match:
            return named_callable_match.group(1).lower()
        return ""

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
        if re.search(r"\b(?:identify|identifies|flag|flags)\b.{0,40}\brisky\b", lowered):
            has_risk_rule = bool(
                re.search(
                    r"\b(?:risk[_\s-]?score|threshold|score|rule|criteria|criterion)\b.{0,30}"
                    r"(?:>=|<=|>|<|equal|at least|at most|\d)",
                    lowered,
                )
            )
            if not has_risk_rule:
                flags.append("Risk classification criteria are materially unspecified.")
        if re.search(r"\b(?:appropriate|suitable)\s+report\b", lowered):
            flags.append("Report schema and output format are materially unspecified.")
        if re.search(
            r"\b(?:formally\s+)?unprovable\b|\binherently\s+ambiguous\b|"
            r"\bno\s+mechanism\b.{0,80}\bdefined\b",
            lowered,
        ):
            flags.append(
                "Requirement explicitly declares behavior materially unspecified or unprovable."
            )
        if (
            re.search(r"\b(?:pseudo[- ]random|prng|random\s+generator)\b", lowered)
            and re.search(r"\bseed(?:ed)?\b", lowered)
            and not re.search(
                r"\b(?:mt19937|mersenne\s+twister|pcg(?:32|64)?|xorshift|"
                r"splitmix|chacha(?:8|12|20)?|random\.random|random\.randint)\b",
                lowered,
            )
        ):
            flags.append(
                "Pseudo-random algorithm is materially unspecified; a seed alone does not define "
                "a portable output sequence."
            )
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
                    verification_hint={
                        "interface_contract": "Validate the declared public interface structurally.",
                        "static_analysis": "Validate through static source and dependency analysis.",
                        "property_test": "Validate through executable property-oriented tests.",
                        "universal_proof": "Require explicit proof evidence; finite examples are insufficient.",
                    }.get(atom.verification_method, "Validate through executable behavior and tests."),
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
                ArtifactTargetType.PIPELINE,
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
        return requirement.strip(" .")

    def _extract_atomic_clauses(self, body: str) -> List[str]:
        clauses: List[str] = []
        clause_verb = (
            r"builds?|creates?|implements?|develops?|delivers?|provides?|designs?|defines?|"
            r"writes?|reads?|extracts?|flags?|includes?|guarantees?|supports?|validates?|"
            r"processes?|identifies?|produces?|parses?|rejects?|computes?|aggregates?|"
            r"exposes?|accepts?|yields?|preserves?|tolerates?|inverts?|maps?|sorts?|"
            r"returns?|raises?|skips?|handles?|exits?|merges?|compares?|detects?|removes?|outputs?|uses?"
            r"|survives?"
        )
        quality_clause_start = (
            r"persistent\b|(?:a\s+)?full\s+audit\b|structured\b|integration\s+tests?\b|"
            r"health\s+(?:check|endpoint)\b|versioned\s+schema\b"
        )
        boundary = re.compile(
            rf",\s*(?=(?:and\s+|then\s+)?(?:must\b|shall\b|should\b|will\b|"
            rf"{clause_verb}\b|{quality_clause_start}))|"
            rf"\s+and\s+(?=(?:must\b|shall\b|should\b|will\b|{clause_verb}\b))|"
            rf"\s+that\s+(?=(?:must\b|shall\b|should\b|will\b|{clause_verb}\b))",
            re.IGNORECASE,
        )
        for sentence in re.split(r"(?<=[.!?])\s+|;\s*", body):
            for clause in boundary.split(sentence):
                cleaned = re.sub(r"^(and|then)\s+", "", clause.strip(" ,.;"), flags=re.IGNORECASE)
                if cleaned:
                    clauses.append(cleaned)
        return self._normalize_clause_boundaries(clauses)

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
        universal_tokens = (
            "every possible",
            "all possible",
            "any possible",
            "arbitrary",
            "guarantee",
            "guarantees",
            "for every",
            "for all",
        )
        validation_tokens = ("test", "tests", "malformed", "invalid", "reject", "validate", "verif")
        quality_tokens = (
            "latency",
            "performance",
            "memory",
            "secure",
            "security",
            "reliable",
            "scalable",
            "persistent",
            "survives restart",
            "audit",
            "logging",
            "health",
            "monitoring",
            "observability",
            "production-grade",
        )
        comparator_pattern = re.compile(
            r"\b(exactly|strictly|at least|at most|no more than|does not exceed|less than|greater than)\b",
            re.IGNORECASE,
        )
        functional_tokens = (
            "build",
            "create",
            "implement",
            "develop",
            "deliver",
            "provide",
            "design",
            "define",
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
            "expose",
            "accept",
            "yield",
            "preserve",
            "tolerate",
            "invert",
            "map",
            "sort",
            "return",
            "raise",
            "skip",
            "handle",
            "exit",
            "merge",
            "compare",
            "detect",
            "remove",
            "output",
        )

        if any(token in lowered for token in universal_tokens):
            return "universal_constraint"
        if any(token in lowered for token in validation_tokens):
            return "validation"
        if re.match(
            r"^(builds?|creates?|implements?|develops?|delivers?|provides?|designs?|defines?|writes?|"
            r"reads?|extracts?|flags?|includes?|processes?|identifies?|produces?|parses?|rejects?|"
            r"computes?|aggregates?|exposes?|accepts?|yields?|preserves?|tolerates?|inverts?|maps?|"
            r"sorts?|returns?|raises?|skips?|handles?|exits?|merges?|compares?|detects?|removes?|outputs?)\b",
            lowered,
        ):
            return "functional"
        if any(token in lowered for token in quality_tokens):
            return "quality"
        if re.search(r"\b(?:has|have|contains?|consists?\s+of|requires?)\b", lowered):
            return "non_functional"
        if comparator_pattern.search(clause):
            return "non_functional"
        if re.search(r"\b(?:must(?:\s+not)?|shall(?:\s+not)?)\b", lowered):
            return "functional"
        if any(token in lowered for token in functional_tokens):
            return "functional"
        return "ambiguity"

    def _extract_evidence_terms(self, clause: str) -> List[str]:
        lowered = clause.lower()
        terms: List[str] = []

        def add(term: str) -> None:
            if term not in terms:
                terms.append(term)

        if re.search(r"\bcli\b", lowered):
            add("cli_entrypoint")
        if re.search(r"\b(?:json\s+lines?|jsonl)\b", lowered):
            add("input_jsonl" if re.match(r"^(reads?|parses?|processes?)\b", lowered) else "jsonl")
        if re.search(r"\breads?\b.*\bcsv\b", lowered):
            add("input_csv")
        if "summary csv" in lowered:
            add("summary_csv")

        for identifier in re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", lowered):
            add(identifier)

        semantic_patterns = (
            ("timestamp", r"\btimestamps?\b"),
            ("malformed_records", r"\bmalformed\b"),
            ("duplicate_ids", r"\bduplicate\s+ids?\b"),
            ("missing_fields", r"\bmissing\s+fields?\b"),
            ("invalid_timestamp", r"\binvalid\s+timestamps?\b"),
            ("quarantine", r"\bquarantin(?:e|es|ed|ing)\b"),
            ("minimum", r"\bminimum\b"),
            ("maximum", r"\bmaximum\b"),
            ("average", r"\baverage\b"),
            ("aggregation", r"\baggregat(?:e|es|ed|ing|ion)\b"),
            ("per_device", r"\bper[-\s]+device\b"),
            ("per_customer", r"\bper[-\s]+customer\b"),
            ("summary_json", r"\bsummary\s+json\b"),
            ("idempotent_event", r"\bidempotent\b.*\bevent_id\b|\bevent_id\b.*\bidempotent\b"),
            ("totals", r"\btotals?\b"),
            ("counts", r"\bcounts?\b"),
            ("invalid_dates", r"\binvalid\s+dates?\b"),
            ("malformed_rows", r"\bmalformed\s+rows?\b"),
            ("cli_flow", r"\bcli\s+flow\b"),
            (
                "recursive_json_merge",
                r"\b(?:merge|merges|merged)\b.*\bjson\b.*\brecursiv(?:e|ely)\b|"
                r"\bjson\b.*\brecursiv(?:e|ely)\b.*\b(?:merge|merges|merged)\b",
            ),
            (
                "json_list_replacement",
                r"\breplac(?:e|es|ed|ing)\b.*\blists?\b|\blists?\b.*\breplac(?:e|es|ed|ing)\b",
            ),
            (
                "json_object_root_validation",
                r"\b(?:reject|rejects|validate|validates)\b.*\bnon[-\s]+object\s+root\b",
            ),
        )
        for term, pattern in semantic_patterns:
            if re.search(pattern, lowered):
                add(term)
        return terms

    def _verification_method_for_clause(self, clause: str, category: str) -> str:
        lowered = clause.lower()
        absolute_universal_tokens = (
            "every possible",
            "all possible",
            "any possible",
            "guarantee",
            "guarantees",
            "for every",
            "for all",
        )
        if any(token in lowered for token in absolute_universal_tokens):
            return "universal_proof"
        if category == "universal_constraint":
            return "property_test"
        if re.search(r"\bdef\s+[a-z_][a-z0-9_]*\s*\(", lowered):
            return "interface_contract"
        if re.match(
            r"^(?:build|create|implement|develop|provide|define)\b.*"
            r"\b(?:cli|command|function|method|module|library|service|component)\b",
            lowered,
        ):
            return "interface_contract"
        if "standard library" in lowered or "stdlib" in lowered:
            return "static_analysis"
        return "behavioral_test"

    def _strength_for_clause(self, clause: str, category: str) -> str:
        lowered = clause.lower()
        if category == "ambiguity":
            return "ambiguous"
        universal_tokens = (
            "every possible",
            "all possible",
            "any possible",
            "arbitrary",
            "guarantee",
            "guarantees",
            "for every",
            "for all",
        )
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
