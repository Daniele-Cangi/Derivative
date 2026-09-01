import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from core.forge.contracts import (
    ConditionalNormalizationIssue,
    ConditionalObligation,
    CoverageDirective,
    RequirementAtom,
)


@dataclass
class ConditionalNormalizationResult:
    obligations: list[ConditionalObligation] = field(default_factory=list)
    coverage_directives: list[CoverageDirective] = field(default_factory=list)
    issues: list[ConditionalNormalizationIssue] = field(default_factory=list)


class ConditionalObligationNormalizer:
    """Compile high-confidence conditional semantics without replacing parent atoms."""

    _FINITE_PREDICATE = (
        r"(?:is|are|was|were|has|have|contains?|consists?|matches?|fails?|exceeds?|"
        r"rejects?|accepts?|includes?|starts?|ends?)"
    )
    _CONSEQUENT_START = re.compile(
        r"^(?:(?:the\s+)?(?:cli|tool|function|system|program|application|process|"
        r"result|output|record|line)|it|its\s+value)\b.{0,80}"
        r"\b(?:returns?|outputs?|writes?|written|exits?|is|becomes?|remains?|raises?|"
        r"reject(?:s|ed)?|skip(?:s|ped)?|omit(?:s|ted)?|ignore(?:s|d)?|"
        r"preserv(?:es|ed))\b|"
        r"^(?:return|output|write|raise|exit|reject|skip)\b",
        re.IGNORECASE,
    )

    def normalize(self, atoms: Iterable[RequirementAtom]) -> ConditionalNormalizationResult:
        result = ConditionalNormalizationResult()
        atoms = list(atoms)
        for atom in atoms:
            if atom.category == "coverage_directive":
                continue
            obligations, issue = self._normalize_atom(atom)
            result.obligations.extend(obligations)
            if issue is not None:
                result.issues.append(issue)

        for atom in atoms:
            if atom.category != "coverage_directive":
                continue
            result.coverage_directives.append(
                self._compile_coverage_directive(atom, result.obligations)
            )
        return result

    def _normalize_atom(
        self,
        atom: RequirementAtom,
    ) -> tuple[list[ConditionalObligation], ConditionalNormalizationIssue | None]:
        if atom.category == "negative_constraint" or self._is_explicit_negative(atom.text):
            obligation = self._compile_negative(atom)
            return ([obligation] if obligation is not None else []), None

        parsed = self._conditional_parts(atom.text)
        if parsed is None:
            return [], self._compound_issue(atom) if self._looks_like_compound_conditional(atom.text) else None

        antecedent, consequent = parsed
        triggers = self._split_antecedent(antecedent)
        observations = self._parse_observations(consequent)
        if not triggers or not observations:
            if self._looks_compound(antecedent) and atom.strength in {"hard", "universal"}:
                return [], ConditionalNormalizationIssue(
                    parent_requirement_id=atom.requirement_id,
                    source_fragment=atom.source_fragment,
                    reason="hard_compound_conditional_not_safely_normalized",
                )
            return [], None

        obligations: list[ConditionalObligation] = []
        for branch_index, trigger in enumerate(triggers, start=1):
            witness_class, precondition = self._predicate_contract(trigger)
            for observation_index, observation in enumerate(observations, start=1):
                obligations.append(
                    ConditionalObligation(
                        obligation_id=(
                            f"{atom.requirement_id}.B{branch_index:02d}.O{observation_index:02d}"
                        ),
                        parent_requirement_id=atom.requirement_id,
                        trigger=trigger,
                        precondition=precondition,
                        observable_channel=observation["channel"],
                        comparison_relation=observation["relation"],
                        expected_value=observation["value"],
                        polarity=observation["polarity"],
                        observation_fidelity=observation["fidelity"],
                        verification_method=(
                            "deterministic_probe" if witness_class else atom.verification_method
                        ),
                        source_fragment=atom.source_fragment,
                        witness_class=witness_class,
                    )
                )
        return obligations, None

    def _conditional_parts(self, text: str) -> tuple[str, str] | None:
        normalized = re.sub(
            r"^(?:deterministic\s+failure:\s*|for\s+any\s+[^,]+,\s*)",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        )
        match = re.match(r"^(?:if|when|unless)\s+(.+)$", normalized, re.IGNORECASE)
        if match is None:
            return None
        body = match.group(1)
        for comma in (item.start() for item in re.finditer(r",", body)):
            antecedent = body[:comma].strip(" ,")
            consequent = body[comma + 1 :].strip(" ,")
            if antecedent and consequent and self._CONSEQUENT_START.search(consequent):
                return antecedent, consequent
        return None

    def _split_antecedent(self, antecedent: str) -> list[str]:
        raw_parts = re.split(r"\s*,?\s+or\s+(?:if\s+)?", antecedent, flags=re.IGNORECASE)
        triggers: list[str] = []
        shared_subject = ""
        shared_predicate = ""
        for index, raw in enumerate(raw_parts):
            cleaned = re.sub(r"^if\s+", "", raw.strip(" ,"), flags=re.IGNORECASE)
            if index == 0:
                shared_subject, shared_predicate = self._shared_subject_predicate(cleaned)
            elif shared_subject:
                if re.match(rf"^{self._FINITE_PREDICATE}\b", cleaned, re.IGNORECASE):
                    cleaned = f"{shared_subject} {cleaned}"
                elif not re.search(rf"\b{self._FINITE_PREDICATE}\b", cleaned, re.IGNORECASE):
                    cleaned = f"{shared_subject} {shared_predicate} {cleaned}"
            partitions = self._taxonomy_partitions(cleaned)
            for partition in partitions:
                if partition and partition.lower() not in {item.lower() for item in triggers}:
                    triggers.append(partition)
        return triggers

    @classmethod
    def _shared_subject_predicate(cls, trigger: str) -> tuple[str, str]:
        match = re.match(
            rf"^(?P<subject>.+?)\s+(?P<predicate>{cls._FINITE_PREDICATE})\s+.+$",
            trigger,
            re.IGNORECASE,
        )
        if match is None:
            return "", ""
        return match.group("subject").strip(), match.group("predicate").strip()

    @staticmethod
    def _taxonomy_partitions(trigger: str) -> list[str]:
        slash_failure = re.search(
            r"\b(file\s+)?([a-z][a-z -]*)/([a-z][a-z -]*)\s+fails?\b",
            trigger,
            re.IGNORECASE,
        )
        if slash_failure is None:
            return [trigger]
        prefix = slash_failure.group(1) or ""
        first = f"{prefix}{slash_failure.group(2).strip()} fails"
        second = f"{prefix}{slash_failure.group(3).strip()} fails"
        return [first, second]

    def _parse_observations(self, consequent: str) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []

        for match in re.finditer(
            r"outputs?\s+exactly\s+(['\"])(.*?)\1\s+to\s+(stderr|stdout)",
            consequent,
            re.IGNORECASE,
        ):
            observations.append(self._observation(match.group(3).lower(), "equals", match.group(2), "exact_text"))

        for channel in ("stdout", "stderr"):
            if re.search(
                rf"(?:producing|with)\s+no\s+output\s+(?:to|on)\s+{channel}\b",
                consequent,
                re.IGNORECASE,
            ):
                observations.append(self._observation(channel, "equals", "", "exact_text"))

        if re.search(r"\boutput\s+is\s+empty\b", consequent, re.IGNORECASE):
            observations.append(self._observation("stdout", "equals", "", "exact_text"))

        if re.search(
            r"\b(?:not|never)\s+(?:be\s+)?(?:written|emitted|output)\s+"
            r"(?:to|on)\s+(?:the\s+)?output\b|"
            r"\b(?:omitted|excluded)\s+from\s+(?:the\s+)?output\b",
            consequent,
            re.IGNORECASE,
        ):
            observations.append(
                self._observation("stdout", "excludes", "triggering_input", "semantic")
            )

        exit_match = re.search(
            r"\b(?:exits?\s+)?with\s+(?:exit\s+)?code\s+(-?\d+)\b|"
            r"\bexit\s+code\s+(?:is|equals?)\s+(-?\d+)\b",
            consequent,
            re.IGNORECASE,
        )
        if exit_match:
            value = exit_match.group(1) or exit_match.group(2)
            observations.append(self._observation("exit_code", "equals", int(value), "exact_scalar"))

        raise_match = re.search(r"\braises?\s+([A-Za-z_][A-Za-z0-9_]*)\b", consequent)
        if raise_match:
            observations.append(self._observation("exception", "raises", raise_match.group(1), "exact_type"))

        if re.search(r"\breturn\s+every\s+row\s+unmodified\b", consequent, re.IGNORECASE):
            observations.append(self._observation("return_value", "equals", "input_rows", "semantic"))
        if re.search(r"\b(?:values?|result)\s+remain(?:s)?\s+unchanged\b", consequent, re.IGNORECASE):
            observations.append(self._observation("return_value", "equals", "unchanged", "semantic"))
        if re.search(r"\bvalue\s+becomes?\s+none\b", consequent, re.IGNORECASE):
            observations.append(self._observation("return_value", "contains", None, "semantic"))

        deduped: list[dict[str, Any]] = []
        for item in observations:
            if item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _observation(channel: str, relation: str, value: Any, fidelity: str) -> dict[str, Any]:
        return {
            "channel": channel,
            "relation": relation,
            "value": value,
            "polarity": (
                "negative"
                if relation in {"excludes", "not_contains", "not_equals"}
                else "positive"
            ),
            "fidelity": fidelity,
        }

    def _compile_negative(self, atom: RequirementAtom) -> ConditionalObligation | None:
        lowered = atom.text.lower()
        separator_match = re.search(r"\bno\s+separator\s+is\s+inserted\b", lowered)
        if separator_match:
            return ConditionalObligation(
                obligation_id=f"{atom.requirement_id}.N01",
                parent_requirement_id=atom.requirement_id,
                trigger="always",
                precondition={"kind": "always"},
                observable_channel="stdout",
                comparison_relation="not_contains",
                expected_value="separator",
                polarity="negative",
                observation_fidelity="exact_text",
                verification_method="property_test",
                source_fragment=atom.source_fragment,
                witness_class="no_separator_output",
            )
        close_match = re.search(r"\b(?:must|shall)\s+not\s+close\b", lowered)
        if close_match:
            return ConditionalObligation(
                obligation_id=f"{atom.requirement_id}.N01",
                parent_requirement_id=atom.requirement_id,
                trigger="always",
                precondition={"kind": "always"},
                observable_channel="resource_state",
                comparison_relation="not_equals",
                expected_value="closed",
                polarity="negative",
                observation_fidelity="semantic",
                verification_method=atom.verification_method,
                source_fragment=atom.source_fragment,
                witness_class="resource_remains_open",
            )
        return ConditionalObligation(
            obligation_id=f"{atom.requirement_id}.N01",
            parent_requirement_id=atom.requirement_id,
            trigger="always",
            precondition={"kind": "always"},
            observable_channel="behavior",
            comparison_relation="excludes",
            expected_value=atom.text,
            polarity="negative",
            observation_fidelity="semantic",
            verification_method=atom.verification_method,
            source_fragment=atom.source_fragment,
        )

    def _compile_coverage_directive(
        self,
        atom: RequirementAtom,
        obligations: list[ConditionalObligation],
    ) -> CoverageDirective:
        witness_classes = self._coverage_witness_classes(atom.text)
        referenced = [
            obligation.obligation_id
            for obligation in obligations
            if obligation.witness_class in witness_classes
        ]
        return CoverageDirective(
            directive_id=f"{atom.requirement_id}.C01",
            parent_requirement_id=atom.requirement_id,
            referenced_obligation_ids=referenced,
            witness_classes=witness_classes,
            source_fragment=atom.source_fragment,
        )

    def _coverage_witness_classes(self, text: str) -> list[str]:
        lowered = text.lower()
        witnesses: list[str] = []
        patterns = (
            ("invalid_positive_integer", r"invalid\s+[^,]*sizes?|non[- ]integer"),
            ("empty_input", r"empty\s+(?:files?|inputs?)"),
            ("utf8_decode_failure", r"not\s+valid\s+utf[- ]?8|invalid\s+utf[- ]?8"),
            ("malformed_record", r"malformed\s+(?:rows?|records?)"),
            ("missing_field", r"missing\s+(?:fields?|columns?)"),
        )
        for witness, pattern in patterns:
            if re.search(pattern, lowered):
                witnesses.append(witness)
        return witnesses

    @staticmethod
    def _predicate_contract(trigger: str) -> tuple[str, dict[str, Any]]:
        lowered = " ".join(trigger.lower().replace("-", " ").split())
        taxonomy = (
            ("empty_input", r"\b(?:file|input|field_order|collection)\s+is\s+empty\b", {"kind": "empty_input"}),
            (
                "numeric_argument_exceeds_input_length",
                r"\b(?:chunk\s+)?size\s+exceeds?\s+(?:the\s+)?input\s+length\b",
                {"kind": "numeric_argument_exceeds_input_length"},
            ),
            ("invalid_argument_count", r"\bnot\s+exactly\s+\w+\s+arguments?\b", {"kind": "invalid_argument_count"}),
            (
                "invalid_positive_integer",
                r"\bnot\s+a\s+valid\s+positive\s+integer\b|\bnon[- ]?integer\b|\binvalid\s+.*size\b",
                {"kind": "invalid_positive_integer"},
            ),
            ("file_read_failure", r"\bfile\s+(?:reading|read)\s+fails?\b", {"kind": "file_read_failure"}),
            ("utf8_decode_failure", r"\b(?:file\s+)?decod(?:e|ing)\s+fails?\b|\binvalid\s+utf\s*8\b", {"kind": "utf8_decode_failure"}),
            ("zero_value", r"\b(?:shift|value|size)\s+is\s+zero\b", {"kind": "zero_value"}),
            ("missing_field", r"\bfield\s+is\s+missing\b", {"kind": "missing_field"}),
            ("invalid_type", r"\bnot\s+a\s+list\s+of\s+str\b", {"kind": "invalid_type", "expected_type": "list[str]"}),
        )
        for witness_class, pattern, precondition in taxonomy:
            if re.search(pattern, lowered):
                return witness_class, dict(precondition)
        return "", {"kind": "textual_precondition", "text": trigger}

    @staticmethod
    def _looks_compound(antecedent: str) -> bool:
        return bool(re.search(r"\bor\b|/", antecedent, re.IGNORECASE))

    @staticmethod
    def _looks_like_compound_conditional(text: str) -> bool:
        return bool(
            re.match(r"^(?:deterministic\s+failure:\s*)?(?:if|when|unless)\b", text, re.IGNORECASE)
            and re.search(r"\bor\b|/", text, re.IGNORECASE)
        )

    @staticmethod
    def _is_explicit_negative(text: str) -> bool:
        return bool(
            re.match(r"^(?:no|never|without)\b", text, re.IGNORECASE)
            or re.search(r"\b(?:must|shall)\s+not\b", text, re.IGNORECASE)
        )

    @staticmethod
    def _compound_issue(atom: RequirementAtom) -> ConditionalNormalizationIssue | None:
        if atom.strength not in {"hard", "universal"}:
            return None
        return ConditionalNormalizationIssue(
            parent_requirement_id=atom.requirement_id,
            source_fragment=atom.source_fragment,
            reason="hard_compound_conditional_not_safely_normalized",
        )
