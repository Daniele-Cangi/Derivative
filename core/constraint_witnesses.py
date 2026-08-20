import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FiniteWitnessContradiction:
    contract_id: str
    witness_code_point: str
    witness_name: str
    mapping_operation: str
    mapped_code_points: tuple[str, ...]
    input_length: int
    mapped_length: int
    message: str

    def to_evidence(self) -> dict[str, Any]:
        return asdict(self)


def finite_witness_contradictions(
    problem: str,
) -> list[FiniteWitnessContradiction]:
    return _unicode_case_cardinality_contradictions(problem)


def _unicode_case_cardinality_contradictions(
    problem: str,
) -> list[FiniteWitnessContradiction]:
    normalized = " ".join(problem.lower().split())
    universal_unicode_letters = re.search(
        r"\b(?:each|every|all)\s+unicode\s+letters?\b",
        normalized,
    )
    case_inversion = bool(
        re.search(r"\bcase\s+invert(?:ed|s|ing|ion)?\b", normalized)
        or re.search(r"\binvert(?:ed|s|ing)?\s+(?:its\s+)?case\b", normalized)
    )
    fixed_length = bool(
        re.search(r"\bsame\s+length\s+as\s+(?:the\s+)?input\b", normalized)
        or re.search(r"\bpreserv(?:e|es|ed|ing)\b.{0,30}\blength\b", normalized)
        or re.search(r"\blength\b.{0,30}\b(?:unchanged|preserved)\b", normalized)
    )
    if not (universal_unicode_letters and case_inversion and fixed_length):
        return []

    for witness in ("\u0130", "\u00df", "\ufb03"):
        if witness.isupper():
            operation = "lower"
            mapped = witness.lower()
        elif witness.islower():
            operation = "upper"
            mapped = witness.upper()
        else:
            continue
        if len(mapped) == 1:
            continue
        code_point = f"U+{ord(witness):04X}"
        witness_name = unicodedata.name(witness, "UNNAMED")
        mapped_code_points = tuple(f"U+{ord(value):04X}" for value in mapped)
        return [
            FiniteWitnessContradiction(
                contract_id="unicode_case_cardinality",
                witness_code_point=code_point,
                witness_name=witness_name,
                mapping_operation=operation,
                mapped_code_points=mapped_code_points,
                input_length=1,
                mapped_length=len(mapped),
                message=(
                    f"INFEASIBLE: finite Unicode witness {code_point} {witness_name} "
                    f"has length 1 but its Python {operation}case mapping contains "
                    f"{len(mapped)} code points ({', '.join(mapped_code_points)}). "
                    "Requiring case inversion for every Unicode letter and an output "
                    "with the same length cannot both hold."
                ),
            )
        ]
    return []
