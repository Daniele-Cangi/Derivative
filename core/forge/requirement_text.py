"""Normalize prose without changing the values of quoted contract literals."""
import re


QUOTED_LITERAL_PATTERN = r"'(?:[^'\\]|\\[\s\S])*'|\"(?:[^\"\\]|\\[\s\S])*\""
_QUOTED_LITERAL = re.compile(rf"(?<!\w)(?:{QUOTED_LITERAL_PATTERN})")


def normalize_requirement_text(text: str) -> str:
    chunks: list[str] = []
    start = 0
    for match in _QUOTED_LITERAL.finditer(text):
        chunks.append(re.sub(r"\s+", " ", text[start:match.start()]))
        chunks.append(match.group())
        start = match.end()
    chunks.append(re.sub(r"\s+", " ", text[start:]))
    return "".join(chunks).strip()


def requirement_clause_key(text: str) -> str:
    """Deduplicate prose case-insensitively, retaining case inside literal values."""
    chunks: list[str] = []
    start = 0
    for match in _QUOTED_LITERAL.finditer(text):
        chunks.extend((text[start:match.start()].lower(), match.group()))
        start = match.end()
    chunks.append(text[start:].lower())
    return "".join(chunks)


def mask_quoted_literals(text: str) -> str:
    """Preserve offsets while hiding literal contents from prose recognizers."""
    return _QUOTED_LITERAL.sub(lambda match: "~" * len(match.group()), text)


def explicit_cli_argument_index(requirement: str, role_pattern: str) -> int | None:
    """Resolve one role to one user argv slot without borrowing another binding.

    Both role-before-index and index-before-role forms must stay within the
    same declaration. Conflicting indices remain unknown rather than picking
    the first or last match. Quoted output examples are not declarations.
    """
    text = mask_quoted_literals(requirement)
    role = rf"\b(?:{role_pattern})\b"
    argv = r"\bargv\s*\[\s*(?P<index>\d+)\s*\]"
    bridge = r"(?:(?!\bargv\s*\[|\b(?:and|or)\b|[.,;:!?\n~]).){0,80}?"
    indexes = {
        int(match.group("index"))
        for pattern in (rf"{role}{bridge}{argv}", rf"{argv}{bridge}{role}")
        for match in re.finditer(pattern, text, re.IGNORECASE)
    }
    if len(indexes) != 1:
        return None
    index = indexes.pop()
    return index - 1 if index > 0 else None


def split_requirement_text(text: str, boundary: re.Pattern[str]) -> list[str]:
    masked = mask_quoted_literals(text)
    chunks: list[str] = []
    start = 0
    for match in boundary.finditer(masked):
        chunks.append(text[start:match.start()])
        start = match.end()
    chunks.append(text[start:])
    return chunks


def explicit_test_coverage_target(requirement: str) -> float | None:
    text = mask_quoted_literals(requirement)
    matches = re.finditer(
        r"\b(?:test|code)\s+coverage\s+"
        r"(?:(?:of|at\s+least|minimum(?:\s+of)?|target(?:\s+of)?|"
        r"must\s+be(?:\s+at\s+least)?)\s+)?"
        r"(?P<percent>\d+(?:\.\d+)?)\s*(?:%|percent\b)",
        text, re.IGNORECASE,
    )
    targets = [
        float(match.group("percent")) / 100 for match in matches
        if not prose_mention_is_negated(text, match.start(), match.end())
    ]
    return max(targets) if targets else None


def prose_mention_is_negated(text: str, start: int, end: int) -> bool:
    """Recognize local rejections, not arbitrary natural-language implication.

    Callers mask literal values first. Coordinated feature lists retain their
    polarity; a new directive or contrasted clause starts a separate scope.
    """
    prefix = re.split(r"[.;:!?]|\b(?:but|however|whereas)\b", text[:start], flags=re.IGNORECASE)[-1]
    boundaries = list(re.finditer(r"(?:\b(?:and|then)|,)\s+", prefix, re.IGNORECASE))
    for boundary in reversed(boundaries):
        local_prefix = prefix[boundary.end():]
        # A subject may stand between the coordinator and the new predicate:
        # "and authentication must use JWT", or "integration tests are required".
        # Keep the local prefix (including any local negation), not just the text
        # after the verb. A bare coordinated noun list still inherits polarity.
        new_predicate = re.search(
            r"\b(?:requires?|includes?|provides?|uses?|enforces?|demands?|"
            r"adds?|enables?|implements?|exposes?|offers?|runs?)\b",
            local_prefix, re.IGNORECASE,
        )
        required_suffix = re.match(
            r"\s+(?:(?:authentication|auth)\s+)?(?:is|are)\s+"
            r"(?:required|needed|mandatory)\b", text[end:], re.IGNORECASE,
        )
        if new_predicate or required_suffix:
            prefix = local_prefix
            break
    prefix = re.sub(r"\bnot\s+only\b", "", prefix, flags=re.IGNORECASE)
    negated_prefix = re.search(
        r"\b(?:no|without|neither|nor|never|forbid(?:s|den)?|"
        r"(?:do|does|must|is|are)n['’]t|"
        r"(?:must|shall|may|does|do|is|are)\s+not)\b",
        prefix, re.IGNORECASE,
    )
    negated_suffix = re.match(
        r"\s+(?:(?:authentication|auth)\s+)?"
        r"(?:(?:is|are)\s+not\s+|(?:is|are)n['’]t\s+|"
        r"(?:should|must|shall|may|will|would|can|could|need)\s+not\s+be\s+|"
        r"(?:should|must|would|could)n['’]t\s+be\s+)"
        r"(?:required|needed|mandatory)\b|"
        r"\s+(?:is|are)\s+(?:optional|unnecessary)\b",
        text[end:], re.IGNORECASE,
    )
    bare_rejection = re.search(r"\bnot\s+(?:an?\s+)?$", prefix, re.IGNORECASE)
    return bool(negated_prefix or bare_rejection or negated_suffix)
