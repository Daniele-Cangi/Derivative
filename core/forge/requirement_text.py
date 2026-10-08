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


def split_requirement_text(text: str, boundary: re.Pattern[str]) -> list[str]:
    masked = mask_quoted_literals(text)
    chunks: list[str] = []
    start = 0
    for match in boundary.finditer(masked):
        chunks.append(text[start:match.start()])
        start = match.end()
    chunks.append(text[start:])
    return chunks
