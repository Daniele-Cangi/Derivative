"""File confinement and content checks at artifact stage boundaries."""

import hashlib
from pathlib import Path, PureWindowsPath
from typing import Iterable

from core.forge.contracts import GeneratedFile


class ArtifactPathError(ValueError):
    """An artifact does not declare distinct, confined relative file paths."""


def artifact_file_targets(
    files: Iterable[GeneratedFile],
    root: Path,
    reserved_names: Iterable[str] = (),
) -> dict[str, Path]:
    """Check the entire file set before any file is written."""
    root = root.resolve()
    reserved = {name.casefold() for name in reserved_names}
    targets: dict[str, Path] = {}
    names: set[str] = set()
    device_names = {"con", "prn", "aux", "nul", "conin$", "conout$"} | {
        prefix + number for prefix in ("com", "lpt") for number in "123456789¹²³"
    }
    for generated in files:
        path = generated.path
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or ":" in path
            or "\x00" in path
            or PureWindowsPath(path).drive
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or any(ord(char) < 32 or char in '<>"|?*' for char in path)
            or any(
                part.endswith((".", " "))
                or part.partition(".")[0].rstrip(" ").casefold() in device_names
                for part in path.split("/")
            )
        ):
            raise ArtifactPathError(f"Invalid artifact relative path: {path!r}")
        name = path.casefold()
        if name in names or name.split("/")[0] in reserved:
            raise ArtifactPathError(f"Duplicate or reserved artifact path: {path!r}")
        target = root / path
        if target.resolve() != target or not target.is_relative_to(root):
            raise ArtifactPathError(f"Artifact path is not confined: {path!r}")
        if target.is_symlink():
            raise ArtifactPathError(f"Artifact file is a symbolic link: {path!r}")
        names.add(name)
        targets[path] = target
    for name in names:
        parts = name.split("/")
        if any("/".join(parts[:index]) in names for index in range(1, len(parts))):
            raise ArtifactPathError(f"Artifact file/directory conflict: {name!r}")
    return targets


def artifact_file_hashes(materialized: dict[str, Path]) -> dict[str, str]:
    """Snapshot actual written bytes, including platform newline conversion."""
    return {
        path: hashlib.sha256(target.read_bytes()).hexdigest()
        for path, target in materialized.items()
    }


def artifact_file_changes(
    materialized: dict[str, Path], expected: dict[str, str], root: Path,
) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    root = root.resolve()
    for path, digest in expected.items():
        target = materialized[path]
        try:
            if target.is_symlink() or target.resolve() != target or not target.is_relative_to(root):
                reason = "path_changed"
            elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                reason = "content_changed"
            else:
                continue
        except (OSError, ValueError):
            reason = "file_unavailable"
        changes.append({"path": path, "reason": reason})
    return changes
