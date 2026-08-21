import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_ROOT = REPOSITORY_ROOT / "requirements"
EXPECTED_COMPLETE_SET = {
    "dowhy",
    "networkx",
    "openai",
    "pgmpy",
    "pint",
    "pytest",
    "python-dotenv",
    "qiskit",
    "qiskit-aer",
    "rich",
    "scipy",
    "sympy",
    "typer",
    "z3-solver",
}


def _dependency_name(requirement: str) -> str:
    name = re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def _resolve_requirements(path: Path, visited: set[Path] | None = None) -> set[str]:
    resolved_path = path.resolve()
    visited = set() if visited is None else visited
    if resolved_path in visited:
        return set()
    visited.add(resolved_path)

    dependencies: set[str] = set()
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            included_path = resolved_path.parent / line.removeprefix("-r ").strip()
            dependencies.update(_resolve_requirements(included_path, visited))
            continue
        dependencies.add(_dependency_name(line))
    return dependencies


def test_dependency_profile_files_are_explicit_and_complete():
    expected_profiles = {
        "all.txt",
        "base.txt",
        "causal.txt",
        "dev.txt",
        "forge.txt",
        "formal.txt",
        "model.txt",
        "physical.txt",
        "probabilistic.txt",
        "quantum.txt",
        "research.txt",
        "symbolic.txt",
        "topology.txt",
    }

    assert {path.name for path in REQUIREMENTS_ROOT.glob("*.txt")} == expected_profiles


def test_forge_profile_contains_only_the_minimal_host_runtime():
    expected_base = {"python-dotenv", "rich", "typer"}

    assert _resolve_requirements(REQUIREMENTS_ROOT / "base.txt") == expected_base
    assert _resolve_requirements(REQUIREMENTS_ROOT / "forge.txt") == expected_base


def test_capability_profiles_map_to_their_runtime_packages():
    expected = {
        "model.txt": {"openai"},
        "symbolic.txt": {"sympy"},
        "topology.txt": {"networkx"},
        "formal.txt": {"z3-solver"},
        "probabilistic.txt": {"pgmpy"},
        "causal.txt": {"dowhy"},
        "quantum.txt": {"qiskit", "qiskit-aer"},
        "physical.txt": {"pint", "scipy"},
    }

    for profile, packages in expected.items():
        assert _resolve_requirements(REQUIREMENTS_ROOT / profile) == packages


def test_aggregate_and_legacy_profiles_preserve_the_complete_environment():
    research = _resolve_requirements(REQUIREMENTS_ROOT / "research.txt")
    complete = _resolve_requirements(REQUIREMENTS_ROOT / "all.txt")
    legacy = _resolve_requirements(REPOSITORY_ROOT / "requirements.txt")

    assert "openai" not in research
    assert "pytest" not in research
    assert {"python-dotenv", "rich", "typer"}.issubset(research)
    assert complete == EXPECTED_COMPLETE_SET
    assert legacy == EXPECTED_COMPLETE_SET
