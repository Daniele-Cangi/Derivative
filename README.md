# Derivative

Derivative is a CLI reasoning engine with an execution-grounded substrate.

It now includes **Forge**, a production-oriented orchestrator that turns natural-language software requirements into:
- verified packaged artifacts, or
- infeasibility/validation-failed terminal outcomes with typed evidence.

## Core Modules

### Derivative substrate
- `core/substrate.py`: cognitive framing with installed computational lenses.
- `core/kernel.py`: synthesis over framings with execution grounding.
- `core/execution_loop.py`: code execution loop, contradiction/infeasibility handling.
- `core/obligation_compiler.py`: obligation schema extraction/evaluation.
- `audit/trail.py`: persistent audit trail.
- `memory/delta.py`, `memory/gene_pool.py`: memory + lineage persistence.
- `core/workspace.py`: artifact emission/export.

### Forge pipeline
- `core/forge/requirement_compiler.py`
- `core/forge/planner_stage.py`
- `core/forge/coder_stage.py`
- `core/forge/validator_stage.py`
- `core/forge/packaging_stage.py`
- `forge.py` (thin orchestrator)

## Forge Execution Flow

`RequirementCompiler -> PlannerStage -> (InfeasibilityCertificate | CoderStage -> ValidatorStage -> (validation_failed | PackagingStage))`

Terminal statuses are normalized:
- `verified`
- `infeasible_proven`
- `validation_failed`

## Requirement Preservation and Coverage Gate

Forge preserves atomic requirement units end-to-end.

### Requirement atoms
Each requirement is compiled into typed atoms with:
- stable `requirement_id`
- `text`
- `category` (`functional`, `non_functional`, `quality`, `validation`, `universal_constraint`, `ambiguity`)
- `strength` (`hard`, `soft`, `universal`, `ambiguous`)
- `source_fragment`

### Propagation
- `BuildSpec.requirement_atoms` carries atomic requirements.
- `AcceptanceCriterion.requirement_ids` links acceptance to atoms.
- `FeasiblePlan.requirement_coverage` maps each requirement to files/tests/acceptance.
- `PlanTest.requirement_ids` keeps test-to-requirement traceability.
- `CodeArtifact` provenance includes `requirement:<id>` tags.

### Validator hard gates
Validation fails closed if:
- an atomic requirement is omitted downstream (`semantic_omission`)
- requirement coverage is missing (`missing_requirement_coverage`)
- a universal/absolute constraint is unproven (`universal_constraint_unproven`)
- tests are superficial/non-semantic (`non_semantic_test`, `fake_acceptance_coverage`)

## Validation Layers

`ValidatorStage` enforces 3 layers:
1. syntax/import/build/run checks
2. obligations/tests/acceptance checks
3. adversarial checks (manifest/provenance/entrypoint/superficiality)

A build is verified only if all 3 layers pass.

## CLI Usage

### Derivative CLI
```bash
python derivative.py "Given a problem statement..."
python derivative.py --audit
python derivative.py --memory
python derivative.py --lenses
```

### Forge CLI
```bash
python forge.py "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, flags contracts expiring in less than 90 days, writes a summary CSV, and includes tests."
```

Expected output style is prose and always includes:
- terminal `Status`
- what executed/passed/failed
- artifact path
- execution time

## Generated Artifacts

- Forge run artifacts: `generated_artifacts/forge_runs/`
  - `build_spec.json`
  - `feasible_plan.json` or `infeasibility_certificate.json`
  - `code_artifact.json` (if feasible)
  - `validation_artifact.json` (if coded)
  - `packaged_artifact.json` (if verified)

- Verified packages: `generated_artifacts/forge_packages/`
  - source/test files
  - `forge_package_manifest.json`
  - validation evidence snapshots

## Setup

```bash
python -m pip install -r requirements.txt
```

## Tests

Run all tests:
```bash
python -B -m pytest -q -p no:cacheprovider
```

Key Forge tests include:
- `tests/test_forge_planner_stage.py`
- `tests/test_forge_coder_stage.py`
- `tests/test_forge_validator_stage.py`
- `tests/test_forge_packaging_stage.py`
- `tests/test_forge_orchestration.py`
- `tests/test_forge_requirement_preservation.py`
