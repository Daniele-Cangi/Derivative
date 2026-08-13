# Derivative

**Derivative turns software requirements into verified artifacts — or explicit failure evidence.**

Derivative is an execution-grounded software synthesis engine. Its **Forge** pipeline compiles natural-language requirements into typed obligations and plans, produces candidate software, executes and validates that software, repairs against concrete failure evidence, and packages a result only when the declared validation gates pass.

The terminal outcomes are deliberately simple:

- `verified` — the artifact passed the required execution and validation gates and can be packaged;
- `infeasible_proven` — the requested requirements conflict or cannot be satisfied inside the declared model;
- `validation_failed` — a candidate was produced, but the evidence is not strong enough to call it verified.

The older Derivative reasoning substrate still provides framing, constraint, causal, symbolic, and execution-grounded reasoning. Forge is the product-facing software synthesis path built on top of that substrate.

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
- `core/forge/contracts.py`: typed build, plan, code, validation, quality, and packaging contracts.
- `core/forge/requirement_compiler.py`
- `core/forge/planner_stage.py`
- `core/forge/coder_stage.py`: thin plan-to-artifact facade.
- `core/forge/candidate_compiler.py`: OpenAI-backed complete-candidate fallback for uncovered plan capabilities.
- `core/forge/candidate_preflight.py`: executable and semantic preflight for complete candidate transactions.
- `core/forge/domains/`: deterministic CLI, service, and pipeline code-generation adapters plus registry.
- `core/forge/capabilities/`: composable capability renderers used by domain adapters.
- `core/forge/validator_stage.py`: thin validation orchestration facade.
- `core/forge/validation/`: runtime, obligation, capability-contract, quality-contract, and adversarial validation components.
- `core/forge/packaging_stage.py`
- `forge.py` (thin orchestrator)

## Libraries

The substrate and CLI rely on these packages:
- `sympy`: symbolic algebra and recurrence solving.
- `networkx`: graph and topology modeling/enumeration.
- `qiskit`, `qiskit-aer`: quantum circuit construction and simulation.
- `z3-solver`: satisfiability and constraint proving.
- `pgmpy`, `dowhy`: probabilistic/causal reasoning support.
- `scipy`, `pint`: scientific computation and unit-aware calculations.
- `typer`, `rich`: command-line interface and structured console output.
- `python-dotenv`: runtime environment loading.

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

## Quality Contracts and Domain Adapters

`RequirementCompiler` extracts a typed `QualityContract` for security, rate limiting, persistence, auditability, observability, and test depth. The contract is propagated through planning, recorded in the artifact manifest, and checked against generated source and executed tests before verification.

`CoderStage` selects a domain adapter from typed plan structure, never from evolutionary seeds or freeform generation state. Current adapters are:
- `cli`
- `service`
- `pipeline`

The selected adapter is recorded in `artifact_manifest.metadata.domain_adapter`. New domains must implement the same deterministic adapter contract instead of adding branches to `CoderStage`.

Each adapter also declares the concrete capabilities it implements. The obligation layer derives required capabilities from hard requirement atoms and compares them with the selected adapter independently of the artifact manifest. A target label such as `CLI` is therefore insufficient to certify an unrelated template; missing capabilities fail with `adapter_capability_mismatch`, while forged or stale manifest declarations fail with `adapter_capability_manifest_mismatch`.

When a deterministic adapter lacks required capabilities, `hybrid` and `remote-only` modes can route the typed plan to the Candidate Compiler. It must replace every planned source and test file as one allowlisted transaction; omitted files, extra paths, failed executable/semantic preflight, stale digests, or provenance mismatches reject the whole candidate. Manifest capability declarations, SHA-256 digests, transaction lineage, and provenance are rebuilt locally rather than accepted from model output. `local-only` never invokes this fallback and remains fail-closed.

## Capability Composition

For service builds, `PlannerStage` compiles a typed `ImplementationBlueprint` made of `CapabilitySpec` records. Each capability declares its module path, interfaces, dependencies, linked requirement IDs, quality fields, and deterministic configuration.

The service adapter composes these capabilities into separate modules:
- API and entrypoint (`src/service.py`)
- request workflow (`src/domain.py`)
- persistence (`src/storage.py`)
- authentication (`src/auth.py`)
- rate limiting (`src/rate_limit.py`)
- audit trail (`src/audit.py`)
- observability (`src/observability.py`)

Generated-file provenance records the producing capability, its dependencies, requirement IDs, and quality fields. The capability contract gate rejects missing modules, invalid dependency imports, blueprint/manifest drift, or provenance mismatches with `missing_capability` or `capability_contract_violation`.

### Validator hard gates
Validation fails closed if:
- an atomic requirement is omitted downstream (`semantic_omission`)
- requirement coverage is missing (`missing_requirement_coverage`)
- a universal/absolute constraint is unproven (`universal_constraint_unproven`)
- a planned capability is absent or inconsistent (`missing_capability`, `capability_contract_violation`)
- the selected domain adapter cannot implement the required behavior (`adapter_capability_mismatch`)
- tests are superficial/non-semantic (`non_semantic_test`, `fake_acceptance_coverage`)

## Validation Layers

`ValidatorStage` composes independent runtime, obligation, quality, and adversarial components and enforces 3 layers:
1. syntax/import/build/run checks
2. obligations/tests/acceptance checks
3. adversarial checks (manifest/provenance/entrypoint/superficiality)

A build is verified only if all 3 layers pass.

## Failure-Guided Repair

Coder retries are grounded in `ValidationArtifact` failure signatures and evidence. Forge compiles a typed repair directive, records target files and operations, and accepts a revised `CodeArtifact` only when its source, tests, manifest, or provenance actually changes. A repair that is unsupported, not applicable, or byte-equivalent terminates as `validation_failed`; it cannot consume a synthetic retry or reach packaging.

In `hybrid` or `remote-only` mode, Forge may ask the existing cognitive substrate and reasoning kernel for complete replacements of validator-targeted files. These revisions are untrusted candidates: path allowlists prevent unplanned file or manifest changes, repair lineage is persisted, and all three validation layers must pass again before packaging. `local-only` remains deterministic and uses canonical plan regeneration without requiring a remote model.

For an uncovered capability, Forge uses the stricter complete Candidate Compiler transaction instead of a partial repair. Preflight executes all generated tests and checks mapped semantic evidence before the candidate reaches `ValidatorStage`; the validator remains the final authority and can still reject it. Repository-held acceptance oracles are used only by the held-out benchmark and are never exposed to generation.

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

Live `hybrid` and `remote-only` reasoning use the OpenAI Responses API. Configure:

```dotenv
OPENAI_API_KEY="your-api-key-here"
OPENAI_MODEL="gpt-4.1-mini"
```

`local-only` does not require an API key.

## License

Derivative is released under the [MIT License](LICENSE).

## Tests

Run all tests:
```bash
python -B -m pytest -q -p no:cacheprovider
```

Run the Forge benchmark quality gate locally (same thresholds used in CI):
```bash
python forge_benchmark.py --preset extended --enforce-thresholds --min-status-accuracy 0.95 --min-verified-at-1 0.95 --max-false-verified-rate 0.00 --min-infeasible-detection-rate 1.00
```

Run the separate held-out benchmark with independent acceptance oracles:
```bash
python forge_heldout_benchmark.py
```

The held-out runner does not trust Forge's generated tests. For feasible cases it executes a repository-maintained oracle against the packaged artifact and reports `External Verified@1`, oracle pass rate, and external false-verified rate. This benchmark is intentionally not a CI gate until a stable external baseline is established.

Key Forge tests include:
- `tests/test_forge_planner_stage.py`
- `tests/test_forge_coder_stage.py`
- `tests/test_forge_validator_stage.py`
- `tests/test_forge_packaging_stage.py`
- `tests/test_forge_orchestration.py`
- `tests/test_forge_requirement_preservation.py`
- `tests/test_forge_heldout_benchmark.py`
