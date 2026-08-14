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
- `library`

The adapters are split into narrow domain profiles rather than one growing template. Current deterministic profiles include CSV business CLIs, JSON log and merge CLIs, scored-record JSON sorting, event services, telemetry/sales/JSONL pipelines, email normalization, largest-remainder allocation, Semantic Versioning precedence, and interval merging. Plans outside implemented capability profiles remain fail-closed locally and may use the complete Candidate Compiler transaction in `hybrid` or `remote-only` mode.

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

## Current Verification Status

The repository currently has the following verified baseline:

- 273 repository tests passing locally. The corresponding GitHub Actions run is required before this checkpoint is considered closed.
- A 30-case extended benchmark balanced across `verified`, `validation_failed`, and `infeasible_proven`, enforced as a CI quality gate.
- A held-out benchmark with repository-maintained acceptance oracles that execute independently against packaged artifacts.
- A sealed blind-v2 calibration bundle containing 10 cases: 6 expected verified builds, 2 expected validation failures, and 2 expected infeasibility proofs.

The original sealed blind-v2 run is preserved unchanged and scored 6/10 with an external false-verified rate of 0.0. Later post-fix replays and targeted oracle runs are regression evidence, not new blind evidence, because those requirements are now known to the development process. The latest targeted rerun of the three previously failing feasible cases (`B001`, `B004`, and `B005`) passed 3/3 with all external oracles, but this must not be reported as a fresh blind 10/10 result.

The blind manifest keeps the original expected Forge baseline digest and file count. Reports expose both expected and currently observed baseline metadata; any implementation change keeps `baseline_verified=false` unless the exact sealed implementation is used. The baseline is never silently resealed.

## Scope and Maturity

Forge is currently an advanced Python software-synthesis system, not yet a general-purpose repository coding agent.

Strongest supported areas:

- greenfield Python CLI, service, pipeline, and library artifacts;
- requirement preservation and executable acceptance coverage;
- deterministic domain profiles with model-backed fallback;
- conservative validation and explicit infeasibility proofs;
- audit, provenance, repair lineage, and verified packaging.

Current boundaries:

- modifying large existing repositories is not yet a first-class workflow;
- unknown domains still depend on model-generated complete candidate transactions;
- generated code runs in temporary workspaces but not yet inside an OS-level container or microVM sandbox;
- packaging does not yet provide wheels, containers, lockfiles, SBOMs, or supply-chain attestations;
- semantic evidence combines execution with typed, AST, and domain-specific checks and therefore remains incomplete outside covered contracts;
- current public benchmarks are regression suites once their requirements have been used for implementation fixes.

## Development Roadmap

The current phase is evidence closure, not feature expansion. Forge will remain focused on its existing Python CLI, service, pipeline, and library surface until that surface has been independently evaluated.

Current phase priorities:

1. Freeze a genuinely new blind-v3 dataset and its independent acceptance oracles before Forge executes any requirement.
2. Run generated code inside an isolated sandbox or container with explicit filesystem, network, timeout, process, and resource policies.
3. Measure external Verified@1, success after repair, false-verified rate, repair count, execution latency, and model cost.
4. Correct only structural failures exposed by the frozen blind evaluation. A blind case must not cause a new case-specific template or weakened verification gate.
5. Preserve capability composition and verification contracts as the only accepted extension mechanism inside the current surface.

This phase is complete only when:

- the blind-v3 manifest, dataset, and oracle digests were frozen before the first Forge execution;
- generated code and acceptance tests executed under the declared isolation policy;
- the complete metric report and per-case evidence were persisted;
- the external false-verified rate remained zero;
- every post-blind code change is traceable to a structural failure class rather than a benchmark-specific implementation;
- a final replay uses the unchanged frozen dataset and oracles and is clearly labeled as post-fix regression evidence.

Existing-repository mode, additional languages, frontend generation, new broad domain adapters, wheel/container distribution packaging, SBOM generation, and dependency auditing are explicitly deferred. They are not acceptance criteria for closing the current Forge phase.

The intended strategy remains a general truth-preserving substrate with incrementally certified software domains. Forge expands only after the current system has demonstrated the same obligation, execution, adversarial-validation, and fail-closed guarantees on independent evidence.

## Tests

Run all tests:
```bash
python -B -m pytest -q -p no:cacheprovider tests
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

Run the sealed blind-v2 benchmark against the exact Forge baseline recorded in its manifest:
```bash
python forge_blind_benchmark.py --mode hybrid
```

`benchmarks/blind_v2/manifest.json` locks the dataset, every external oracle, and the protected Forge source baseline with SHA-256 digests. Loading fails before execution if any locked input or Forge implementation file changed. The bundled calibration set includes contracts derived from Semantic Versioning 2.0.0, RFC 6901, and RFC 3339 plus independently specified behavioral cases. Forge receives only each natural-language requirement; the acceptance oracle runs afterward against the packaged artifact. The first run is preserved in `benchmarks/blind_v2/baseline_result.json`. For later genuinely private evaluations, pass an independently maintained bundle with `--manifest` and do not commit it into the repository.

Freeze a new externally authored blind bundle before its first Forge execution:
```bash
python forge_blind_freeze.py PATH_TO_PRIVATE_BUNDLE \
  --bundle-id forge-blind-v3-external-001 \
  --producer "Independent benchmark producer" \
  --requirements-origin "Requirements authored outside the Forge process" \
  --oracle-origin "Independent black-box acceptance suite" \
  --declaration "Requirements and oracles were finalized before Forge execution" \
  --source-url https://example.com/benchmark-spec
```

The freezer accepts an existing `cases.json` and its referenced oracle files; it never generates either. It writes a schema-v2 manifest once and refuses overwrite. The manifest records UTC freeze time, explicit provenance attestations, optional HTTPS source URLs, and SHA-256 digests for the dataset, each oracle, and the protected Forge baseline. The exact private bundle can then be executed with `python forge_blind_benchmark.py --manifest PATH_TO_PRIVATE_BUNDLE/manifest.json`. See `benchmarks/blind_v3/README.md` for the intake protocol. Provenance is an auditable declaration rather than cryptographic proof, so the external producer remains responsible for keeping inputs hidden until freeze.

Key Forge tests include:
- `tests/test_forge_planner_stage.py`
- `tests/test_forge_coder_stage.py`
- `tests/test_forge_validator_stage.py`
- `tests/test_forge_packaging_stage.py`
- `tests/test_forge_orchestration.py`
- `tests/test_forge_requirement_preservation.py`
- `tests/test_forge_heldout_benchmark.py`
- `tests/test_forge_blind_benchmark.py`
- `tests/test_forge_blind_freeze.py`
