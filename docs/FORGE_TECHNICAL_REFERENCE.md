# Forge Technical Reference

This document describes the execution contracts behind Forge. For first use, start with the [project README](../README.md). For architecture-loading details, see [Derivative and Forge Architecture Boundary](DERIVATIVE_FORGE_ARCHITECTURE.md).

## Pipeline Contracts

Forge stages exchange typed dataclasses from `core/forge/contracts.py`.

### BuildSpec

`RequirementCompiler` converts the original requirement into:

- normalized requirement text;
- atomic requirement units with stable IDs, source fragments, categories, and strengths;
- functional goals and non-functional constraints;
- ambiguity flags;
- acceptance and obligation contracts;
- a public module/symbol contract when declared;
- a quality contract;
- target artifact type and risk hints.

Universal language such as `every`, `all`, `any`, `guarantee`, `exactly`, `arbitrary`, `malformed`, and `invalid` is preserved. Underspecified behavior is marked as ambiguous instead of receiving invented semantics.

### PlannerStageOutput

`PlannerStage` uses the Derivative `CognitiveSubstrate` and `ReasoningKernel`. It returns exactly one of:

- `FeasiblePlan`, with architecture, planned files, interfaces, tests, obligations, validation strategy, implementation blueprint, and requirement coverage;
- `InfeasibilityCertificate`, with contradictions, violated obligations, execution evidence, minimal relaxations, and `terminal_status="infeasible_proven"`.

A proven contradiction is not represented as generic failure or non-convergence.

### CodeArtifact

`CoderStage` expands a feasible plan into generated source, tests, runnable entrypoints, a manifest, and provenance. Every planned file is linked to the plan element, capability, interface, test requirement, and requirement IDs that produced it.

The coder does not decide that the artifact is correct.

### ValidationArtifact

`ValidatorStage` returns:

- `passed`;
- explicit failures and failure signatures;
- structured evidence;
- metrics;
- terminal failure category when applicable.

A passing artifact has `failure_category=None`. Evidence remains populated for both passing and failing results.

### PackagedArtifact

`PackagingStage` accepts only a passing, integrity-sealed `ValidationArtifact`. Its manifest records build, plan, artifact, and package IDs; evidence paths; code and manifest digests; behavioral-contract, validated-artifact, and validation-artifact seals; the canonical validation receipt digest; and `terminal_status="verified"`.

## Requirement Preservation

Requirement traceability is a hard gate:

```text
original text
  -> RequirementAtom
  -> AcceptanceCriterion / ObligationContract
  -> FeasiblePlan.requirement_coverage
  -> planned source and tests
  -> CodeArtifact provenance
  -> executed assertion evidence
  -> ValidationArtifact
```

Validation fails closed when an atom disappears, a hard requirement lacks acceptance coverage, or a universal requirement has no defensible proof. Relevant signatures include:

- `semantic_omission`;
- `missing_requirement_coverage`;
- `missing_acceptance_coverage`;
- `missing_requirement_assertion_evidence`;
- `universal_constraint_unproven`.

Tests containing only `assert True`, file-presence assertions, or assertions disconnected from target behavior do not count as semantic coverage.

## Quality Contracts

`QualityContract` encodes requested engineering depth:

- authentication level and plaintext-secret policy;
- rate-limit scope and persistence;
- schema versioning and audit trail;
- health endpoint and structured logging;
- test coverage target and integration-test requirement;
- derived overall quality level.

The compiler derives this contract from explicit language. For example, `production-grade`, `hashed API keys`, `persistent per-user rate limiting`, `audit trail`, and `structured JSON logging` become concrete obligations.

The coder must generate the requested implementation. The validator independently checks the result. It does not permit silent downgrade, such as SHA-256 fallback when bcrypt is required or an in-memory counter when persistent rate limiting is required.

A mismatch produces `quality_contract_violation`.

## Domain and Capability Composition

Current deterministic adapters:

- `cli`;
- `service`;
- `pipeline`;
- `library`.

Adapter selection follows typed plan structure and public contracts, not evolutionary seeds or raw keyword shortcuts. Plans outside the deterministic surface remain fail-closed in `local-only` mode and may use the complete Candidate Compiler in `hybrid` or `remote-only`.

Service plans use a typed `ImplementationBlueprint` built from `CapabilitySpec` records. Typical modules are:

- `src/service.py`: API and entrypoint;
- `src/domain.py`: request workflow;
- `src/storage.py`: persistence;
- `src/auth.py`: authentication;
- `src/rate_limit.py`: rate limiting;
- `src/audit.py`: audit events;
- `src/observability.py`: health and logging.

Pipeline plans similarly separate ingestion, watching, schema validation, quarantine, persistence, and observability when required.

Each capability declares its module, interfaces, dependencies, requirement IDs, and quality fields. Validation rejects absent modules, invalid imports, blueprint drift, manifest drift, and provenance mismatches.

## Candidate Compiler

When no deterministic adapter can cover the complete typed plan, model-backed modes use `CandidateCompiler`.

The model output is an untrusted complete-candidate transaction:

- every planned source and test file must be replaced;
- extra and omitted paths are rejected;
- path traversal is rejected;
- public module and symbol contracts must remain intact;
- executable and semantic preflight must pass;
- SHA-256 digests and provenance are rebuilt locally;
- the host-computed, schema-versioned behavioral-contract seal is embedded in candidate evidence and the rebuilt manifest;
- model-provided verification claims are ignored.

`local-only` never invokes the model fallback.

## Validation

A build is verified only when all three layers pass.

### Layer 1: Runtime

- materialize files into a temporary workspace;
- parse and import Python modules;
- verify declared entrypoints;
- run minimal entrypoint probes;
- execute generated tests;
- capture command, exit code, stdout, stderr, timing, and sandbox evidence.

### Layer 2: Contracts

- acceptance and obligation coverage;
- requirement-to-test mappings;
- public import contract;
- adapter and capability contract;
- quality contract;
- requirement-specific behavioral evidence;
- assertion-level semantic evidence.

The validator recomputes the canonical UTF-8/SHA-256 behavioral-contract seal from the plan. Candidate manifests and every material repair lineage record must declare the same seal; absence or mismatch fails closed.

The validator also seals the complete `CodeArtifact` it observed and its full `ValidationArtifact`. Trusted orchestration re-finalizes the validation seal after adding repair-effectiveness evidence. Before creating a package directory, `PackagingStage` recomputes every seal and refuses mismatched build, plan, artifact, contract, or validation identities. `validation_evidence.json` is written in the exact canonical encoding covered by the package manifest's SHA-256 receipt; that receipt also binds the artifact-manifest digest and package run ID.

### Layer 3: Adversarial

- missing files despite manifest claims;
- manifest, digest, and provenance mismatch;
- absent or superficial entrypoints;
- superficial source or tests;
- fake acceptance coverage;
- forbidden hidden files or undeclared paths.

Typical signatures include `syntax_error`, `import_failure`, `missing_entrypoint`, `missing_required_file`, `test_execution_failure`, `manifest_mismatch`, `provenance_mismatch`, `superficial_stub`, and `non_semantic_test`.

## Failure-Guided Repair

Validation evidence is compiled into a typed repair directive. The directive records:

- failure signatures;
- target files and functions;
- linked requirement IDs;
- missing assertion terms;
- allowed operations;
- prior artifact digest and repair lineage.

A repair must materially change source, tests, manifest, or provenance. Unsupported, inapplicable, or byte-equivalent revisions terminate as `validation_failed`. Each accepted material revision records the host-computed behavioral-contract seal in its backend evidence; validation rejects lineage bound to a different contract with `behavioral_contract_seal_mismatch`.

Model-backed repair can replace only allowlisted validator-targeted files. Every revision passes through all three validation layers again. Test-only failures do not authorize unrelated source redesign.

## Oracle Contract Preflight

Held-out and blind benchmarks use repository-external acceptance oracles. Before Forge execution, the harness is checked for contradictions with the frozen requirement and for unusable test machinery.

Current checks include:

- explicit regex versus invalid fixtures;
- finite Unicode and fixed-cardinality witnesses;
- public import contract consistency;
- sync and async context-manager binding validity;
- lexical availability and binding lifetime;
- dereference after a context binding that cannot produce a usable value.

An invalid harness terminates as benchmark-only `oracle_invalid`. Forge and model execution are skipped, the frozen files remain unchanged, and the mismatch location is persisted.

## Execution Isolation

Production orchestration requires the Docker backend. Every generated command runs in an ephemeral container with:

- `--network none`;
- read-only root filesystem;
- one writable bind mount containing only the temporary workspace;
- dropped Linux capabilities;
- `no-new-privileges`;
- memory, CPU, PID, tmpfs, output, and timeout limits;
- explicit environment allowlist.

The sandbox image contains Python, pytest, and bcrypt for the currently certified surface:

```bash
docker build --file Dockerfile.forge-sandbox --tag derivative-forge-sandbox:py311 .
```

If Docker or the image is unavailable, Forge returns `sandbox_unavailable`. Production orchestration never silently falls back to local execution. The local backend exists only for trusted tests and cannot authorize packaging through `run_forge`.

## Computational Lenses

Derivative libraries are substrate capabilities, not prompt decoration:

| Runtime | Role |
| --- | --- |
| SymPy | algebra, recurrences, limits, and symbolic obligations |
| NetworkX | graph enumeration, connectivity, and topology witnesses |
| Qiskit / Aer | circuit generation and executable quantum checks |
| Z3 | satisfiability and contradiction witnesses |
| pgmpy | probabilistic model support |
| DoWhy | causal model support |
| SciPy / Pint | numerical and unit-aware physical reasoning |
| OpenAI | candidate compilation and targeted repair, never final validation |

Scientific and model runtimes load lazily when the selected problem or execution mode requires them.

## Dependency Profiles

- `requirements/forge.txt`: minimal deterministic Forge host.
- `requirements/model.txt`: OpenAI-backed modes.
- `requirements/symbolic.txt`: SymPy.
- `requirements/topology.txt`: NetworkX.
- `requirements/formal.txt`: Z3.
- `requirements/probabilistic.txt`: pgmpy.
- `requirements/causal.txt`: DoWhy.
- `requirements/quantum.txt`: Qiskit and Aer.
- `requirements/physical.txt`: SciPy and Pint.
- `requirements/research.txt`: complete local Derivative substrate.
- `requirements/dev.txt`: repository test runner.
- `requirements/all.txt`: complete development environment.

Missing selected runtimes fail at their execution boundary. Importing Forge or displaying CLI help does not eagerly load optional scientific packages.

## Artifact Layout

```text
generated_artifacts/
|-- forge_runs/<timestamp>_<build-id>_<status>/
|   |-- build_spec.json
|   |-- feasible_plan.json | infeasibility_certificate.json
|   |-- code_artifact.json
|   |-- validation_artifact.json
|   `-- packaged_artifact.json
`-- forge_packages/<package-id>/
    |-- src/
    |-- tests/
    |-- validation_evidence.json
    |-- code_artifact_manifest_dump.json
    `-- forge_package_manifest.json
```

Partial and failed runs remain auditable. Only verified artifacts are packaged.

## Configuration

`local-only` needs no API key. Model-backed modes use the OpenAI Responses API:

```dotenv
OPENAI_API_KEY="your-api-key-here"
OPENAI_MODEL="gpt-4.1-mini"
```

Optional cost telemetry is enabled only by explicit pricing metadata:

```dotenv
OPENAI_INPUT_COST_PER_1M_TOKENS="..."
OPENAI_OUTPUT_COST_PER_1M_TOKENS="..."
```

Without configured pricing, cost is `null`, never an invented zero.

## Module Map

- `core/forge/contracts.py`: typed contracts.
- `core/forge/requirement_compiler.py`: requirement and quality compilation.
- `core/forge/planner_stage.py`: grounded planning and infeasibility routing.
- `core/forge/coder_stage.py`: plan expansion facade.
- `core/forge/candidate_compiler.py`: complete model-backed candidate transaction.
- `core/forge/candidate_preflight.py`: executable and semantic preflight.
- `core/forge/domains/`: deterministic adapters.
- `core/forge/capabilities/`: capability renderers.
- `core/forge/validator_stage.py`: validation facade.
- `core/forge/validation/`: runtime, contract, quality, and adversarial gates.
- `core/forge/repair.py`, `repair_evidence.py`, `repair_backend.py`: bounded repair.
- `core/forge/execution.py`: local and Docker execution policy.
- `core/forge/oracle_contract.py`: external harness preflight.
- `core/forge/packaging_stage.py`: verified packaging.
- `forge.py`: thin CLI and orchestration root.

Derivative substrate integration points are documented in [Architecture Boundary](DERIVATIVE_FORGE_ARCHITECTURE.md).

## Extending Forge

A new adapter, renderer, or passing generated test suite is not enough to extend the verified surface. Extensions must preserve typed intent, declare capabilities, provide independent executable evidence, maintain provenance, and pass the adversarial layer.

Read [Certified Extension Contract](CERTIFIED_EXTENSION_CONTRACT.md) before adding domains or verification primitives.
