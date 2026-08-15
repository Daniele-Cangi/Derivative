# Blind v3 intake

Blind v3 inputs must be authored outside the Forge development process and frozen
before Forge receives any requirement. Frozen bundles are stored in dedicated
`external_*` directories; their requirements and oracles must not be used for
implementation changes until the first execution is complete.

The external benchmark producer prepares a private directory containing:

- `cases.json` using the held-out benchmark schema;
- one black-box oracle for every case expected to return `verified`;
- provenance text identifying the independent origin of requirements and oracles;
- optional HTTPS source references when the requirements derive from public specifications.

The producer then runs `forge_blind_freeze.py` once. The command writes a schema-v2
`manifest.json` containing the dataset, oracle, and protected Forge baseline
SHA-256 digests. An existing manifest is never overwritten. Any later change to a
sealed input or protected Forge source makes bundle loading fail before execution.

The provenance fields are explicit attestations, not cryptographic proof of
independent authorship. Operational independence still requires the external
producer to keep the cases and oracles hidden from the Forge development process
until the manifest has been frozen.

## Isolated one-shot producer

When a human external producer is unavailable, `forge_blind_produce.py` can use
the OpenAI Responses API as an isolated benchmark producer. It sends no Forge
source or generated artifact to the model. One stateless request creates only the
requirements and expected terminal states; separate stateless requests create
black-box oracles from the frozen requirement text. The producer validates the
case distribution and oracle syntax/semantic depth in a staging directory, calls
the normal schema-v2 freezer, and publishes the output directory only after the
manifest reloads successfully against the current Forge baseline.

Run this only from a clean, committed Forge baseline and use a new destination:

```bash
python forge_blind_produce.py benchmarks/blind_v3/external_001 \
  --bundle-id forge-blind-v3-external-001
```

The command refuses an existing destination and never prints requirement or
oracle contents before sealing. This provides process isolation from Forge, not
cryptographic proof that the model has never encountered related public material.

## First frozen run

`external_002` is the first bundle that reached Forge execution. It contains 12
cases with a frozen distribution of six `verified`, three `validation_failed`,
and three `infeasible_proven` expectations, plus independent black-box oracles
for the six expected verified cases. Requirements and oracles were sealed before
execution.

Two earlier workflow runs stopped during preflight and exposed no cases to Forge:

- run `31875840401` rejected `external_001` because Git changed dataset line endings;
- run `31876481677` rejected the initial `external_002` manifest because the Forge
  source fingerprint hashed platform-specific line endings.

The fingerprint algorithm was then changed to `canonical_lf_v1`, the manifest
metadata was migrated before any case execution, and run `31877186602` passed the
Linux integrity gate and executed the frozen suite once. The unmodified report is
preserved in `external_002/baseline_result.json`.

The first-run result is intentionally reported without retrospective thresholds:

- 3 of 12 cases passed; status accuracy was 0.333;
- external `Verified@1`, success after repair, and oracle pass rate were 0.000;
- external false-verified rate was 1.000;
- infeasibility detection rate was 0.000;
- 10 repairs were attempted, averaging 0.83 per case;
- median runtime was 1.72 seconds and P95 runtime was 2.36 seconds;
- the `local-only` run made no model requests and used zero model tokens.

These metrics are the immutable blind baseline, not a quality gate result. Any
later run is post-fix regression evidence and must not be described as a new blind
evaluation.

## Post-fix replay 001

Run `31879547356` executed the unchanged sealed inputs after structural fixes to
requirement preservation, universal fail-closed handling, and contradiction
detection. Its report is preserved in `external_002/post_fix_replay_001.json` and
is explicitly marked `execution_kind=post_fix_replay` with
`baseline_verified=false`.

The replay produced:

- 6 of 12 cases passed; status accuracy increased from 0.333 to 0.500;
- external false-verified rate decreased from 1.000 to 0.000;
- infeasibility detection increased from 0.000 to 1.000;
- external `Verified@1`, success after repair, and oracle pass rate remained 0.000;
- repairs decreased from 10 to 8;
- median runtime was 1.25 seconds and P95 runtime was 1.61 seconds.

This replay establishes that the first structural repair restored fail-closed
behavior but did not improve feasible build capability. It used `local-only`
execution and made no model requests, so it evaluates deterministic adapters and
verification rather than the model-backed candidate compiler.

## Post-fix replay 002 (hybrid)

Run `31879929684` executed the same sealed inputs through the OpenAI-backed
candidate compiler in `hybrid` mode. Its report is preserved in
`external_002/post_fix_replay_002_hybrid.json` with
`execution_kind=post_fix_replay` and `baseline_verified=false`.

The replay produced:

- 6 of 12 cases passed and status accuracy was 0.583;
- infeasibility detection remained 1.000;
- external `Verified@1`, success after repair, and oracle pass rate were 0.000;
- one internally verified feasible build failed its independent oracle, so the
  external false-verified rate was 1.000;
- 13 repairs were attempted across 38 model requests and 347,287 tokens;
- median runtime was 41.55 seconds and P95 runtime was 185.28 seconds;
- estimated model cost was unavailable because pricing metadata was not set.

The false verification was caused by a public package contract mismatch: the
requirement requested a `service` module exporting `hash_stream`, while planning
and generation substituted `library` as the package path. Internal generated
tests validated the substituted path, but the independent oracle correctly
failed to import `service.hash_stream`. The remaining feasible failures also
expose overly literal semantic evidence terms, coarse universal-constraint
classification, and repair routing that does not consistently invoke the
candidate compiler for semantic omissions. These are structural findings, not
justifications for benchmark-specific templates.

## Post-fix replay 003 (hybrid)

Run `31882743848` executed the unchanged sealed inputs after structural changes
to preserve exact public modules, distinguish property tests from absolute
universal proofs, accept AST-grounded entrypoint evidence, and route unsupported
adapter semantics to the candidate compiler. Its report is preserved in
`external_002/post_fix_replay_003_hybrid.json` with
`execution_kind=post_fix_replay` and `baseline_verified=false`.

The replay produced:

- 6 of 12 cases passed and status accuracy increased to 0.833;
- infeasibility detection remained 1.000;
- external `Verified@1` remained 0.000 and success after repair was 0.167;
- one of five executed independent oracles passed, for an oracle pass rate of
  0.200;
- five of six internally verified builds failed either an oracle or the expected
  terminal status, so external false-verified rate was 0.833;
- 11 repairs were attempted across 32 model requests and 299,800 tokens;
- median runtime was 44.79 seconds and P95 runtime was 240.55 seconds;
- estimated model cost remained unavailable because pricing metadata was not set.

The exact `service.hash_stream` public contract now passes all five independent
oracle tests. Four other feasible artifacts still substitute a generic module
for a public name declared as a library function, component, CLI tool, or
standalone function. The remaining terminal-status mismatch is a seeded
pseudo-random CLI whose algorithm is unspecified but was not treated as a
material ambiguity. These are the next structural contract-extraction gaps; the
result does not establish a new blind baseline.

## Post-fix replay 004 (hybrid)

Run `31885082173` executed the unchanged sealed inputs after generalizing public
module extraction across functions, components, CLI commands, and standalone
callables, and after treating an unspecified seeded pseudo-random algorithm as a
material ambiguity. Its report is preserved in
`external_002/post_fix_replay_004_hybrid.json` with
`execution_kind=post_fix_replay` and `baseline_verified=false`.

The replay produced:

- 6 of 12 cases passed and status accuracy was 0.667;
- infeasibility detection remained 1.000;
- external `Verified@1` remained 0.000 and success after repair was 0.333;
- two of four executed independent oracles passed, for an oracle pass rate of
  0.500;
- external false-verified rate improved from 0.833 to 0.667;
- 9 repairs were attempted across 30 model requests and 264,277 tokens;
- median runtime was 46.64 seconds and P95 runtime was 138.25 seconds;
- estimated model cost remained unavailable because pricing metadata was not set.

`invert_dictionary` and `filter_by_predicate` now preserve their public modules
and pass the independent oracles. The unspecified pseudo-random CLI now fails
closed as expected. Two requirements containing explicit `unprovable` and
`inherently ambiguous` language still became false `verified`, revealing a
material-ambiguity extraction gap. The `service.hash_stream` regression came
from path-based adapter selection overriding the typed library target when the
declared module happened to be named `service`.

The remaining V3-005 and V3-006 oracle failures are not clean implementation
signals. The frozen V3-005 oracle requires `SystemExit` rather than accepting a
non-zero return code and its redirect fixture does not capture the observed
stdout under pytest. The frozen V3-006 oracle comments `0d` as allowed while
also parametrizing it as an input that must raise `ValueError`. These oracle
files remain untouched so the sealed bundle and recorded metrics stay auditable.
