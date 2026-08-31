# Benchmark Evidence

This ledger separates immutable blind evidence from regression work. It is intentionally more detailed than the [project README](../README.md).

## Reporting Rules

- A blind bundle is frozen before Forge sees its requirement or oracle.
- Baseline reports are append-only and never rescored after implementation changes.
- A replay against a known bundle is labeled `post_fix_replay`, never a new blind score.
- Internal `verified` and external oracle acceptance are separate facts.
- Invalid oracles are reported as `oracle_invalid`, excluded with evidence, and never converted into passes.
- Schema-v3 adjudicated rates store `numerator`, `denominator`, and `value`; `value` is `null` when the denominator is zero.
- Legacy raw summaries retain their historical scalar schema, including `0.0` for some empty denominators. The ledger labels those fields as legacy and states the zero denominator instead of treating them as measured zero rates.
- Rates in this ledger include explicit numerators and denominators whenever the source receipt defines the population.
- Model cost is reported only when pricing metadata was configured for that run.

## Metric Definitions

- **Verified@1**: an expected verified artifact passed its first internal validation attempt.
- **External Verified@1**: first-attempt internal verification plus independent oracle acceptance.
- **Success after repair**: externally accepted repaired artifacts among expected verified cases that did not pass first attempt.
- **External false-verified rate**: internally verified artifacts rejected by an executable external oracle.
- **Infeasibility detection**: expected infeasible cases ending as `infeasible_proven`.
- **Invalid-benchmark rejection**: incoherent benchmark harnesses stopped before Forge execution.
- **Cost per externally accepted artifact**: configured model cost divided by externally accepted artifacts; `null` when the denominator is zero.
- **Repairs per successful build**: repair attempts divided by externally accepted successful artifacts.

## Repository Gates

At the current frozen V9 checkpoint (`4d8ee7d`):

- Linux/Python 3.11 Forge CI: **548 passed**;
- complete local Windows suite: **546 passed, 2 skipped**;
- full Docker-backed 30-case internal benchmark: all configured thresholds passed;
- CodeQL: passing.

These are repository regression gates, not independent blind proof. The corresponding receipts are [Forge CI run 33298732605](https://github.com/Daniele-Cangi/Derivative/actions/runs/33298732605) and [CodeQL run 33298732338](https://github.com/Daniele-Cangi/Derivative/actions/runs/33298732338).

At the V7 oracle-preflight checkpoint (historical):

- local suite: **470 passed, 2 skipped**;
- minimal Forge runtime workflow: passing;
- full Linux/Docker test and benchmark gate: passing;
- internal 30-case benchmark: deterministic regression evidence, not independent blind proof.

## Blind V5

Blind V5 was frozen before first execution and established the current evidence-closure discipline.

Immutable baseline:

- 12 cases;
- raw report `status_accuracy`: 4/12 (`0.417`);
- external Verified@1: 0/6 (`0.000`);
- external false-verified rate: 1/1 (`1.000`);
- infeasibility detection: 1/3 (`0.333`);
- model tokens: `680,399`;
- estimated model cost: `$1.83934`.

Later runs are isolated regression receipts. They are not combined into a synthetic current V5 score.

Selected outcomes:

- `V5-002`: verified after one repair; independent oracle 7/7; 64,234 tokens and four model requests.
- `V5-005`: verified; independent oracle passed.
- `V5-010` and `V5-012`: terminal `infeasible_proven`.
- `V5-001`: `oracle_invalid`; process-style `argv[0]` conflicted with an importable `main(argv)` contract.
- `V5-003`: `oracle_invalid`; universal Unicode case inversion conflicted with a fixed-length contract.
- `V5-004`: `oracle_invalid`; an invalid fixture contradicted its explicit regular expression and was rejected with zero model calls.
- `V5-006`: verified after two repairs; independent oracle 9/9; 116,861 tokens; estimated cost `$0.304336`; 190.22 seconds.

Sources:

- [Manifest](../benchmarks/blind_v5/external_001/manifest.json)
- [Immutable baseline](../benchmarks/blind_v5/external_001/baseline_result.json)
- [Evidence closure report](FORGE_V5_EVIDENCE_CLOSURE.md)
- Replay receipts are stored beside the baseline as `post_fix_replay_*.json`.

## Blind V6

Blind V6 was independently produced, frozen, and executed once against its sealed baseline.

Immutable raw report:

- 12 cases;
- raw report `status_accuracy`: 4/12 (`0.333`);
- external Verified@1: 0/6 (`0.000`);
- external false-verified rate: 2/2 (`1.000`);
- infeasibility detection: 0/3 (`0.000`);
- model tokens: `743,949`;
- estimated model cost: `$1.885866`.

Two independent label reviewers received no Forge output, generated code, failure signatures, oracle results, or baseline report. They agreed on six original labels and three corrected labels, and disagreed on three. A deterministic check rejected the reviewer consensus for V6-007 because its five-item example returned four items despite a same-length requirement.

The adjudicated receipt therefore includes eight cases, excludes three unresolved cases plus V6-007, and reports status accuracy 3/8 (`0.375`).

V6 predates schema-v3 typed public import contracts. No verified case has both a frozen oracle and schema-v3 `public_contract`, so definitive external Verified@1, oracle pass, repair success, false-verified, and infeasibility rates are `null`.

Receipt integrity:

- adjudicated receipt SHA-256: `8D7CF76BF927CC86CA3523762C920B8489D631888F2F7260974AA49135AEF96C`.

Sources:

- [Manifest](../benchmarks/blind_v6/external_001/manifest.json)
- [Immutable baseline](../benchmarks/blind_v6/external_001/baseline_result.json)
- [Requirement adjudication](../benchmarks/blind_v6/external_001/requirement_adjudication.json)
- [Adjudicated metrics](../benchmarks/blind_v6/external_001/adjudicated_metrics.json)

## Blind V7

Blind V7 is the first schema-v3 evaluation. A separate `gpt-4.1` producer created 12 cases with typed public import contracts, separate requirement and oracle reviews, and no Forge source or earlier blind case in its prompts.

Frozen bundle integrity:

- manifest SHA-256: `F84222978A21DC38C2534BE7BEF29A7A2F94BE0BE36CAA8E2AA6B8DF761CC80E`;
- dataset SHA-256: `DE39E16364468F727C7685823359B3945448AC82BB8A3C50A1E47EE27EBA7920`;
- protected Forge baseline SHA-256: `9EA720203B0FC6D07831F4DEAD08A351DC94C2991E04E280FD1541596BD45E22`.

### Immutable baseline

The [first baseline workflow](https://github.com/Daniele-Cangi/Derivative/actions/runs/32532939841) completed its measurement in Docker.

Raw results:

- claimed status matches: 4/12;
- raw report `status_accuracy`: 4/12 (`0.333`);
- external Verified@1: 0/6 (`0.000`);
- success after repair: 0/6 (`0.000`);
- legacy raw external false-verified field: `0.000` with 0 observed verified artifacts and denominator 0;
- schema-v3 adjudicated external false-verified value: `null` (0/0);
- infeasibility detection: 1/3 (`0.333`);
- model requests: `54`;
- model tokens: `761,049`;
- repairs: `17`;
- estimated model cost: `$2.048034`;
- median latency: `124.91s`;
- P95 latency: `245.62s`.

Workflow success means the measurement completed, not that Forge passed the benchmark.

Independent label adjudication found five valid labels, six invalid labels, and one unresolved label. All six corrections were `verified`: three Unicode requirements were objectively specified, and three impossible filtering predicates still described feasible programs whose correct output is always empty.

The adjudicated receipt includes 11 cases and reports status accuracy 0/11. External Verified@1 and success after repair are 0/5. External false-verification, oracle-pass, and infeasibility rates are `null` because their adjudicated denominators are zero.

### Public-import replay

The first full post-fix replay remained labeled `post_fix_replay` with `baseline_verified=false`.

Raw results:

- case passes: 3/12;
- raw status matches: 6/12 (`0.500`);
- external Verified@1: 0/6 (`0.000`);
- oracle pass rate: 0/3 (`0.000`);
- external false-verified rate: 3/3 (`1.000`);
- infeasibility detection: 1/3 (`0.333`).

The adjudicated replay includes 11 definitive cases and reports status accuracy 5/11, external Verified@1 0/5, success after repair 0/5, oracle pass 0/3, and external false verification 3/3.

V7-001 through V7-003 reached internal `verified` but exposed `src/cli.py` instead of their declared `forge_blind_v7.*` public modules. These are preserved false-verified incidents. Typed public-import closure was added structurally; the old artifacts now fail with `missing_public_module`.

### Targeted post-closure replay

The [targeted workflow](https://github.com/Daniele-Cangi/Derivative/actions/runs/32564951523) evaluated V7-001 through V7-005 and remained a regression replay.

Raw results:

- externally accepted artifacts: 0/5;
- external Verified@1: 0/5 (`0.000`);
- external false-verified rate: 1/1 (`1.000`);
- model requests: `32`;
- model tokens: `636,018`;
- repairs: `9`;
- estimated model cost: `$1.847928`;
- median latency: `237.78s`;
- P95 latency: `313.72s`.

Public import closure worked for V7-001, but its frozen oracle was unusable: `redirect_std.__enter__()` did not return the object bound by `with redirect_std() as redir`, while the oracle dereferenced `redir.stdout` and `redir.stderr`.

The raw replay remains unchanged. Oracle contract preflight now detects this harness defect as `oracle_invalid` before Forge execution and before any model request.

V7-002 through V7-005 remained genuine fail-closed candidate failures after two repairs each. Their evidence is concentrated in semantic preflight, missing requirement assertions, non-semantic coverage, and generated-test syntax failures. None was packaged.

Targeted receipt SHA-256:

`8412E0886F806415893D27D83939BCF3E18B916D5E64DEADEDA75572A4008821`

Sources:

- [Manifest](../benchmarks/blind_v7/external_001/manifest.json)
- [Immutable baseline](../benchmarks/blind_v7/external_001/baseline_result.json)
- [Requirement adjudication](../benchmarks/blind_v7/external_001/requirement_adjudication.json)
- [Baseline adjudicated metrics](../benchmarks/blind_v7/external_001/adjudicated_metrics.json)
- [First post-fix replay](../benchmarks/blind_v7/external_001/post_fix_replay_001.json)
- [Replay adjudicated metrics](../benchmarks/blind_v7/external_001/post_fix_replay_001_adjudicated_metrics.json)
- [Targeted post-closure replay](../benchmarks/blind_v7/external_001/post_fix_replay_002_targeted.json)

Additional SHA-256 values:

- baseline report: `C571B8E4891C9CC45B69BAF63DACE6A00FB8B9A2DCB468DBE4BC107451FC2F9D`;
- adjudication: `1461B0719315EE41244B271C0C2371E5BB46A53AF5BF5E1797B1BC2C45B0A8CB`;
- baseline metrics: `F11218B6B3B41CC0D3BDE191BEE6579EF97BD14F676F89D537935EC15BBD51E6`;
- first replay: `91B23F2B6ED0F9B1EA105A268728E346566EEE64C605A2AAF937C6D1B98B7D15`;
- replay metrics: `60DECC40B202E409C4E6A3EEA9D7A91A5E49D42C40E8DCA7A1E3225259CE8D24`.

## Blind V8

Blind V8 has already been executed and is a known regression corpus. Its manifest remains unchanged at SHA-256 `071C7D11D5A6D6CAB46198B1464CD03F8B8AFCDCCEAAD23E7F93D7E283AE12C1`. V8-005 may be used for regression diagnosis only; its replays are never new blind evidence.

### Targeted V8-005 exact-output replay

The [targeted workflow](https://github.com/Daniele-Cangi/Derivative/actions/runs/33297062090) replayed V8-005 on commit `9d2f3a3` with GPT-5 nano, the locked Linux/Python 3.11 dependency set, and Docker isolation. The report records `execution_kind=post_fix_replay` and `baseline_verified=false`.

Observed regression result:

- expected and observed terminal status: `verified`;
- first validation failed; one bounded repair succeeded on the second validation;
- independent frozen oracle: valid, executed, and passed 12/12;
- external Verified@1: 0/1 (`0.000`);
- success after repair: 1/1 (`1.000`);
- external false-verified rate: 0/1 (`0.000`);
- model requests: `6`;
- model tokens: `123,257`;
- configured estimated cost: `$0.0157035`;
- total case runtime: `152.40s`.

The successful repair reduced nine internal validation failures to zero without introducing a new signature. All three validation layers passed. The packaged validation receipt SHA-256 is `966A49ECCF18BF276FF3318F83EF0956F999EE17FB6B2A8D6060978F760EEE2C`; its exact bytes and the canonical artifact-manifest dump were rehashed after download and matched the declared receipt. The artifact, validation, and behavioral-contract seals also matched.

This is attributable regression evidence for the general exact-output and observation-fidelity mechanism. It does not update or replace any V8 blind baseline score.

Sources:

- [Manifest](../benchmarks/blind_v8/external_001/manifest.json)
- [Targeted regression workflow](https://github.com/Daniele-Cangi/Derivative/actions/runs/33297062090)

## Blind V9

Blind V9 was created by the isolated one-shot producer on commit `a7ec4ad`, using separate stateless generation and review requests and no Forge source, generated artifact, or earlier blind case in its prompts. It produced six `verified`, three `validation_failed`, and three `infeasible_proven` cases, all on the already supported Python CLI surface. The producer made 55 model requests, used 93,000 tokens, and recorded configured cost of `$0.324510` before atomically publishing the frozen bundle.

Frozen bundle integrity:

- manifest SHA-256: `02303B01513189A3668A24C8E95AE22D9414E1E30ED683A23A28581B7005258F`;
- dataset SHA-256: `F29C91BE7680ED4EE3857370E5B523033C66D82F2E93568AFD26CBE7E072E50C`;
- protected 68-file Forge baseline SHA-256: `D721261CC7795541B0BB1B43B2AD4AEC0390782D85DE0D195C688488870EF88B`;
- frozen commit: `4d8ee7d2c391db60089412e955764984c0850794`.

### Immutable baseline

The [first and only baseline workflow](https://github.com/Daniele-Cangi/Derivative/actions/runs/33298884420) ran the locked Linux/Python 3.11 environment and Docker sandbox against the frozen commit. It completed in 21m17s and recorded `execution_kind=sealed_baseline` with `baseline_verified=true`. Workflow success means the measurement and evidence upload completed; thresholds were deliberately non-blocking so a negative result could not suppress its receipt.

Raw results:

- externally passed cases: 4/12;
- terminal-status matches: 7/12 (`0.583`);
- externally accepted expected-verified artifacts: 1/6;
- external Verified@1: 0/6 (`0.000`);
- external success after repair: 1/6 (`0.167`);
- oracle pass rate: 1/4 (`0.250`), with zero invalid oracles;
- external false-verified rate: 4/5 (`0.800`): three oracle-rejected expected-verified artifacts and one expected-infeasible case observed as `verified`;
- infeasibility detection: 0/3 (`0.000`);
- expected `validation_failed` cases: 3/3 correctly remained `validation_failed`;
- V9-003 and V9-005 ended as pre-model exceptions with `TypeError: Object of type bytes is not JSON serializable` and zero model requests;
- model requests: `92`;
- model tokens: `1,461,956` (`1,277,818` input and `184,138` output);
- model-cost coverage: 10/12 (`0.833`); the ten covered cases total `$4.028740`, while total cost and cost per accepted artifact correctly remain `null` because two case costs are unavailable;
- repairs: `18`, or `18.0` per externally accepted artifact;
- median latency: `110.23s`;
- P95 latency: `174.26s`;
- total case runtime: `1,210.80s`.

Five artifacts passed all three internal validation layers and were packaged, but only one passed its independent oracle. After download, every package's exact file set and code digest matched its manifest; every validation receipt and canonical artifact-manifest dump rehashed to the declared SHA-256; and behavioral-contract, validated-artifact, and validation-artifact seals were mutually consistent. The failure is therefore evidence-fidelity and classification behavior, not a corrupt receipt or invalid oracle.

The complete GitHub artifact is `forge-blind-v9-first-run`, artifact id `9728521686`, SHA-256 `99A7FE5684F2AD51B1391DADA973A7BDF70D5F28B0467184470132D6DF1DC6FE`. The immutable baseline report SHA-256 is `D9089F58A449658CDAFC0816F25398DBA22B0159C237E9730C1676A9AF4011EC`.

V9 became a known regression corpus when this baseline ran. Any later V9 execution must be labeled `post_fix_replay` and cannot be reported as new blind evidence.

### Targeted post-fix replay

The [targeted replay workflow](https://github.com/Daniele-Cangi/Derivative/actions/runs/33442276463) ran V9-001, V9-002, V9-003, V9-005, and V9-006 on commit `8744f00`, using the baseline model `gpt-4.1-2025-04-14`, the byte-identical locked dependency snapshot, Python 3.11, and Docker isolation. The receipt records `execution_kind=post_fix_replay` and `baseline_verified=false`; this is regression evidence against a known corpus, not a new blind score.

The run completed in 12m16s. V9-006 reached internal `verified` after two repairs and passed its frozen external oracle 4/4. V9-001 and V9-002, which the baseline had internally verified before their external oracles rejected them, instead remained fail-closed with explicit capability or semantic mismatches. V9-003 and V9-005 no longer raised `TypeError: Object of type bytes is not JSON serializable`; both completed with structured validation failures. Across the five selected cases, one artifact was externally accepted, four remained fail-closed, and no internally verified artifact failed an external oracle. The replay used 46 model requests and 719,803 tokens, with recorded cost of `$1.991756`.

The complete artifact is `forge-v9-targeted-post-fix-33442276463`, artifact id `9777049372`, SHA-256 `ADEB45016CBA19FCD3FE4910E1AE0B2B33AEE606BB852074DB9F48FB37EF779E`. The uploaded report SHA-256 is `C05F6B7C205278CD221F625D8C8A8942C6C94F70963A09CE66D5D6EC883E382D`; the repository copy differs only by its normalized terminal newline and has SHA-256 `3EC7EA5A1823EFCD7B67700AA0714FC131DAECC3C1DE588F061A90A4CAA5541F`. The execution context is preserved byte-for-byte with SHA-256 `7CDABF7FDFC5A1EE25DCECC2A5E68CD66F19339AC102DD8EF2006F72ABB55058`. The dependency snapshot is byte-identical to the archived V9 baseline snapshot, SHA-256 `A8A4DEF3784A76DC57F7D29BA6502F7ECE4346F8B8C8E7D97FF0DB4EB5929FED`.

Sources:

- [Manifest](../benchmarks/blind_v9/external_001/manifest.json)
- [Immutable baseline](../benchmarks/blind_v9/external_001/baseline_result.json)
- [Execution context](../benchmarks/blind_v9/external_001/baseline_execution_context.json)
- [Dependency snapshot](../benchmarks/blind_v9/external_001/baseline_dependency_snapshot.txt)
- [First and only baseline workflow](https://github.com/Daniele-Cangi/Derivative/actions/runs/33298884420)
- [Targeted post-fix replay](../benchmarks/blind_v9/external_001/post_fix_replay_001_targeted.json)
- [Targeted replay execution context](../benchmarks/blind_v9/external_001/post_fix_replay_001_targeted_execution_context.json)
- [Targeted post-fix replay workflow](https://github.com/Daniele-Cangi/Derivative/actions/runs/33442276463)

## Historical V2/V3 Evidence

Blind V2 and V3 remain immutable historical evidence under `benchmarks/`.

The original V2 run scored 6/10. Its legacy summary records external false-verified rate `0.0` without a persisted denominator, so this ledger does not reinterpret it as a schema-v3 rate. Later targeted runs are regression evidence, including a 3/3 rerun of previously failing feasible cases; that result is not reported as a fresh blind 10/10.

V3 exposed requirement extraction, domain routing, semantic test alignment, public module preservation, ambiguity handling, and contradiction-detection gaps. Successive known-case replays improved mechanisms but also exposed false-verification and frozen-oracle defects. The final recorded hybrid replay (`execution_kind=post_fix_replay`) matched all 12 terminal statuses internally but externally passed 9/12, with external false-verified rate 3/6 (`0.500`). Three failures required oracle adjudication rather than silent score adjustment.

The V3 manifest remains sealed to its original implementation digest. Every later report keeps `baseline_verified=false`.

## Running Evaluation

Run the internal deterministic regression gate:

```bash
python forge_benchmark.py --preset extended \
  --execution-backend docker \
  --sandbox-image derivative-forge-sandbox:py311 \
  --enforce-thresholds \
  --min-status-accuracy 0.95 \
  --min-verified-at-1 0.95 \
  --max-false-verified-rate 0.00 \
  --min-infeasible-detection-rate 1.00
```

Run repository-maintained held-out oracles:

```bash
python forge_heldout_benchmark.py
```

Run a known blind case only as an explicitly labeled replay:

```bash
python forge_blind_benchmark.py \
  --manifest benchmarks/blind_v5/external_001/manifest.json \
  --case-id V5-006 \
  --mode hybrid \
  --post-fix-replay
```

## Freezing a New Blind

The freezer accepts an externally authored `cases.json` and referenced oracle files. It never generates either and refuses overwrite.

```bash
python forge_blind_freeze.py PATH_TO_PRIVATE_BUNDLE \
  --bundle-id forge-blind-external-001 \
  --producer "Independent benchmark producer" \
  --requirements-origin "Requirements authored outside the Forge process" \
  --oracle-origin "Independent black-box acceptance suite" \
  --declaration "Requirements and oracles were finalized before Forge execution" \
  --source-url https://example.com/benchmark-spec
```

Schema v3 requires every case to declare `public_contract` with module, symbol, and kind, and every verified oracle must import exactly that target.

When no human producer is available, an isolated one-shot producer can create and freeze a private bundle:

```bash
python forge_blind_produce.py PATH_TO_PRIVATE_BUNDLE \
  --bundle-id forge-blind-external-001
```

This is operational isolation, not cryptographic proof of model independence. The destination must not exist and the Forge baseline must be clean and committed.

## Deriving Adjudicated Metrics

After a sealed baseline and independent label-adjudication receipt exist:

```bash
python forge_blind_metrics.py \
  --manifest PATH_TO_BUNDLE/manifest.json \
  --baseline-report PATH_TO_BUNDLE/baseline_result.json \
  --adjudication PATH_TO_BUNDLE/requirement_adjudication.json \
  --output PATH_TO_BUNDLE/adjudicated_metrics.json
```

This command makes no model calls, validates bundle hashes and case IDs, writes once, records exclusions, and never edits source receipts.

## Next Evaluation Rule

V5, V6, V7, V8, and V9 are known regression corpora. They must not be optimized into new blind claims. V8-005 may be used only as a known regression case and never as new blind evidence. The next generality measurement must use a new schema-v3 bundle frozen before Forge sees its requirements or oracles.

The target metrics are reported together: External Verified@1, success after repair, external acceptance, false verification, infeasibility detection, invalid-benchmark rejection, median/P95 latency, tokens, configured cost per externally accepted artifact, and repairs per successful build.
