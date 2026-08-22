# Benchmark Evidence

This ledger separates immutable blind evidence from regression work. It is intentionally more detailed than the [project README](../README.md).

## Reporting Rules

- A blind bundle is frozen before Forge sees its requirement or oracle.
- Baseline reports are append-only and never rescored after implementation changes.
- A replay against a known bundle is labeled `post_fix_replay`, never a new blind score.
- Internal `verified` and external oracle acceptance are separate facts.
- Invalid oracles are reported as `oracle_invalid`, excluded with evidence, and never converted into passes.
- Undefined rates are `null`, not synthetic zeroes.
- Every rate reports an explicit numerator and denominator.
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

At the V7 oracle-preflight checkpoint:

- local suite: **470 passed, 2 skipped**;
- minimal Forge runtime workflow: passing;
- full Linux/Docker test and benchmark gate: passing;
- internal 30-case benchmark: deterministic regression evidence, not independent blind proof.

## Blind V5

Blind V5 was frozen before first execution and established the current evidence-closure discipline.

Immutable baseline:

- 12 cases;
- raw status accuracy: 4/12 (`0.417`);
- external false-verified rate: `1.000`;
- infeasibility detection: `0.333`;
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
- status accuracy: `0.333`;
- external Verified@1: `0.000`;
- external false-verified rate: `1.000`;
- infeasibility detection: `0.000`;
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
- status accuracy: `0.333`;
- external Verified@1: `0.000`;
- success after repair: `0.000`;
- external false-verified rate: `0.000`;
- infeasibility detection: `0.333`;
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
- raw status accuracy: `0.500`;
- external Verified@1: `0.000`;
- oracle pass rate: `0.000`;
- external false-verified rate: `1.000`;
- infeasibility detection: `0.333`.

The adjudicated replay includes 11 definitive cases and reports status accuracy 5/11, external Verified@1 0/5, success after repair 0/5, oracle pass 0/3, and external false verification 3/3.

V7-001 through V7-003 reached internal `verified` but exposed `src/cli.py` instead of their declared `forge_blind_v7.*` public modules. These are preserved false-verified incidents. Typed public-import closure was added structurally; the old artifacts now fail with `missing_public_module`.

### Targeted post-closure replay

The [targeted workflow](https://github.com/Daniele-Cangi/Derivative/actions/runs/32564951523) evaluated V7-001 through V7-005 and remained a regression replay.

Raw results:

- externally accepted artifacts: 0/5;
- external false-verified rate: `1.000`;
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

## Historical V2/V3 Evidence

Blind V2 and V3 remain immutable historical evidence under `benchmarks/`.

The original V2 run scored 6/10 with external false-verified rate 0.0. Later targeted runs are regression evidence, including a 3/3 rerun of previously failing feasible cases; that result is not reported as a fresh blind 10/10.

V3 exposed requirement extraction, domain routing, semantic test alignment, public module preservation, ambiguity handling, and contradiction-detection gaps. Successive known-case replays improved mechanisms but also exposed false-verification and frozen-oracle defects. The final recorded hybrid replay matched all 12 terminal statuses internally but externally passed 9/12, with external false-verified rate 0.500. Three failures required oracle adjudication rather than silent score adjustment.

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

V5, V6, and V7 are known regression corpora. They must not be optimized into new blind claims. The next generality measurement must use a new schema-v3 bundle frozen before Forge sees its requirements or oracles.

The target metrics are reported together: External Verified@1, success after repair, external acceptance, false verification, infeasibility detection, invalid-benchmark rejection, median/P95 latency, tokens, configured cost per externally accepted artifact, and repairs per successful build.
