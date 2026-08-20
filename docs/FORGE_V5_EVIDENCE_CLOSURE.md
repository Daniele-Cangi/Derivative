# Forge Evidence Closure - Blind V5

This document defines the frozen Blind V5 evidence snapshot. It records what was observed; it does not reinterpret post-fix replays as a new blind run.

## Closure boundary

- Blind V5 requirements and external oracles were frozen before first Forge execution.
- `benchmarks/blind_v5/external_001/baseline_result.json` remains the immutable first-run result.
- `post_fix_replay_*.json` files remain revision-specific regression receipts.
- Oracle-invalid findings remain attached to the unchanged frozen fixtures.
- False-positive incidents remain preserved in their original reports.
- No aggregate is assembled by selecting the best result for each case across revisions.
- No further V5-driven feature or template work belongs to this snapshot.

The closure implementation has a local regression result of **407 passed, 2 skipped**. The release is created only after the repository CI test-and-gate workflow passes on the same commit.

## Immutable baseline

The first Blind V5 execution reported:

- 4 of 12 expected terminal statuses matched;
- status accuracy 0.417;
- external false-verified rate 1.000;
- infeasibility detection rate 0.333;
- 680,399 model tokens;
- estimated model cost $1.83934.

These values are the Blind V5 score. Later receipts are regression evidence and do not replace them.

## Attributable regression evidence

- `V5-002`: verified after one repair; independent oracle passed 7/7; 64,234 model tokens; four model requests; external false-verified rate 0.000 for that replay.
- `V5-005`: verified and accepted by its independent oracle.
- `V5-006`: verified after two repairs and three validation attempts; independent oracle passed 9/9; 116,861 model tokens; seven model requests; estimated cost $0.304336; case runtime 190.22 seconds; external false-verified rate 0.000 for that replay.
- `V5-010` and `V5-012`: terminated as `infeasible_proven` in preserved receipts.
- `V5-001`, `V5-003`, and `V5-004`: terminated as benchmark-only `oracle_invalid` after executable preflight identified contradictions in the frozen acceptance fixtures. Forge execution is skipped for an invalid oracle.

Other V5 cases are not promoted by omission. Their historical results remain in the baseline and replay files.

## Meaning of the snapshot

The snapshot demonstrates a functioning research system with typed requirement preservation, isolated execution, independent validation, bounded evidence-targeted repair, explicit infeasibility proof, and preserved benchmark adjudication.

It does not establish broad general-purpose coding-agent performance. The known V5 distribution has already influenced development, so it is a regression corpus after closure.

## Next independent evaluation

The next blind bundle must be authored and frozen outside the Forge development loop before Forge can inspect it. It must keep requirements and black-box oracles hidden until the freeze succeeds.

The evaluation should report together:

- internal Verified@1;
- external Verified@1;
- verified after repair;
- external oracle pass rate;
- external false-verified rate;
- infeasibility detection rate;
- invalid benchmark rejection rate;
- median and P95 latency;
- model input, output, and total tokens;
- cost per internally verified artifact;
- cost per externally accepted artifact;
- repairs per successful build.

Maintaining a near-zero external false-verified rate remains a release condition. Throughput improvements do not justify weakening validation or changing frozen inputs.
