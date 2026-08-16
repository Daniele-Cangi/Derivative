# Contributing to Derivative

Derivative welcomes contributions that expand its verified software surface without weakening the distinction between generated code and verified artifacts.

The project is deliberately conservative about the word **verified**. A new generator, adapter, model prompt, or passing test suite is not enough by itself to extend the verified surface. Contributions that add new behavior must preserve requirement traceability, fail-closed validation, independent execution evidence, and the benchmark discipline already used by Forge.

Before implementing a new domain or capability, read [`docs/CERTIFIED_EXTENSION_CONTRACT.md`](docs/CERTIFIED_EXTENSION_CONTRACT.md).

## Good contribution areas

Contributions are especially useful in these areas:

- new certified domain or capability extensions;
- requirement and interface extraction that improves the typed build contract;
- planner and capability-model improvements that remove ad-hoc domain assumptions;
- semantic and adversarial validators;
- execution isolation and evidence capture;
- repair grounding and provenance;
- independent benchmark cases and oracle infrastructure;
- documentation, diagnostics, and contributor tooling.

Small bug fixes, tests, and documentation improvements do not need an issue first. Please discuss large changes to verification semantics, benchmark governance, execution policy, or the Forge contract before implementation.

## Core invariant

Derivative does not trust a candidate because it was generated successfully or because its own generated tests pass.

A contribution must not make `verified` easier to reach by reducing evidence requirements. When the system cannot establish the declared behavior inside its current verification model, the correct result is normally `validation_failed`, not an optimistic success.

In particular, do not:

- add benchmark-case-specific templates after observing a blind case;
- weaken or bypass validation gates to improve a benchmark score;
- treat generated acceptance tests as independent evidence;
- silently change frozen benchmark inputs, oracles, manifests, or historical reports;
- report a post-fix replay as a fresh blind result;
- mark an adapter as semantically complete when it only renders runnable code;
- encode a new domain only as keyword routing when a typed contract can represent the distinction instead.

## Certified extensions

A domain or capability extension should normally supply all of the following layers:

1. **Recognition / typed intent** — the requirement is represented in the build contract without relying on a case-specific phrase.
2. **Planning** — files, interfaces, tests, requirements, and capabilities are mapped explicitly.
3. **Generation** — deterministic rendering or an explicitly untrusted candidate-compiler route produces the planned transaction.
4. **Capability declaration** — the adapter declares only the semantics it actually implements.
5. **Verification** — runtime, obligation, semantic, quality, and adversarial checks can reject incorrect candidates independently of generator claims.
6. **Evidence** — requirement-to-file/test provenance and relevant execution evidence are retained.
7. **Evaluation** — the extension is exercised by tests and, when it changes the verified surface, by independent or held-out evidence appropriate to the claim.

A useful extension is therefore more than a renderer. See the certified extension contract for the acceptance boundary.

## Working with blind benchmarks

Blind and sealed benchmark artifacts are evidence, not a development fixture library.

Once a blind requirement or oracle has been observed by the development process, later runs against the same bundle are regression replays. Structural fixes may be informed by the failure class, but a contribution must not add a special-case implementation whose purpose is to solve the known case.

Frozen reports remain unchanged even when an oracle defect is later identified. Corrections or adjudications should be recorded separately so the original evidence remains auditable.

## Development workflow

1. Fork the repository and create a focused branch.
2. Keep one conceptual change per pull request where practical.
3. Add or update tests that demonstrate the structural behavior being changed.
4. Run the repository test suite.
5. Run the relevant Forge benchmark or focused validation path when the change affects synthesis, validation, repair, or packaging.
6. In the pull request, explain the failure class or capability being addressed, the evidence added, and why the change generalizes beyond a single observed case.

For ordinary repository tests:

```bash
python -B -m pytest -q -p no:cacheprovider tests
```

The production Forge validation path requires the Docker sandbox described in the README. Changes that affect execution, verification, candidate preflight, repair preflight, or external oracle execution should be validated through that boundary rather than only through the trusted local backend.

## Pull request notes

Please include:

- the problem or extension boundary;
- affected requirement/capability/adapter/validator layers;
- tests and benchmark commands run;
- whether any previously sealed benchmark case was already known during development;
- new failure signatures or evidence fields, if any;
- any claim that remains intentionally unsupported or fail-closed.

Contributions are welcome even when they expose a failure rather than making more cases pass. A reproducible failure class with clear evidence can be more valuable than a broader but weakly verified success.