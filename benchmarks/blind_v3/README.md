# Blind v3 intake

This directory intentionally contains no benchmark cases or acceptance oracles.
Blind v3 inputs must be authored outside the Forge development process and frozen
before Forge receives any requirement.

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
