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
