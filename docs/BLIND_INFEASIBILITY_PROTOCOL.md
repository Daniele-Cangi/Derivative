# Blind infeasibility admission protocol

## Trust boundary

An infeasible expected label needs a deterministic proof **of mandatory requirements**,
not an independent false formula, a model's confidence, or an implementation failure.
The independent certificate governs benchmark admission only. Forge may separately
prove an infeasible requirement from its **public normative block**, but never from
the private certificate or expected label. Neither path bypasses fail-closed
validation or changes best-candidate rollback.

Version `finite-linear-v1` is deliberately restricted. It supports independently
authored, explicit finite integer output contracts. It does **not** formalize arbitrary
natural language and does not establish equivalence between prose and mathematics.
This changes the requirement format; results must be reported separately from
prose-only blind benchmarks. Historical bundles remain unchanged.

## Public obligation and private certificate

The independent producer authors both the requirement prose and a `formal_obligation`:

- `protocol`: `finite-linear-v1`;
- `variables`: ordered objects with `name`, inclusive integer `lower` and `upper`;
- `constraints`: objects with an ordered integer `coefficients` vector, `relation`
  (`le`, `eq`, `ne`), and integer `rhs`.

Every constraint means `sum(coefficients[i] * variables[i]) relation rhs`.
All constraints are conjunctive, not alternative branches. A trusted renderer appends
an exact normative block to the requirement. It requires the declared public callable,
called with **no arguments**, to terminate normally and return a dict of those integer
values. There are no preconditions, optional obligations, error results, or fallback
paths. The block explicitly prohibits exceptions, None, and impossibility reports.
Other signatures, CLI entrypoints, conditional obligations, and unbounded domains
are outside this version. A conditional scenario cannot be certified by extracting
only its impossible branch.

The formal block is an authored part of the specification, not a repair or inferred
translation of rejected prose. The independent reviewer still checks that the prose
and interface agree with it and rejects conflicting scopes, escape paths and unrelated
decorative contradictions. This semantic review is not a formal equivalence proof.
No prior rejected proposal may be converted into new blind evidence.

The certificate, stored separately from the public requirement, records protocol,
method, SHA-256 of the complete UTF-8 requirement, and assignments checked. The
verifier checks the exact block and interface binding and **recomputes** the exhaustive
proof. A hash binds bytes; it does not prove that unrelated prose means the same thing.
Forge receives the public requirement, including the normative constraints, but not
the certificate, expected label, review findings, or enumeration outcome.

After the first V10 baseline exposed missed contradictions, Forge's planner gained a
separate, general reader for the exact public block. It validates the full binding
and independently enumerates the finite domains before any model planning. An
unsatisfiable block yields an internal infeasibility proof; a satisfiable block
continues ordinary planning; a malformed or partially altered block fails closed.
No requirement is inferred from prose and no benchmark-private data is consulted.
This shared formal format narrows the independence claim: it tests execution of an
explicit public contract, not discovery of contradictions in unrestricted prose.
All V10 checks after the sealed baseline are known-case regressions, not blind scores.

## Soundness and resource limits

Each variable has a nonempty finite domain. The checker enumerates their full Cartesian
product using exact Python integers. One satisfying assignment rejects the infeasible
claim. Only when every assignment violates at least one mandatory constraint does
admission succeed. No generated code, eval, floating-point arithmetic, solver packages,
network calls, or model adjudication are used for this proof.

Limits: 1–8 variables, 1–16 constraints, domain width at most 16, at most 4,096 total
assignments, integer bounds/coefficients/RHS of absolute value at most 1,000,000.
Booleans are not integers for this schema. Unknown fields, versions, relations,
malformed vectors, empty domains, excessive budgets, altered bindings, and forged
certificates are rejected, never classified as infeasible by default.

## Rollout checkpoints

Enable the mode explicitly with `forge_blind_produce.py ... --infeasibility-protocol
finite-linear-v1`. The default remains the legacy prose-only producer. Only infeasible
slots request a formal obligation; verified and ambiguous slots keep their existing
schemas. Proof generation adds no model calls and no retry allowance. The existing
bounded replacement loop remains unchanged. Missing/invalid/satisfiable obligations
are rejected before independent review; certificate failures never fall back to the
legacy recognizers. Review rejection remains authoritative even with a valid proof.

The freezer persists certified bundles as **manifest schema 4**, including every
certificate in the hashed dataset. Schema 4 requires a certificate for every infeasible
case and disallows a mix with legacy infeasibility admission. The loader rejects
downgrades and rechecks every proof even when baseline verification is disabled for a
replay. Older clients reject schema 4 as unsupported. Legacy bundles stay schemas 1–3;
new legacy freezes remain schema 3. No existing bundle is rewritten.

1. Exact verifier, canonical rendering, adversarial and independent-reference tests.
2. Explicit producer opt-in, preflight, independent review, persistence, freeze/load
   verification, and a test demonstrating no certificate-to-observed-status shortcut.
3. Green offline/CI checks and a committed baseline before any new API production.
4. Only a separately authorized fresh production run may create blind evidence.

Synthetic protocol tests are regressions, not blind evidence. V8-005 remains only a
known regression. No extra retry, task-specific recognizer, or execution domain is
introduced by this protocol.
