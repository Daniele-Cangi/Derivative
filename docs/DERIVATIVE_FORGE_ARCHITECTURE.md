# Derivative and Forge Architecture Boundary

Derivative and Forge are intentionally connected, but they have different responsibilities.

## Responsibility boundary

Derivative is the computational reasoning substrate. It owns cognitive lenses, deterministic solvers, obligation compilation, execution-grounded reasoning, contradiction witnesses, audit persistence, and design memory. Its output can support a plan or prove that stated constraints are jointly infeasible.

Forge is the software-synthesis pipeline built on that substrate. It preserves a natural-language requirement as typed atoms and contracts, asks Derivative for a grounded plan, expands the plan into a candidate, validates the candidate independently, performs bounded evidence-targeted repair, and packages only a fully validated artifact.

The relationship is therefore not a legacy dependency or two agents chained together. Forge reuses Derivative as its truth-producing planning substrate:

```text
requirement
  -> Forge RequirementCompiler (typed intent)
  -> Forge PlannerStage
  -> Derivative CognitiveSubstrate + ReasoningKernel
  -> execution evidence or infeasibility witness
  -> Forge FeasiblePlan or InfeasibilityCertificate
  -> CoderStage / CandidateCompiler
  -> isolated ValidatorStage
  -> bounded repair or verified package
```

The validator remains independent of generated claims and model confidence. Derivative can ground planning and candidate revision, but it cannot mark a Forge build verified. Only the Forge validation contracts and observed execution evidence can authorize packaging.

## V5 closure import coupling

The semantic boundary is sound; the dependency-loading boundary is not yet optimized.

`forge.py` imports `PlannerStage`. `PlannerStage` imports and constructs `CognitiveSubstrate` and `ReasoningKernel`. `CognitiveSubstrate` imports every lens module and instantiates every lens at startup. Several lens modules import their scientific runtime eagerly, while `ReasoningKernel` imports `ExecutionLoop` and the topology solver, which import NetworkX at module load.

On the V5 closure environment, a clean `import forge` measured approximately 42 seconds, loaded 5,423 Python modules, and loaded these optional or domain-specific roots even though no requirement had been compiled:

- `dowhy`
- `networkx`
- `pgmpy`
- `pint`
- `qiskit`
- `scipy`
- `sympy`

It also loaded host dependencies such as `openai`, `typer`, and `rich`. This explains why Forge carried the complete Derivative dependency set at startup in the frozen V5 snapshot.

## First post-V5 boundary improvement

The Forge composition root now imports default stages and model-backed repair components only when `run_forge()` needs to construct them. Importing the module or rendering `forge.py --help` no longer constructs the Derivative substrate.

Using the same clean-process probe on the same environment, `import forge` now measures approximately 1.298 seconds and 360 loaded modules. None of the observed optional substrate roots (`dowhy`, `networkx`, `pgmpy`, `pint`, `qiskit`, `scipy`, or `sympy`) is loaded. Typer and Rich remain because they implement the host CLI.

This is a composition-root fix, not complete capability loading. A default Forge run still constructs `PlannerStage`, whose current `CognitiveSubstrate` loads all lens implementations. Selective runtime loading and optional installation groups remain the next dependency task.

## Post-V5 dependency objective

The dependency objective after the composition-root fix is capability-driven runtime loading, not separation into a parallel truth system.

The intended change is:

1. keep the existing typed `CognitiveSubstrate` and `ReasoningKernel` contracts;
2. represent lenses and deterministic solvers through lightweight capability descriptors;
3. import a scientific runtime only after requirement classification selects the corresponding capability;
4. construct model-backed candidate and repair components only in modes that can use them;
5. split installation metadata into a minimal Forge/runtime group plus explicit symbolic, topology, probabilistic, causal, quantum, and scientific extras;
6. retain an `all` installation path for the complete Derivative research substrate;
7. fail explicitly when a selected required capability is unavailable, rather than silently replacing it with narration.

This work must preserve infeasibility routing, obligation semantics, audit evidence, and fail-closed validation. Startup improvements are not allowed to create a second planner or weaken the reason Derivative and Forge are connected.

## Architectural invariant

Derivative supplies grounded reasoning capabilities. Forge supplies software-build contracts and independent certification gates. Dependency loading may become selective; the source of truth must remain shared.
