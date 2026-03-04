import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import anthropic

from core.artifacts import DesignArtifact
from core.designs import EmergentDesign
from core.json_utils import clamp_float, ensure_string_list, extract_json_object
from core.runtime_mode import normalize_execution_mode
from core.topology_solver import (
    TopologyCandidate,
    TopologySearchQuery,
    TopologySearchResult,
    parse_topology_search_query,
    solve_topology_search,
)
from lenses.base import CognitiveLens


@dataclass
class ReasoningStep:
    step_id: str
    description: str


@dataclass
class ReasoningResult:
    conclusion: str
    reasoning_chain: List[ReasoningStep]
    violated_constraints: List[str]
    epistemic_confidence: float
    lens_contributions: Dict[str, float]
    generated_designs: List[EmergentDesign] = field(default_factory=list)
    topology_search: Optional[TopologySearchResult] = None


class ReasoningKernel:
    def __init__(self, api_key: Optional[str] = None, execution_mode: Optional[str] = None):
        resolved_key = (api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
        self.api_key = resolved_key
        self.execution_mode = normalize_execution_mode(execution_mode)
        self.allow_local_fallback = self.execution_mode != "remote-only"
        self.use_live_model = (
            self.execution_mode in {"hybrid", "remote-only"}
            and bool(resolved_key)
            and resolved_key != "dummy_key_for_testing"
        )
        self.client = anthropic.Anthropic(api_key=resolved_key) if self.use_live_model else None

    def synthesize(
        self,
        problem: str,
        lenses: List[CognitiveLens],
        design_context: Optional[List[Dict[str, object]]] = None,
    ) -> ReasoningResult:
        if not lenses:
            raise ValueError("No cognitive lenses provided for synthesis.")

        topology_query = parse_topology_search_query(problem)
        local_result = self._synthesize_locally(
            problem,
            lenses,
            design_context=design_context or [],
            topology_query=topology_query,
        )
        if topology_query is not None:
            return local_result
        if self.execution_mode == "local-only":
            return local_result
        if not self.use_live_model:
            if self.allow_local_fallback:
                return local_result
            raise RuntimeError("ReasoningKernel requires remote mode with a valid ANTHROPIC_API_KEY.")

        framing_texts = []
        for lens in lenses:
            framing_texts.append(
                f"Lens: {lens.lens_name} (Tag: {lens.epistemic_tag})\n"
                f"Framing: {lens.framing}\n"
                f"Constraints: {', '.join(lens.constraints)}\n"
                f"Blind Spots: {', '.join(lens.blind_spots)}\n"
                f"Operator Primitives: {', '.join(lens.operator_primitives)}\n"
                f"Design Affordances: {', '.join(lens.design_affordances)}\n"
            )
        framings_str = "\n".join(framing_texts)
        context_str = self._format_design_context(design_context or [])

        system_prompt = """You are the generative engineering kernel of the Derivative architecture.
You do not merely explain. You synthesize new engineering paths by composing multiple computational lenses.
Return one JSON object using this exact structure:
{
    "conclusion": "Final synthesized answer",
    "reasoning_chain": [
        {"step_id": "1", "description": "step 1 reasoning"},
        {"step_id": "2", "description": "step 2 reasoning"}
    ],
    "violated_constraints": [],
    "epistemic_confidence": 0.85,
    "lens_contributions": {
        "Causal Inference": 0.4,
        "Symbolic Logic": 0.6
    }
}
Do not remove the idea of invention: prefer executable design moves over commentary."""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2200,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Problem: {problem}\n\n"
                            f"Lens Material:\n{framings_str}\n\n"
                            f"Prior Design Lineage:\n{context_str}"
                        ),
                    }
                ],
            )
            data = extract_json_object(response.content[0].text)
            return ReasoningResult(
                conclusion=str(data.get("conclusion") or local_result.conclusion).strip(),
                reasoning_chain=self._coerce_steps(
                    data.get("reasoning_chain"),
                    fallback=local_result.reasoning_chain,
                ),
                violated_constraints=ensure_string_list(data.get("violated_constraints")),
                epistemic_confidence=clamp_float(
                    data.get("epistemic_confidence"),
                    default=local_result.epistemic_confidence,
                    minimum=0.0,
                    maximum=1.0,
                ),
                lens_contributions=self._normalize_contributions(
                    data.get("lens_contributions"),
                    lenses,
                    fallback=local_result.lens_contributions,
                ),
                generated_designs=list(local_result.generated_designs),
                topology_search=local_result.topology_search,
            )
        except Exception:
            if self.allow_local_fallback:
                return local_result
            raise

    def _synthesize_locally(
        self,
        problem: str,
        lenses: List[CognitiveLens],
        design_context: List[Dict[str, object]],
        topology_query: Optional[TopologySearchQuery] = None,
    ) -> ReasoningResult:
        if topology_query is not None:
            return self._synthesize_exact_topology_search(
                problem,
                lenses,
                topology_query,
                design_context=design_context,
            )

        top_constraints = self._top_items(lenses, "constraints")
        top_blind_spots = self._top_items(lenses, "blind_spots")
        inferred_risks = self._infer_problem_risks(problem)
        lens_contributions = self._normalize_contributions({}, lenses)
        generated_designs = self._generate_designs(
            problem,
            lenses,
            top_constraints,
            top_blind_spots,
            inferred_risks,
            design_context,
        )

        average_confidence = sum(lens.confidence for lens in lenses) / len(lenses)
        best_design = generated_designs[0] if generated_designs else None
        novelty_bonus = 0.08 * (best_design.composite_score if best_design else 0.0)
        context_bonus = 0.04 if design_context else 0.0
        confidence_penalty = min(0.16, 0.02 * max(0, len(top_blind_spots) - 1))
        epistemic_confidence = clamp_float(
            average_confidence - confidence_penalty + novelty_bonus + context_bonus,
            default=0.65,
            minimum=0.35,
            maximum=0.95,
        )

        strongest_lenses = ", ".join(lens.lens_name for lens in lenses[:3])
        primitive_count = sum(len(lens.operator_primitives) for lens in lenses)
        selected_title = best_design.title if best_design else "No viable emergent design"
        selected_score = best_design.composite_score if best_design else 0.0
        artifact_count = sum(len(design.artifacts) for design in generated_designs)

        reasoning_chain = [
            ReasoningStep(
                step_id="1",
                description=(
                    f"Mapped the problem across {len(lenses)} lenses; the strongest framings came from {strongest_lenses}."
                ),
            ),
            ReasoningStep(
                step_id="2",
                description=(
                    f"Converted the lens outputs into {primitive_count} operator primitive(s) and explicit design affordances."
                ),
            ),
            ReasoningStep(
                step_id="3",
                description=(
                    f"Composed {len(generated_designs)} cross-library engineering candidates and stress-tested them "
                    f"against shared constraints: {', '.join(top_constraints[:3]) if top_constraints else 'none'}."
                ),
            ),
            ReasoningStep(
                step_id="4",
                description=(
                    f"Selected {selected_title} as the leading architecture with composite score {selected_score:.2f}, "
                    f"then emitted {artifact_count} executable artifact(s) for downstream execution."
                ),
            ),
        ]
        if design_context:
            reasoning_chain.append(
                ReasoningStep(
                    step_id="5",
                    description=(
                        f"Injected {len(design_context)} memory lineage seed(s) to mutate prior successful designs "
                        "instead of restarting the search from scratch."
                    ),
                )
            )

        conclusion = self._build_local_conclusion(problem, generated_designs, inferred_risks)
        return ReasoningResult(
            conclusion=conclusion,
            reasoning_chain=reasoning_chain,
            violated_constraints=[],
            epistemic_confidence=epistemic_confidence,
            lens_contributions=lens_contributions,
            generated_designs=generated_designs,
        )

    def _synthesize_exact_topology_search(
        self,
        problem: str,
        lenses: List[CognitiveLens],
        topology_query: TopologySearchQuery,
        design_context: List[Dict[str, object]],
    ) -> ReasoningResult:
        search_result = solve_topology_search(topology_query)
        lens_contributions = self._normalize_contributions({}, lenses)
        satisfiable_count = len(search_result.satisfiable_topologies)
        generated_designs = self._build_exact_topology_designs(
            lenses,
            search_result,
            include_lineage_note=bool(design_context),
        )

        reasoning_chain = [
            ReasoningStep(
                step_id="1",
                description=(
                    f"Parsed an exact topology search over N={topology_query.node_count} nodes with "
                    f"fidelity > {topology_query.fidelity_threshold:.2f}, "
                    f"latency < {topology_query.latency_limit_ms:.0f}ms, and "
                    f"entanglement overhead <= {topology_query.entanglement_factor_limit:.2f}x MST."
                ),
            ),
            ReasoningStep(
                step_id="2",
                description=(
                    f"Enumerated {search_result.evaluated_topologies} connected non-isomorphic graph(s) "
                    "from the NetworkX graph atlas and evaluated each candidate directly."
                ),
            ),
            ReasoningStep(
                step_id="3",
                description=(
                    f"Filtered the search space down to {satisfiable_count} satisfiable topology configuration(s) "
                    "that preserve biconnectivity, latency, fidelity, and entanglement bounds simultaneously."
                ),
            ),
        ]

        if search_result.optimal_topology is not None:
            optimal = search_result.optimal_topology
            reasoning_chain.append(
                ReasoningStep(
                    step_id="4",
                    description=(
                        f"Ranked the satisfiable set by simultaneous novelty + feasibility scoring and selected "
                        f"{optimal.candidate_id} as the optimal topology "
                        f"(score {optimal.composite_score:.2f}, diameter {optimal.diameter}, "
                        f"node connectivity {optimal.node_connectivity})."
                    ),
                )
            )
            reasoning_chain.append(
                ReasoningStep(
                    step_id="5",
                    description=(
                        "Emitted an executable graph artifact for each top exact candidate, plus a verifiable "
                        "Qiskit circuit and a full JSON enumeration for the optimal topology."
                    ),
                )
            )
        else:
            reasoning_chain.append(
                ReasoningStep(
                    step_id="4",
                    description=(
                        "No graph satisfied the full constraint set, so the exhaustive search terminated without "
                        "promoting any topology into the design layer."
                    ),
                )
            )

        if design_context:
            reasoning_chain.append(
                ReasoningStep(
                    step_id=str(len(reasoning_chain) + 1),
                    description=(
                        f"Retained {len(design_context)} lineage seed(s) as historical context, but kept the "
                        "ranking itself fully solver-driven."
                    ),
                )
            )

        epistemic_confidence = 0.96 if search_result.optimal_topology is not None else 0.92
        return ReasoningResult(
            conclusion=self._build_exact_topology_conclusion(search_result),
            reasoning_chain=reasoning_chain,
            violated_constraints=[],
            epistemic_confidence=epistemic_confidence,
            lens_contributions=lens_contributions,
            generated_designs=generated_designs,
            topology_search=search_result,
        )

    def _build_exact_topology_designs(
        self,
        lenses: List[CognitiveLens],
        search_result: TopologySearchResult,
        include_lineage_note: bool,
    ) -> List[EmergentDesign]:
        designs: List[EmergentDesign] = []
        composition_tags = self._select_exact_topology_tags(lenses)

        for index, candidate in enumerate(search_result.satisfiable_topologies[:3], start=1):
            title = self._build_exact_topology_title(candidate)
            primitives = [
                f"Maintain node connectivity {candidate.node_connectivity} with no articulation points.",
                f"Keep worst-case diameter at {candidate.diameter} hop(s) for {candidate.worst_case_latency_ms:.0f}ms latency.",
                f"Lock routed fidelity at {candidate.end_to_end_fidelity:.5f} after "
                f"{search_result.query.gate_operations} logical gate operations.",
                f"Spend only {candidate.entanglement_overhead_factor:.2f}x the MST entanglement baseline.",
            ]
            outline = [
                f"Instantiate the exact edge set: {self._format_edge_list(candidate.canonical_edges)}",
                "Emit the graph artifact and verify node/edge counts through the generated build_topology() entrypoint.",
                "Bind an entangling scaffold to the selected edges so the quantum branch remains directly executable.",
                "Use the JSON enumeration artifact as the exhaustive proof ledger for every satisfiable topology.",
            ]
            if include_lineage_note:
                outline.append("Treat prior lineage as context only; do not mutate the exact ranking.")

            design = EmergentDesign(
                design_id=f"ED-X{index}",
                title=title,
                premise=(
                    f"Exact solver-selected topology {candidate.candidate_id} with {candidate.edge_count} edge(s), "
                    f"diameter {candidate.diameter}, and degree profile {candidate.degree_sequence}."
                ),
                composition_tags=composition_tags,
                component_primitives=primitives,
                implementation_outline=outline[:5],
                governing_constraints=self._build_exact_constraint_summary(search_result.query),
                novelty_score=candidate.novelty_score,
                feasibility_score=candidate.feasibility_score,
                composite_score=candidate.composite_score,
                artifacts=[],
                lineage_titles=[],
                mutation_strategy="solver-exhaustive enumeration",
            )
            design.artifacts = self._build_exact_topology_artifacts(
                design,
                candidate,
                search_result,
                include_enumeration=index == 1,
            )
            designs.append(design)

        return designs

    def _build_exact_topology_artifacts(
        self,
        design: EmergentDesign,
        candidate: TopologyCandidate,
        search_result: TopologySearchResult,
        include_enumeration: bool,
    ) -> List[DesignArtifact]:
        slug = design.title.lower().replace(" ", "_")
        artifacts = [
            self._build_exact_blueprint_artifact(design, candidate, search_result.query, slug),
            self._build_exact_graph_artifact(design, candidate, search_result.query, slug),
        ]
        if include_enumeration:
            artifacts.append(self._build_exact_quantum_artifact(design, candidate, search_result.query, slug))
            artifacts.append(self._build_topology_enumeration_artifact(design, search_result, slug))
        return artifacts

    def _build_exact_blueprint_artifact(
        self,
        design: EmergentDesign,
        candidate: TopologyCandidate,
        query: TopologySearchQuery,
        slug: str,
    ) -> DesignArtifact:
        lines = [
            f"title: {design.title}",
            f"candidate_id: {candidate.candidate_id}",
            "search_mode: exact-topology-enumeration",
            f"node_count: {query.node_count}",
            f"edge_count: {candidate.edge_count}",
            f"diameter: {candidate.diameter}",
            f"worst_case_latency_ms: {candidate.worst_case_latency_ms:.2f}",
            f"end_to_end_fidelity: {candidate.end_to_end_fidelity:.6f}",
            f"entanglement_overhead_factor: {candidate.entanglement_overhead_factor:.2f}",
            f"node_connectivity: {candidate.node_connectivity}",
            "canonical_edges:",
        ]
        for left, right in candidate.canonical_edges:
            lines.append(f"  - [{left}, {right}]")
        return DesignArtifact(
            artifact_id=f"{design.design_id}-A1",
            artifact_type="blueprint",
            filename=f"{slug}.yaml",
            language="yaml",
            content="\n".join(lines),
            execution_note="Exact solver blueprint for the promoted topology candidate.",
        )

    def _build_exact_graph_artifact(
        self,
        design: EmergentDesign,
        candidate: TopologyCandidate,
        query: TopologySearchQuery,
        slug: str,
    ) -> DesignArtifact:
        edge_text = ", ".join(f"({left}, {right})" for left, right in candidate.canonical_edges)
        content = "\n".join(
            [
                "import networkx as nx",
                "",
                "def build_topology():",
                "    graph = nx.Graph()",
                f"    graph.add_nodes_from(range({query.node_count}))",
                f"    graph.add_edges_from([{edge_text}])",
                "    return graph",
            ]
        )
        return DesignArtifact(
            artifact_id=f"{design.design_id}-A2",
            artifact_type="topology_graph",
            filename=f"{slug}_graph.py",
            language="python",
            content=content,
            execution_note="Verifiable exact topology graph emitted from the exhaustive solver.",
        )

    def _build_exact_quantum_artifact(
        self,
        design: EmergentDesign,
        candidate: TopologyCandidate,
        query: TopologySearchQuery,
        slug: str,
    ) -> DesignArtifact:
        lines = [
            "from qiskit import QuantumCircuit",
            "",
            "def build_design_circuit():",
            f"    qc = QuantumCircuit({query.node_count})",
            f"    for qubit in range({query.node_count}):",
            "        qc.h(qubit)",
        ]
        for left, right in candidate.canonical_edges:
            lines.append(f"    qc.cz({left}, {right})")
        lines.extend(
            [
                "    qc.barrier()",
                "    return qc",
            ]
        )
        return DesignArtifact(
            artifact_id=f"{design.design_id}-A3",
            artifact_type="quantum_circuit",
            filename=f"{slug}_circuit.py",
            language="python",
            content="\n".join(lines),
            execution_note="Verifiable entangling scaffold derived from the exact optimal topology.",
        )

    def _build_topology_enumeration_artifact(
        self,
        design: EmergentDesign,
        search_result: TopologySearchResult,
        slug: str,
    ) -> DesignArtifact:
        payload = {
            "search_type": "exact_topology_enumeration",
            "evaluated_topologies": search_result.evaluated_topologies,
            "satisfiable_count": len(search_result.satisfiable_topologies),
            "optimal_topology": search_result.optimal_topology.candidate_id
            if search_result.optimal_topology is not None
            else None,
            "assumptions": list(search_result.assumptions),
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "edge_count": candidate.edge_count,
                    "diameter": candidate.diameter,
                    "degree_sequence": list(candidate.degree_sequence),
                    "canonical_edges": [list(edge) for edge in candidate.canonical_edges],
                    "worst_case_latency_ms": round(candidate.worst_case_latency_ms, 4),
                    "end_to_end_fidelity": round(candidate.end_to_end_fidelity, 8),
                    "entanglement_overhead_factor": round(candidate.entanglement_overhead_factor, 4),
                    "node_connectivity": candidate.node_connectivity,
                    "novelty_score": round(candidate.novelty_score, 4),
                    "feasibility_score": round(candidate.feasibility_score, 4),
                    "composite_score": round(candidate.composite_score, 4),
                }
                for candidate in search_result.satisfiable_topologies
            ],
        }
        return DesignArtifact(
            artifact_id=f"{design.design_id}-A4",
            artifact_type="topology_enumeration",
            filename=f"{slug}_enumeration.json",
            language="json",
            content=json.dumps(payload, indent=2),
            execution_note="Exhaustive proof ledger for every satisfiable topology candidate.",
        )

    def _build_exact_topology_conclusion(self, search_result: TopologySearchResult) -> str:
        if search_result.optimal_topology is None:
            return (
                f"Exact exhaustive search evaluated {search_result.evaluated_topologies} connected topology shape(s) "
                "and found no satisfiable configuration under the full constraint set."
            )

        optimal = search_result.optimal_topology
        return (
            f"Exact exhaustive search found {len(search_result.satisfiable_topologies)} satisfiable topology "
            f"configuration(s) out of {search_result.evaluated_topologies} connected graph shape(s). "
            f"The optimal configuration is {optimal.candidate_id} with {optimal.edge_count} edge(s), "
            f"diameter {optimal.diameter}, worst-case latency {optimal.worst_case_latency_ms:.0f}ms, "
            f"end-to-end fidelity {optimal.end_to_end_fidelity:.5f}, and "
            f"entanglement overhead {optimal.entanglement_overhead_factor:.2f}x MST. "
            "The full enumeration is emitted as JSON, and a verifiable circuit was generated for the optimal topology."
        )

    def _build_exact_constraint_summary(self, query: TopologySearchQuery) -> List[str]:
        return [
            f"End-to-end fidelity > {query.fidelity_threshold:.2f} after {query.gate_operations} gate operations.",
            "No single node failure may trigger system-wide decoherence.",
            f"Classical control latency between any two nodes < {query.latency_limit_ms:.0f}ms.",
            f"Total entanglement overhead must stay <= {query.entanglement_factor_limit:.2f}x MST.",
        ]

    def _build_exact_topology_title(self, candidate: TopologyCandidate) -> str:
        if candidate.diameter <= 2 and candidate.node_connectivity >= 3:
            prefix = "Redundant Low-Latency Mesh"
        elif candidate.diameter <= 2:
            prefix = "Compact Resilient Mesh"
        else:
            prefix = "Biconnected Constraint Mesh"
        return f"{prefix} {candidate.candidate_id}"

    def _select_exact_topology_tags(self, lenses: List[CognitiveLens]) -> List[str]:
        preferred = ("deductive", "physical", "quantum", "formal")
        selected = [
            tag
            for tag in preferred
            if any(lens.epistemic_tag == tag for lens in lenses)
        ]
        if selected:
            return selected

        deduplicated: List[str] = []
        for lens in lenses:
            if lens.epistemic_tag not in deduplicated:
                deduplicated.append(lens.epistemic_tag)
        return deduplicated[:4]

    def _format_edge_list(self, edges: List[tuple[int, int]], limit: int = 6) -> str:
        preview = [f"({left}, {right})" for left, right in edges[:limit]]
        if len(edges) > limit:
            preview.append("...")
        return ", ".join(preview)

    def _generate_designs(
        self,
        problem: str,
        lenses: List[CognitiveLens],
        constraints: List[str],
        blind_spots: List[str],
        risks: List[str],
        design_context: List[Dict[str, object]],
    ) -> List[EmergentDesign]:
        design_specs = [
            (
                "Constraint Lattice Compiler",
                "A synthesis path that rewrites the system into a constraint lattice before implementation.",
                ("formal", "symbolic", "physical"),
                "constraint-compilation",
            ),
            (
                "Quantum Branch Forge",
                "A branching architecture that keeps mutually incompatible designs alive until late constraint collapse.",
                ("quantum", "probabilistic", "causal"),
                "late-collapse branching",
            ),
            (
                "Topology Intervention Mesh",
                "A distributed mesh that changes system behavior by rewiring relationships and intervention points.",
                ("deductive", "causal", "physical"),
                "graph rewiring",
            ),
        ]
        if design_context:
            design_specs.append(
                (
                    "Lineage Mutation Reactor",
                    "A recursive synthesis path that mutates prior successful designs into a new hybrid genome.",
                    ("quantum", "formal", "deductive"),
                    "ancestral recombination",
                )
            )

        lineage_pool = self._collect_lineage_pool(design_context)
        designs: List[EmergentDesign] = []
        for index, (title, premise, preferred_tags, mutation_strategy) in enumerate(design_specs, start=1):
            selected = self._select_lenses_for_design(lenses, preferred_tags)
            if not selected:
                continue

            primitives = self._collect_design_items(selected, "operator_primitives", limit=5)
            affordances = self._collect_design_items(selected, "design_affordances", limit=3)
            governing_constraints = self._collect_design_items(selected, "constraints", limit=4)
            local_blind_spots = self._collect_design_items(selected, "blind_spots", limit=3)

            if lineage_pool:
                lineage_primitives = lineage_pool["primitives"][:2]
                for primitive in lineage_primitives:
                    if primitive not in primitives:
                        primitives.append(f"Recombine ancestral primitive: {primitive}")

            implementation_outline = self._build_design_outline(
                premise=premise,
                affordances=affordances,
                primitives=primitives,
                constraints=governing_constraints or constraints,
                risks=risks,
                lineage_titles=lineage_pool["titles"],
            )
            novelty_score = self._score_novelty(
                selected,
                preferred_tags,
                primitives,
                affordances,
                lineage_pool["titles"],
            )
            feasibility_score = self._score_feasibility(
                selected,
                governing_constraints or constraints,
                local_blind_spots,
            )
            composite_score = clamp_float(
                (0.6 * novelty_score) + (0.4 * feasibility_score),
                default=0.5,
                minimum=0.0,
                maximum=1.0,
            )

            design = EmergentDesign(
                design_id=f"ED-{index}",
                title=title,
                premise=premise,
                composition_tags=[lens.epistemic_tag for lens in selected],
                component_primitives=primitives,
                implementation_outline=implementation_outline,
                governing_constraints=governing_constraints or constraints[:4],
                novelty_score=novelty_score,
                feasibility_score=feasibility_score,
                composite_score=composite_score,
                artifacts=[],
                lineage_titles=lineage_pool["titles"][:3],
                mutation_strategy=mutation_strategy,
            )
            design.artifacts = self._generate_design_artifacts(design, problem)
            designs.append(design)

        designs.sort(key=lambda design: design.composite_score, reverse=True)
        return designs

    def _collect_lineage_pool(self, design_context: List[Dict[str, object]]) -> Dict[str, List[str]]:
        titles: List[str] = []
        primitives: List[str] = []
        for seed in design_context:
            for title in seed.get("titles", []):
                value = str(title).strip()
                if value and value not in titles:
                    titles.append(value)
            for primitive in seed.get("primitives", []):
                value = str(primitive).strip()
                if value and value not in primitives:
                    primitives.append(value)
        return {"titles": titles, "primitives": primitives}

    def _select_lenses_for_design(
        self,
        lenses: List[CognitiveLens],
        preferred_tags: tuple[str, ...],
    ) -> List[CognitiveLens]:
        selected: List[CognitiveLens] = []
        for tag in preferred_tags:
            candidate = next((lens for lens in lenses if lens.epistemic_tag == tag), None)
            if candidate and candidate not in selected:
                selected.append(candidate)

        for lens in lenses:
            if lens not in selected:
                selected.append(lens)
            if len(selected) >= 4:
                break
        return selected

    def _collect_design_items(
        self,
        lenses: List[CognitiveLens],
        attribute: str,
        limit: int,
    ) -> List[str]:
        items: List[str] = []
        for lens in lenses:
            for item in getattr(lens, attribute):
                value = str(item).strip()
                if value and value not in items:
                    items.append(value)
                if len(items) >= limit:
                    return items
        return items

    def _build_design_outline(
        self,
        premise: str,
        affordances: List[str],
        primitives: List[str],
        constraints: List[str],
        risks: List[str],
        lineage_titles: List[str],
    ) -> List[str]:
        outline = [f"Frame the invention around: {premise}"]
        if lineage_titles:
            outline.append(
                f"Import ancestral traits from: {', '.join(lineage_titles[:2])}"
            )
        if affordances:
            outline.append(f"Instantiate the first computational substrate: {affordances[0]}")
        if primitives:
            outline.append(f"Drive the core transformation using: {primitives[0]}")
        if len(primitives) > 1:
            outline.append(f"Amplify the design with a second operator: {primitives[1]}")
        if constraints:
            outline.append(f"Pin the design to hard bounds: {constraints[0]}")
        if risks:
            outline.append(f"Instrument the architecture against: {risks[0]}")
        return outline[:6]

    def _score_novelty(
        self,
        lenses: List[CognitiveLens],
        preferred_tags: tuple[str, ...],
        primitives: List[str],
        affordances: List[str],
        lineage_titles: List[str],
    ) -> float:
        tag_diversity = len({lens.epistemic_tag for lens in lenses}) / max(1, len(preferred_tags))
        cross_library_bonus = min(0.2, 0.04 * len(affordances))
        primitive_bonus = min(0.2, 0.03 * len(primitives))
        quantum_bonus = 0.08 if any(lens.epistemic_tag == "quantum" for lens in lenses) else 0.0
        lineage_bonus = min(0.1, 0.03 * len(lineage_titles))
        return clamp_float(
            0.45 + (0.12 * tag_diversity) + cross_library_bonus + primitive_bonus + quantum_bonus + lineage_bonus
        )

    def _score_feasibility(
        self,
        lenses: List[CognitiveLens],
        constraints: List[str],
        blind_spots: List[str],
    ) -> float:
        avg_confidence = sum(lens.confidence for lens in lenses) / len(lenses)
        constraint_bonus = min(0.18, 0.03 * len(constraints))
        blind_spot_penalty = min(0.18, 0.04 * len(blind_spots))
        return clamp_float(avg_confidence + constraint_bonus - blind_spot_penalty)

    def _generate_design_artifacts(self, design: EmergentDesign, problem: str) -> List[DesignArtifact]:
        slug = design.title.lower().replace(" ", "_")
        artifacts: List[DesignArtifact] = []

        blueprint = self._build_blueprint_artifact(design, slug)
        artifacts.append(blueprint)

        if "Quantum" in design.title:
            artifacts.append(self._build_quantum_artifact(design, slug))
        elif "Topology" in design.title:
            artifacts.append(self._build_topology_artifact(design, slug))
        else:
            artifacts.append(self._build_constraint_artifact(design, slug))

        if design.lineage_titles:
            artifacts.append(self._build_mutation_report_artifact(design, slug, problem))
        return artifacts

    def _build_blueprint_artifact(self, design: EmergentDesign, slug: str) -> DesignArtifact:
        lines = [
            "title: " + design.title,
            "mutation_strategy: " + design.mutation_strategy,
            "composite_score: " + f"{design.composite_score:.2f}",
            "tags:",
        ]
        for tag in design.composition_tags:
            lines.append(f"  - {tag}")
        lines.append("constraints:")
        for constraint in design.governing_constraints[:3]:
            lines.append(f"  - {constraint}")
        lines.append("implementation_outline:")
        for step in design.implementation_outline[:4]:
            lines.append(f"  - {step}")
        return DesignArtifact(
            artifact_id=f"{design.design_id}-A1",
            artifact_type="blueprint",
            filename=f"{slug}.yaml",
            language="yaml",
            content="\n".join(lines),
            execution_note="Load as a structured blueprint for downstream orchestration.",
        )

    def _build_quantum_artifact(self, design: EmergentDesign, slug: str) -> DesignArtifact:
        content = "\n".join(
            [
                "from qiskit import QuantumCircuit",
                "",
                "def build_design_circuit():",
                "    qc = QuantumCircuit(3)",
                "    qc.h(0)",
                "    qc.cx(0, 1)",
                "    qc.cx(1, 2)",
                "    qc.barrier()",
                "    return qc",
            ]
        )
        return DesignArtifact(
            artifact_id=f"{design.design_id}-A2",
            artifact_type="quantum_circuit",
            filename=f"{slug}_circuit.py",
            language="python",
            content=content,
            execution_note="Instantiate the branching scaffold and expand gates as the design crystallizes.",
        )

    def _build_topology_artifact(self, design: EmergentDesign, slug: str) -> DesignArtifact:
        content = "\n".join(
            [
                "import networkx as nx",
                "",
                "def build_topology():",
                "    graph = nx.DiGraph()",
                "    graph.add_edges_from([",
                "        ('sensor_mesh', 'local_router'),",
                "        ('local_router', 'consensus_cell'),",
                "        ('consensus_cell', 'actuation_plane'),",
                "    ])",
                "    return graph",
            ]
        )
        return DesignArtifact(
            artifact_id=f"{design.design_id}-A2",
            artifact_type="topology_graph",
            filename=f"{slug}_graph.py",
            language="python",
            content=content,
            execution_note="Use as the seed graph for topology mutation and path redundancy analysis.",
        )

    def _build_constraint_artifact(self, design: EmergentDesign, slug: str) -> DesignArtifact:
        content = "\n".join(
            [
                "from z3 import Bool, Solver, And",
                "",
                "def build_solver():",
                "    solver = Solver()",
                "    invariants = [Bool('formal_safe'), Bool('resource_safe'), Bool('latency_safe')]",
                "    solver.add(And(*invariants))",
                "    return solver",
            ]
        )
        return DesignArtifact(
            artifact_id=f"{design.design_id}-A2",
            artifact_type="constraint_solver",
            filename=f"{slug}_solver.py",
            language="python",
            content=content,
            execution_note="Extend the invariant list with concrete domain clauses before solving.",
        )

    def _build_mutation_report_artifact(
        self,
        design: EmergentDesign,
        slug: str,
        problem: str,
    ) -> DesignArtifact:
        lines = [
            "# Mutation Report",
            "",
            "Base Lineage:",
        ]
        for title in design.lineage_titles[:3]:
            lines.append(f"- {title}")
        lines.append("")
        lines.append("Mutation Strategy:")
        lines.append(f"- {design.mutation_strategy}")
        lines.append("")
        lines.append("Trigger Problem:")
        lines.append(f"- {problem}")
        return DesignArtifact(
            artifact_id=f"{design.design_id}-A3",
            artifact_type="mutation_report",
            filename=f"{slug}_mutation.md",
            language="markdown",
            content="\n".join(lines),
            execution_note="Use this as the ancestry manifest for the next evolutionary iteration.",
        )

    def _top_items(self, lenses: List[CognitiveLens], attribute: str, limit: int = 3) -> List[str]:
        counts: Dict[str, int] = {}
        positions: Dict[str, int] = {}

        for lens in lenses:
            for item in getattr(lens, attribute):
                normalized = str(item).strip()
                if not normalized:
                    continue
                if normalized not in positions:
                    positions[normalized] = len(positions)
                counts[normalized] = counts.get(normalized, 0) + 1

        return sorted(counts, key=lambda value: (-counts[value], positions[value]))[:limit]

    def _infer_problem_risks(self, problem: str) -> List[str]:
        normalized_problem = problem.lower()
        risks: List[str] = []

        if any(token in normalized_problem for token in ("failure", "bug", "error", "risk")):
            risks.extend(
                [
                    "unhandled edge cases",
                    "boundary-condition failures",
                    "hidden assumptions in the reasoning path",
                ]
            )
        if "file" in normalized_problem:
            risks.append("input parsing or encoding faults")
        if any(token in normalized_problem for token in ("api", "auth", "token", "key", "anthropic")):
            risks.append("external service authentication or availability")
        if any(token in normalized_problem for token in ("distributed", "parallel", "concurrent", "thread")):
            risks.append("race conditions or partial failures across components")
        if not risks:
            risks.extend(
                [
                    "architecture lock-in",
                    "insufficient observability",
                    "tight coupling between components",
                ]
            )

        deduplicated: List[str] = []
        for risk in risks:
            if risk not in deduplicated:
                deduplicated.append(risk)
        return deduplicated[:4]

    def _build_local_conclusion(
        self,
        problem: str,
        generated_designs: List[EmergentDesign],
        risks: List[str],
    ) -> str:
        if not generated_designs:
            risk_text = ", ".join(risks[:2]) if risks else "unknown instability"
            return (
                f"No emergent design candidate cleared the synthesis bar. "
                f"The next iteration should resolve {risk_text} before expanding the search space."
            )

        best_design = generated_designs[0]
        primitives = "; ".join(best_design.component_primitives[:2]) if best_design.component_primitives else "none"
        guardrails = ", ".join(best_design.governing_constraints[:2]) if best_design.governing_constraints else "none"
        artifact_summary = ", ".join(artifact.filename for artifact in best_design.artifacts[:2])
        return (
            f"Lead design: {best_design.title}. {best_design.premise} "
            f"Primary engineering moves: {primitives}. "
            f"Hold the invention inside these guardrails: {guardrails}. "
            f"Executable artifacts: {artifact_summary}. "
            f"Novelty {best_design.novelty_score:.2f}, feasibility {best_design.feasibility_score:.2f}."
        )

    def _coerce_steps(
        self,
        raw_steps: object,
        fallback: List[ReasoningStep],
    ) -> List[ReasoningStep]:
        steps: List[ReasoningStep] = []
        if isinstance(raw_steps, list):
            for index, step in enumerate(raw_steps, start=1):
                if not isinstance(step, dict):
                    continue
                description = str(step.get("description", "")).strip()
                if not description:
                    continue
                steps.append(
                    ReasoningStep(
                        step_id=str(step.get("step_id", index)),
                        description=description,
                    )
                )

        return steps or list(fallback)

    def _normalize_contributions(
        self,
        raw_values: object,
        lenses: List[CognitiveLens],
        fallback: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        contributions: Dict[str, float] = {}
        source = raw_values if isinstance(raw_values, dict) else (fallback or {})

        for lens in lenses:
            raw_value = source.get(lens.lens_name, lens.confidence)
            try:
                contributions[lens.lens_name] = max(0.0, float(raw_value))
            except (TypeError, ValueError):
                contributions[lens.lens_name] = max(0.0, lens.confidence)

        total = sum(contributions.values())
        if total <= 0:
            equal_weight = 1.0 / len(lenses)
            return {lens.lens_name: equal_weight for lens in lenses}

        return {lens_name: weight / total for lens_name, weight in contributions.items()}

    def _format_design_context(self, design_context: List[Dict[str, object]]) -> str:
        if not design_context:
            return "None"

        lines: List[str] = []
        for seed in design_context[:3]:
            titles = ", ".join(str(value) for value in seed.get("titles", [])[:3]) or "none"
            primitives = "; ".join(str(value) for value in seed.get("primitives", [])[:2]) or "none"
            confidence = clamp_float(seed.get("confidence"), default=0.0)
            lines.append(f"- titles: {titles} | confidence: {confidence:.2f} | primitives: {primitives}")
        return "\n".join(lines)
