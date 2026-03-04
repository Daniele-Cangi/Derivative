import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import networkx as nx

from core.json_utils import clamp_float


@dataclass
class TopologySearchQuery:
    node_count: int
    physical_error_rate: float
    fidelity_threshold: float
    gate_operations: int
    latency_limit_ms: float
    entanglement_factor_limit: float
    hop_latency_ms: float = 10.0
    logical_error_exponent: int = 2


@dataclass
class TopologyCandidate:
    candidate_id: str
    edge_count: int
    diameter: int
    degree_sequence: Tuple[int, ...]
    canonical_edges: List[Tuple[int, int]]
    worst_case_latency_ms: float
    end_to_end_fidelity: float
    entanglement_overhead_factor: float
    node_connectivity: int
    novelty_score: float
    feasibility_score: float
    composite_score: float


@dataclass
class TopologySearchResult:
    query: TopologySearchQuery
    evaluated_topologies: int
    satisfiable_topologies: List[TopologyCandidate]
    optimal_topology: Optional[TopologyCandidate]
    assumptions: List[str]


def parse_topology_search_query(problem: str) -> Optional[TopologySearchQuery]:
    lowered = problem.lower()
    if "topological configuration" not in lowered and "topological configurations" not in lowered:
        return None
    if "minimum spanning tree" not in lowered:
        return None

    node_count = _parse_int(problem, r"\bN\s*=\s*(\d+)")
    physical_error_rate = _parse_float(problem, r"\bp\s*=\s*([0-9]*\.?[0-9]+)")
    fidelity_threshold = _parse_float(problem, r"fidelity\s*>\s*([0-9]*\.?[0-9]+)")
    gate_operations = _parse_int(problem, r"after\s+(\d+)\s+gate\s+operations")
    latency_limit_ms = _parse_float(problem, r"latency.*?<\s*([0-9]*\.?[0-9]+)\s*ms")
    entanglement_factor_limit = _parse_float(problem, r"does not exceed\s*([0-9]*\.?[0-9]+)x")

    if None in (
        node_count,
        physical_error_rate,
        fidelity_threshold,
        gate_operations,
        latency_limit_ms,
        entanglement_factor_limit,
    ):
        return None

    if node_count < 2 or node_count > 7:
        return None

    return TopologySearchQuery(
        node_count=node_count,
        physical_error_rate=physical_error_rate,
        fidelity_threshold=fidelity_threshold,
        gate_operations=gate_operations,
        latency_limit_ms=latency_limit_ms,
        entanglement_factor_limit=entanglement_factor_limit,
    )


def solve_topology_search(query: TopologySearchQuery) -> TopologySearchResult:
    atlas_graphs = [
        nx.convert_node_labels_to_integers(graph.copy())
        for graph in nx.graph_atlas_g()
        if graph.number_of_nodes() == query.node_count and nx.is_connected(graph)
    ]

    assumptions = [
        f"Exhaustive search is over all connected non-isomorphic graphs with {query.node_count} nodes from the NetworkX graph atlas.",
        f"Classical latency is approximated as {query.hop_latency_ms:.0f}ms per hop, so the latency bound is enforced on graph diameter.",
        f"Logical error is approximated as p^{query.logical_error_exponent}, reflecting a simplified error-correction suppression model.",
        "Entanglement overhead is measured as |E| / (N-1), using the unweighted minimum spanning tree edge count as the baseline.",
    ]

    max_edges = int(math.floor(query.entanglement_factor_limit * (query.node_count - 1)))
    satisfiable: List[TopologyCandidate] = []

    for graph in atlas_graphs:
        edge_count = graph.number_of_edges()
        if edge_count > max_edges:
            continue
        if not nx.is_biconnected(graph):
            continue

        diameter = nx.diameter(graph)
        latency_ms = diameter * query.hop_latency_ms
        if latency_ms >= query.latency_limit_ms:
            continue

        logical_error_rate = query.physical_error_rate ** query.logical_error_exponent
        routed_gate_operations = query.gate_operations + max(0, diameter - 1)
        fidelity = (1.0 - logical_error_rate) ** routed_gate_operations
        if fidelity <= query.fidelity_threshold:
            continue

        node_connectivity = nx.node_connectivity(graph)
        degree_sequence = tuple(sorted((degree for _, degree in graph.degree()), reverse=True))
        entanglement_overhead_factor = edge_count / max(1, query.node_count - 1)

        novelty_score = _score_novelty(
            graph,
            edge_count=edge_count,
            max_edges=max_edges,
            node_count=query.node_count,
            degree_sequence=degree_sequence,
        )
        feasibility_score = _score_feasibility(
            query=query,
            edge_count=edge_count,
            fidelity=fidelity,
            latency_ms=latency_ms,
            node_connectivity=node_connectivity,
            max_edges=max_edges,
        )
        composite_score = clamp_float(
            (0.55 * feasibility_score) + (0.45 * novelty_score),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )

        satisfiable.append(
            TopologyCandidate(
                candidate_id="",
                edge_count=edge_count,
                diameter=diameter,
                degree_sequence=degree_sequence,
                canonical_edges=sorted((min(left, right), max(left, right)) for left, right in graph.edges()),
                worst_case_latency_ms=latency_ms,
                end_to_end_fidelity=fidelity,
                entanglement_overhead_factor=entanglement_overhead_factor,
                node_connectivity=node_connectivity,
                novelty_score=novelty_score,
                feasibility_score=feasibility_score,
                composite_score=composite_score,
            )
        )

    satisfiable.sort(
        key=lambda candidate: (
            -candidate.composite_score,
            -candidate.feasibility_score,
            candidate.edge_count,
            candidate.degree_sequence,
        )
    )
    for index, candidate in enumerate(satisfiable, start=1):
        candidate.candidate_id = f"T{index:03d}"

    optimal = satisfiable[0] if satisfiable else None
    return TopologySearchResult(
        query=query,
        evaluated_topologies=len(atlas_graphs),
        satisfiable_topologies=satisfiable,
        optimal_topology=optimal,
        assumptions=assumptions,
    )


def _score_novelty(
    graph: nx.Graph,
    edge_count: int,
    max_edges: int,
    node_count: int,
    degree_sequence: Tuple[int, ...],
) -> float:
    cycle_rank = edge_count - node_count + 1
    cycle_rank_max = max(1, max_edges - node_count + 1)
    cycle_component = cycle_rank / cycle_rank_max

    unique_degrees = len(set(degree_sequence))
    diversity_component = 1.0 - ((unique_degrees - 1) / max(1, node_count - 1))

    minimum_edges = node_count - 1
    mid_density = (minimum_edges + max_edges) / 2
    density_denominator = max(1.0, (max_edges - minimum_edges) / 2)
    density_component = 1.0 - min(1.0, abs(edge_count - mid_density) / density_denominator)

    return clamp_float(
        0.35 + (0.30 * cycle_component) + (0.15 * diversity_component) + (0.20 * density_component)
    )


def _score_feasibility(
    query: TopologySearchQuery,
    edge_count: int,
    fidelity: float,
    latency_ms: float,
    node_connectivity: int,
    max_edges: int,
) -> float:
    fidelity_margin = clamp_float(
        (fidelity - query.fidelity_threshold) / max(1e-9, 1.0 - query.fidelity_threshold),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )
    latency_margin = clamp_float(
        (query.latency_limit_ms - latency_ms) / max(1e-9, query.latency_limit_ms),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )
    minimum_edges = query.node_count - 1
    entanglement_margin = clamp_float(
        (max_edges - edge_count) / max(1, max_edges - minimum_edges),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )
    resilience_margin = clamp_float(
        (node_connectivity - 2) / max(1, query.node_count - 3),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )

    return clamp_float(
        0.35
        + (0.25 * fidelity_margin)
        + (0.20 * latency_margin)
        + (0.10 * entanglement_margin)
        + (0.10 * resilience_margin)
    )


def _parse_int(text: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return int(match.group(1))


def _parse_float(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return float(match.group(1))
