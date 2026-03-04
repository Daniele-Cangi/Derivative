from core.topology_solver import parse_topology_search_query, solve_topology_search


EXACT_QUERY = """
Given a quantum error-corrected distributed system with N=4 nodes,
where each node maintains a logical qubit with physical error rate p=0.001,
find all satisfiable topological configurations such that:

1. End-to-end fidelity > 0.95 after 50 gate operations
2. No single node failure causes system-wide decoherence
3. Classical control latency between any two nodes < 50ms
4. Total entanglement overhead does not exceed 3x the minimum spanning tree

Which configurations exist? Which is optimal under simultaneous
novelty + feasibility scoring? Emit a verifiable circuit.
""".strip()


def test_topology_solver_parses_structured_query():
    query = parse_topology_search_query(EXACT_QUERY)

    assert query is not None
    assert query.node_count == 4
    assert query.physical_error_rate == 0.001
    assert query.gate_operations == 50
    assert query.latency_limit_ms == 50


def test_topology_solver_enumerates_satisfiable_graphs():
    query = parse_topology_search_query(EXACT_QUERY)
    result = solve_topology_search(query)

    assert result.evaluated_topologies > 0
    assert len(result.satisfiable_topologies) == 3
    assert result.optimal_topology is not None
    assert result.optimal_topology.candidate_id == "T001"
    assert all(candidate.node_connectivity >= 2 for candidate in result.satisfiable_topologies)
