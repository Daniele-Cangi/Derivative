from lenses.base import BaseLens
try:
    import networkx  # noqa: F401
except ImportError:
    networkx = None

class TopologicalLens(BaseLens):
    epistemic_tag = "deductive"
    lens_name = "Topological/Relational"
    library_focus = "NetworkX"
    analysis_focus = "relationships, graph structure, coupling, and bottlenecks"
    keywords = (
        "graph",
        "network",
        "flow",
        "dependency",
        "topology",
        "bottleneck",
        "distributed",
        "service",
    )
    default_constraints = (
        "Do not ignore coupling between components.",
        "Account for chokepoints and single points of failure.",
        "Preserve path dependencies across the system.",
    )
    default_blind_spots = (
        "Can miss low-level arithmetic or theorem-level correctness issues.",
        "May abstract away semantics while focusing on connectivity.",
    )

    def _collect_library_signals(self, problem: str):
        if networkx is None:
            return [], [], [], 0.0

        tokens = self._problem_tokens(problem)
        if len(tokens) < 2:
            return [], [], [], 0.0

        graph = networkx.Graph()
        for left, right in zip(tokens, tokens[1:]):
            if left == right:
                continue
            if graph.has_edge(left, right):
                graph[left][right]["weight"] += 1
            else:
                graph.add_edge(left, right, weight=1)

        if graph.number_of_nodes() == 0:
            return [], [], [], 0.0

        central_terms = sorted(graph.degree, key=lambda item: (-item[1], item[0]))[:3]
        note = (
            f"NetworkX extracted {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges; "
            f"central terms: {', '.join(name for name, _ in central_terms)}."
        )
        constraints = ["Inspect high-degree dependencies before changing central components."]
        blind_spots = []
        if graph.number_of_edges() < 2:
            blind_spots.append("Sparse dependency signals limit relational confidence.")

        return [note], constraints, blind_spots, 0.06

    def _build_operator_primitives(self, problem: str) -> list[str]:
        primitives = [
            "Rewire high-degree nodes before scaling the rest of the graph.",
            "Split chokepoints into parallel paths so failures degrade instead of cascade.",
        ]
        if "distributed" in problem.lower():
            primitives.append("Promote central coordination into topology-aware local orchestration.")
        return primitives

    def _build_design_affordances(self, problem: str) -> list[str]:
        return [
            "Use graph topology as a design search space, not only as a diagnostic overlay.",
            "Generate new architectures by moving edges, introducing buffers, and redistributing centrality.",
        ]
