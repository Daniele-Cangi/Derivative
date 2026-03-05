from core.problem_classifier import ProblemClassifier, ProblemType


CRYPTO_QUERY = (
    "A cryptographic key exchange protocol uses 5 participants in a ring topology. "
    "Each participant i can only communicate directly with participant i+1 mod 5. "
    "Each communication channel has a probability of interception p=0.03. "
    "Find the minimum number of independent key-agreement rounds required so that the "
    "probability of at least one undetected interception across the full ring falls below 0.001. "
    "Then verify that your answer satisfies the constraint under both independent and correlated "
    "channel failure models."
)


TOPOLOGY_QUERY = (
    "Given a quantum error-corrected distributed system with N=7 nodes, "
    "where each node maintains a logical qubit with physical error rate p=0.001, "
    "find all satisfiable topological configurations such that fidelity > 0.95 "
    "after 50 gate operations, no single node failure causes system-wide decoherence, "
    "classical control latency between any two nodes < 50ms, and total entanglement "
    "overhead does not exceed 3x the minimum spanning tree."
)

RECURRENCE_QUERY = (
    "Given the linear recurrence f(n) = 3f(n-1) - f(n-2), with f(0)=1 and f(1)=2, "
    "find the closed-form formula, verify it satisfies the recurrence, determine the first n "
    "such that f(n) > 10^6, and compute the limit of f(n+1)/f(n)."
)

SURVIVAL_QUERY = (
    "A reliability network has 5 channels with per-hour failure probability p=0.03. "
    "Compute the probability the system survives 8 hours, and compute the expected number "
    "of hours before failure under independent and correlated failure models."
)

REGULAR_ENUMERATION_QUERY = (
    "Enumerate all connected 3-regular graph topologies on 6 nodes and report the total count. "
    "Then return the canonical best configuration."
)


def test_problem_classifier_marks_crypto_problem_probabilistic():
    classifier = ProblemClassifier()

    classification = classifier.classify(CRYPTO_QUERY)

    assert classification.primary_type == ProblemType.PROBABILISTIC
    assert classification.requires_numeric_answer is True
    assert "probability of" in classification.numeric_keywords
    assert classification.recommended_lens == "probabilistic"


def test_problem_classifier_marks_topology_problem_hybrid():
    classifier = ProblemClassifier()

    classification = classifier.classify(TOPOLOGY_QUERY)

    assert classification.primary_type == ProblemType.HYBRID
    assert ProblemType.PROBABILISTIC in classification.secondary_types
    assert classification.requires_numeric_answer is True
    assert classification.recommended_lens == "topological"


def test_problem_classifier_marks_linear_recurrence_symbolic():
    classifier = ProblemClassifier()

    classification = classifier.classify(RECURRENCE_QUERY)

    assert classification.primary_type == ProblemType.SYMBOLIC
    assert classification.recommended_lens == "symbolic"
    assert classification.requires_numeric_answer is True
    assert len(classification.explicit_objectives) >= 3


def test_problem_classifier_marks_survival_query_probabilistic():
    classifier = ProblemClassifier()
    classification = classifier.classify(SURVIVAL_QUERY)

    assert classification.primary_type == ProblemType.PROBABILISTIC
    assert classification.requires_numeric_answer is True


def test_problem_classifier_marks_regular_graph_query_hybrid():
    classifier = ProblemClassifier()
    classification = classifier.classify(REGULAR_ENUMERATION_QUERY)

    assert classification.primary_type == ProblemType.HYBRID
    assert classification.recommended_lens == "topological"
