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
