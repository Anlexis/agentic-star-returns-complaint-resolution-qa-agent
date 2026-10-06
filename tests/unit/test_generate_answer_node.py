# RET-C2-006 — Unit Tests: GenerateAnswerNode (inner domain node 4)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# The grounded answer and citations are DOMAIN fields (not input-mask scan
# targets), so Title-Case knowledge-base titles inside them are safe to
# assert on.
#
# Mirrors docs/03_test_spec.md §2.5.
# Deterministic — rule-assembled from ranked_documents only (grounded by
# construction; no LLM, no network). framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json


def _ranked(*entries):
    return to_json(list(entries))


def _doc(doc_id, title, excerpt, source="seeded kb"):
    return {
        "id": doc_id,
        "title": title,
        "category": "return_eligibility",
        "source": source,
        "score": 0.9,
        "excerpt": excerpt,
    }


def _make_state(ranked_documents, query="return window for defective items", **extra) -> dict:
    state = {
        "ranked_documents": ranked_documents,
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestGroundedAnswer:
    def test_answer_carries_numbered_citation_markers(self):
        ranked = _ranked(
            _doc("kb-001", "Standard return eligibility window", "general merchandise may be returned within 30 days."),
            _doc("kb-002", "Defective or damaged item returns", "defective items may be returned at any time."),
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        answer = result["grounded_answer"]
        assert "[1] Standard return eligibility window:" in answer
        assert "[2] Defective or damaged item returns:" in answer

    def test_lead_sentence_is_static_and_never_quotes_the_query(self):
        """The question SELECTS passages; it is never rendered. Echoing it back
        would put caller-controlled free text on the external surface."""
        marker = "distinctive-caller-marker-string"
        ranked = _ranked(_doc("kb-001", "Standard return eligibility window", "excerpt."))
        result = GenerateAnswerNode()(_make_state(ranked, query=f"return window {marker}"))
        answer = result["grounded_answer"]
        assert marker not in answer
        assert answer.startswith("Based on the returns & complaints knowledge base, the following passages apply")

    def test_lead_sentence_is_identical_with_no_query(self):
        ranked = _ranked(_doc("kb-001", "Standard return eligibility window", "excerpt."))
        with_query = GenerateAnswerNode()(_make_state(ranked, query="anything"))["grounded_answer"]
        without_query = GenerateAnswerNode()(_make_state(ranked, query=""))["grounded_answer"]
        assert with_query == without_query

    def test_citations_mirror_ranked_order(self):
        ranked = _ranked(
            _doc("kb-001", "Standard return eligibility window", "a.", source="Returns & Exchanges Policy"),
            _doc("kb-002", "Defective or damaged item returns", "b."),
        )
        citations = from_json(GenerateAnswerNode()(_make_state(ranked))["citations"])
        assert [c["ref"] for c in citations] == [1, 2]
        assert [c["id"] for c in citations] == ["kb-001", "kb-002"]
        assert citations[0]["source"] == "Returns & Exchanges Policy"

    def test_citations_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        ranked = _ranked(_doc("kb-001", "Standard return eligibility window", "a."))
        result = GenerateAnswerNode()(_make_state(ranked))
        assert isinstance(result["citations"], str)

    def test_answer_is_grounded_in_ranked_passages_only(self):
        ranked = _ranked(
            _doc("kb-001", "Standard return eligibility window", "return window is 30 days from purchase.")
        )
        answer = GenerateAnswerNode()(_make_state(ranked))["grounded_answer"]
        # Every content line traces to the single ranked passage.
        assert "return window is 30 days from purchase." in answer
        assert "[2]" not in answer


class TestNoCoverage:
    def test_empty_ranked_set_yields_no_coverage_answer(self):
        result = GenerateAnswerNode()(_make_state(_ranked()))
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []

    def test_missing_ranked_field_is_treated_as_no_coverage(self):
        state = _make_state(None)
        del state["ranked_documents"]
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
