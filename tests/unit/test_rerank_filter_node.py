# RET-C2-006 — Unit Tests: RerankFilterNode (inner domain node 3)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# execute(self, state) takes NO config parameter — every call here goes through
# node(state); score_threshold / top_k overrides are exercised by seeding
# state["retrieval_config"], never a second execute() argument.
#
# Mirrors docs/03_test_spec.md §2.5.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.rerank_filter_node import RerankFilterNode
from src.schemas.state import from_json, to_json


def _doc(doc_id, score, category="return_eligibility"):
    return {
        "id": doc_id,
        "title": f"entry {doc_id}",
        "category": category,
        "source": "seeded kb",
        "score": score,
        "excerpt": "excerpt text",
    }


def _make_state(candidates, **extra) -> dict:
    state = {
        "retrieved_documents": to_json(candidates),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestThresholdAndCap:
    def test_rrf_01_default_threshold_drops_weak_candidates(self):
        result = RerankFilterNode()(_make_state([_doc("kb-a", 0.9), _doc("kb-b", 0.1)]))
        kept = from_json(result["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]  # 0.1 < default 0.25 floor

    def test_rrf_02_state_score_threshold_override(self):
        # C2 retired: config-passing is now state["retrieval_config"], still node(state).
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.3)],
            retrieval_config=to_json({"score_threshold": 0.5}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_03_state_top_k_override(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.8), _doc("kb-c", 0.7)],
            retrieval_config=to_json({"top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_ranked_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RerankFilterNode()(_make_state([_doc("kb-a", 0.9)]))
        assert isinstance(result["ranked_documents"], str)


class TestCategoryBoost:
    def test_rrf_04_matching_category_is_boosted_and_reranked(self):
        state = _make_state(
            [_doc("kb-a", 0.30, category="exchange_policy"), _doc("kb-b", 0.25, category="warranty")],
            query_filters=to_json({"category": "warranty", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-b", "kb-a"]
        assert kept[0]["score"] == 0.35  # 0.25 + 0.1 category boost

    def test_rrf_05_boost_is_capped_at_one(self):
        state = _make_state(
            [_doc("kb-a", 0.95, category="warranty")],
            query_filters=to_json({"category": "warranty", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert kept[0]["score"] == 1.0


class TestCallerTopK:
    def test_rrf_06_stricter_caller_top_k_wins(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.8), _doc("kb-c", 0.7)],
            query_filters=to_json({"category": None, "top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_06_looser_caller_top_k_does_not_widen(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.8), _doc("kb-c", 0.7)],
            query_filters=to_json({"category": None, "top_k": 10}),
            retrieval_config=to_json({"top_k": 2, "score_threshold": 0.25}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]


class TestRobustness:
    def test_rrf_07_garbage_candidates_are_skipped_or_dropped(self):
        candidates = [
            "not-a-dict",
            {"id": "kb-bad", "title": "b", "category": "x", "source": "s", "score": "NaN?", "excerpt": "e"},
            _doc("kb-a", 0.9),
        ]
        kept = from_json(RerankFilterNode()(_make_state(candidates))["ranked_documents"])
        # The string entry is skipped; the uncoercible score becomes 0.0 and
        # falls below the relevance floor.
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_08_deterministic_tie_break_by_id(self):
        kept = from_json(RerankFilterNode()(_make_state([_doc("kb-b", 0.5), _doc("kb-a", 0.5)]))["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]


class TestSeededConfigMustBeFinite:
    """A hand-seeded retrieval_config gets the same finite+bounded rule as the
    graph-forwarded one, so the guarantee does not depend on the path in.

    The relevance floor is the case that matters: NaN compares False against
    everything, so clamping it (`max(0.0, min(1.0, nan))` returns 1.0) would
    silently replace the declared floor with the strictest possible one and
    drop every passage — the agent would answer "insufficient coverage" for
    questions the corpus covers.
    """

    @pytest.mark.parametrize(
        "bad_threshold",
        [float("nan"), float("inf"), float("-inf"), "0.5", True, None, [], {}],
    )
    def test_non_finite_threshold_falls_back_to_the_declared_default(self, bad_threshold):
        state = _make_state(
            [_doc("kb-001", 0.9), _doc("kb-002", 0.1)],
            retrieval_config=to_json({"score_threshold": bad_threshold}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        # Default floor is 0.25: the 0.9 candidate survives, the 0.1 does not.
        assert [c["id"] for c in kept] == ["kb-001"]

    @pytest.mark.parametrize("bad_top_k", [float("nan"), float("inf"), "3", True, None])
    def test_non_finite_top_k_falls_back_to_the_declared_default(self, bad_top_k):
        state = _make_state(
            [_doc("kb-001", 0.9), _doc("kb-002", 0.8), _doc("kb-003", 0.7)],
            retrieval_config=to_json({"top_k": bad_top_k}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert len(kept) == 3  # default top_k = 4, so all three survive

    def test_boolean_caller_top_k_does_not_collapse_the_result_set(self):
        """True satisfies isinstance(int) and equals 1 — it must not be read as
        a top_k of one."""
        state = _make_state(
            [_doc("kb-001", 0.9), _doc("kb-002", 0.8)],
            query_filters=to_json({"category": None, "top_k": True}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert len(kept) == 2

    @pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), "high", None, True])
    def test_non_finite_candidate_score_is_treated_as_zero_and_dropped(self, bad_score):
        state = _make_state([_doc("kb-001", bad_score)])
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert kept == []
