# RET-C2-006 — Unit Tests: retrieval quality over the seeded KB
#
# Golden-query suite: drives the REAL inner retrieval chain
# (InputValidateNode → RetrieveNode → RerankFilterNode) via node(state) /
# __call__ (ANONYMOUS inner nodes) against
# config/kb/returns_complaints_kb.json and pins the expected top hit per
# domain query. The scorer is deterministic (keyword field-weights, stable
# tie-break, EXACT token match / no stemming), so exact top-1 assertions are
# safe and catch KB / scorer / threshold regressions. Every (query, expected
# top-1) pair below was verified against the real scorer before being
# committed here.
#
# Mirrors docs/03_test_spec.md §2.9 (QUAL-01..QUAL-07).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
import pathlib

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB_IDS = {
    entry["id"]
    for entry in json.loads((_ROOT / "config" / "kb" / "returns_complaints_kb.json").read_text(encoding="utf-8"))
}

_DEFAULT_SCORE_THRESHOLD = 0.25  # mirrors config/agent.yaml retrieval block


def _search(payload: str) -> list[dict]:
    """Run the real inner retrieval chain and return the surviving passages."""
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "quality-session",
        "execution_time": {},
    }
    state.update(InputValidateNode()(state))
    state.update(RetrieveNode()(state))
    state.update(RerankFilterNode()(state))
    return from_json(state["ranked_documents"], [])


# (query, expected top-1 KB entry id) — verified against the deterministic
# keyword scorer (title 1.0 / tags 0.8 / content 0.5, averaged over query
# tokens, no stemming). One query per seeded KB entry.
_GOLDEN_QUERIES = [
    ("return eligibility window for unused items with the receipt", "kb-001"),
    ("defective damaged item replacement or refund", "kb-002"),
    ("how long does a card refund take to process", "kb-003"),
    ("final sale clearance items not eligible for return", "kb-004"),
    ("receipt lookup for store credit without proof of purchase", "kb-005"),
    ("exchange for a different size or color", "kb-006"),
    ("front-line complaint escalation to a supervisor", "kb-007"),
    ("escalation to regional or corporate customer service for safety", "kb-008"),
    ("online order return shipping label drop-off", "kb-009"),
    ("warranty claim outside the standard window", "kb-010"),
]


class TestGoldenQueries:
    @pytest.mark.parametrize(("query", "expected_id"), _GOLDEN_QUERIES)
    def test_qual_01_top_hit_per_golden_query(self, query, expected_id):
        kept = _search(query)
        assert kept, f"no passage cleared the relevance floor for: {query!r}"
        assert kept[0]["id"] == expected_id

    def test_qual_02_all_survivors_clear_the_relevance_floor(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["score"] >= _DEFAULT_SCORE_THRESHOLD

    def test_qual_03_survivor_ids_exist_in_the_seeded_kb(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["id"] in _KB_IDS


class TestPrecision:
    def test_qual_04_complaint_query_keeps_only_the_front_line_entry(self):
        # Off-topic passages score below the floor and are cut — precision, not
        # just recall.
        kept = _search(_GOLDEN_QUERIES[6][0])  # "front-line complaint escalation to a supervisor"
        assert [d["id"] for d in kept] == ["kb-007"]

    def test_qual_05_category_filter_restricts_to_that_category(self):
        payload = json.dumps({"query": "return policy after the standard window", "category": "warranty"})
        kept = _search(payload)
        assert kept, "warranty category carries a seeded entry"
        assert {d["category"] for d in kept} == {"warranty"}
        assert kept[0]["id"] == "kb-010"


class TestNoCoverage:
    def test_qual_06_out_of_domain_query_yields_no_survivors(self):
        assert _search("quantum telepathy sandwich recipes") == []

    def test_qual_07_no_coverage_produces_the_escalation_answer(self):
        state = {
            "ranked_documents": "[]",
            "search_query": "quantum telepathy sandwich recipes",
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "quality-session",
            "execution_time": {},
        }
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []
