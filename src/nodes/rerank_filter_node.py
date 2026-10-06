"""AgentCore Platform v1.0"""

# RET-C2-006 - RerankFilterNode
# Domain node 3: rerank the retrieval candidates and enforce the relevance
# floor. Deterministic: a small category-match boost on top of the retrieval
# score, drop everything below `score_threshold`, cap the survivors at
# `top_k`.
#
# Config knobs: this node takes NO extra execute() parameters. `top_k` /
# `score_threshold` are read from the state-seeded `retrieval_config` JSON
# field (forwarded by ReturnsComplaintSearchGraphNode._parent_config() ->
# DomainWorkflowGraph._extra_initial_state()), falling back to module
# defaults that mirror config/config.yaml when unseeded. A caller-supplied
# top_k override (query_filters) wins when it is stricter.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.schemas.agent_status import AgentStatus
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import finite_in_range, from_json, to_json

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "top_k": 4,
    "score_threshold": 0.25,
}

# Boost applied when a candidate's category matches the caller's filter.
_CATEGORY_BOOST = 0.1


def _resolve_retrieval_config(state: AgentState) -> Dict[str, Any]:
    """Effective retrieval config: state retrieval_config (manifest-forwarded) > defaults."""
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


class RerankFilterNode(FunctionNode):
    """Rerank candidates, apply the score threshold, cap at top_k.

    Input state keys:
        retrieved_documents: JSON list of scored candidates (from RetrieveNode)
        query_filters:       JSON dict with optional category / top_k override
        retrieval_config:    forwarded manifest retrieval block (JSON)

    Output state keys (partial dict):
        ranked_documents: JSON list of surviving passages (score desc, <= top_k)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_documents"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}
        retrieval_cfg = _resolve_retrieval_config(state)

        # The forwarded config is already validated at the graph boundary; this
        # second guard covers a node invoked directly with a hand-seeded
        # retrieval_config, so the same rule holds on every path in.
        top_k = int(finite_in_range(retrieval_cfg.get("top_k"), 1, 20, float(_DEFAULT_RETRIEVAL["top_k"])))
        # A stricter caller override (already validated by InputValidateNode) wins.
        # bool is excluded explicitly: True would satisfy isinstance(int) and
        # silently collapse the result set to one passage.
        caller_top_k = filters.get("top_k")
        if isinstance(caller_top_k, int) and not isinstance(caller_top_k, bool) and 1 <= caller_top_k < top_k:
            top_k = caller_top_k

        # A non-finite relevance floor is the dangerous case: NaN compares False
        # against every candidate, so clamping it would silently replace the
        # declared floor instead of applying it.
        score_threshold = finite_in_range(
            retrieval_cfg.get("score_threshold"), 0.0, 1.0, float(_DEFAULT_RETRIEVAL["score_threshold"])
        )

        category = filters.get("category")

        reranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            entry = dict(candidate)  # local copy - inputs stay immutable
            score = finite_in_range(entry.get("score"), 0.0, 1.0, 0.0)
            if category and str(entry.get("category", "")).lower() == str(category).lower():
                score = min(1.0, score + _CATEGORY_BOOST)
            entry["score"] = round(score, 4)
            reranked.append(entry)

        # Deterministic ordering: score desc, then id asc for stable ties.
        reranked.sort(key=lambda c: (-c.get("score", 0.0), str(c.get("id", ""))))

        kept = [c for c in reranked if c.get("score", 0.0) >= score_threshold][:top_k]
        dropped = len(reranked) - len(kept)

        # Audit: rerank + relevance floor applied.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "kept": len(kept),
                "dropped": dropped,
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {"ranked_documents": to_json(kept)}
