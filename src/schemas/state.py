"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never a Pydantic model. Graph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption. Extend AgentState with agent-specific fields only. Do NOT
# add credentials, secrets, or Pydantic models.
#
# Msgpack safety: structured fields (dict / list[dict]) are stored as JSON
# STRINGS, not bare Python containers - a bare dict/list in a checkpointed
# State field is a state-safety violation. Producers serialize with
# to_json() on write; consumers deserialize with from_json() on read.
#
# RET-C2-006 - Customer Returns & Complaint Resolution Q&A Agent (Cat 2 RAG).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# PII / confidentiality note: direct identifiers (order numbers, card/account
# numbers, phone numbers, e-mail) that a customer-facing question happens to
# contain are stripped by PreProcessNode before any field is written to
# State, and the output boundary re-checks the rendered answer. Only the
# normalised search query, knowledge-base passage summaries, and the final
# grounded answer are persisted - never raw customer identifiers.

import json
import math
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def finite_in_range(value: Any, lo: float, hi: float, default: float) -> float:
    """Parse a numeric setting: a real, FINITE number within [lo, hi], else *default*.

    Rejects bools (which are ints in Python), non-numerics, and - critically -
    non-finite values. float() happily parses "NaN"/"Infinity", and IEEE NaN
    comparisons are ALWAYS False, so a NaN slipped into a threshold does not
    raise and does not clamp: min()/max() silently return the other operand and
    the declared floor is replaced by whatever the clamp happened to yield.
    Every number a node reads from a serialized or externally-seeded source
    comes through here.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    parsed = float(value)
    if not math.isfinite(parsed):
        return default
    return max(lo, min(hi, parsed))


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for RET-C2-006.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are Optional so the schema is valid at graph
    initialisation, before any node has written a value.

    formatted_output is NOT re-declared here - it is inherited from AgentState
    (re-declaring it with a bare type breaks the state contract).
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / ReturnsComplaintSearchGraphNode.merge_output
    # ------------------------------------------------------------------

    # Identifier-stripped, validated query payload produced by PreProcessNode.
    # Raw input is NOT persisted beyond PreProcessNode.
    validated_input: Optional[str]

    # Final returns/complaint search answer, mapped from the inner graph's
    # formatted_answer output via merge_output.
    returns_kb_answer: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text search query (whitespace-collapsed, length-capped).
    search_query: Optional[str]

    # JSON STRING (to_json) of parsed structured query params. Deserialised
    # dict shape: {"category": str | None, "top_k": int | None}.
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: Optional[str]

    # Manifest `retrieval` block forwarded by ReturnsComplaintSearchGraphNode.
    # _parent_config() -> DomainWorkflowGraph._extra_initial_state().
    # JSON STRING (to_json) of {"top_k": int, "score_threshold": float,
    # "kb_path": str}. Consumers (RetrieveNode, RerankFilterNode) read it
    # back via from_json().
    retrieval_config: Optional[str]

    # RetrieveNode output
    # JSON STRING (to_json) of scored KB candidates. Deserialised shape:
    # list[dict], each entry {"id": str, "title": str, "category": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: Optional[str]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: Optional[str]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers.
    grounded_answer: Optional[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "title": str, "source": str}.
    # Consumers (OutputFormatNode) read it back via from_json().
    citations: Optional[str]

    # OutputFormatNode output
    # Final formatted answer (body + sources + advisory disclaimer). Written by
    # OutputFormatNode; surfaced to the outer graph via get_output() ->
    # merge_output().
    formatted_answer: Optional[str]

    # Validation / parse notes accumulated during intake (no PII).
    # JSON STRING (to_json) of list[str].
    intake_notes: Optional[str]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    error_code: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
