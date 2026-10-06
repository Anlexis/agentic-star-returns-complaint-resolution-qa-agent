"""AgentCore Platform v1.0"""

# RET-C2-006 - InputValidateNode
# Domain node 1: resolve the caller's structured parameters and normalise the
# question before retrieval runs.
#
# The caller can reach this node on two channels and BOTH are validated with
# the same rules (src/schemas/caller_contract.py), so neither is a way round
# the other:
#
#   1. input_context - the structured invocation parameters, seeded into inner
#      state by DomainWorkflowGraph._extra_initial_state() (see
#      src/graph/context_bridge.py);
#   2. a JSON envelope inside the question string - {"query": "...",
#      "category": "...", "top_k": N} - the shape this template has always
#      accepted for callers that can only send text.
#
# input_context wins where both carry the same key; an invalid value on EITHER
# channel rejects the whole request. Re-validating here (rather than trusting
# the ingest boundary) means a direct inner-graph invocation gets the identical
# fail-closed contract.
#
# The instruction-override refusal is repeated here for the same reason: it is
# this template's own guarantee, not a platform gate's, and it must hold for a
# caller that reaches the inner graph directly.
#
# Node contract: execute(self, state) -> dict — no extra parameters. Tuning
# knobs (top_k / score_threshold) belong to RetrieveNode / RerankFilterNode and
# arrive through State. Returns only changed state keys (partial dict).

import json
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import INPUT_REJECTED
from src.services.progress import emit_progress
from src.schemas.caller_contract import (
    is_instruction_override,
    normalise_query,
    strip_direct_identifiers,
    validate_identifier_field,
    validate_top_k,
)
from src.schemas.state import to_json


class InputValidateNode(FunctionNode):
    """Resolve validated caller parameters and normalise the question.

    Input state keys:
        validated_input | user_input: the question text, or a JSON envelope
        input_context:                caller parameters (category, top_k),
                                      seeded by the inner graph's
                                      _extra_initial_state() context bridge

    Output state keys (partial dict):
        search_query:     normalised free-text search query
        query_filters:    JSON dict {"category": str|None, "top_k": int|None}
        intake_notes:     (when anomalies were seen) JSON list[str]
        status/error_log: only on a failed-closed validation
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def _reject(self, state: AgentState, reason: str, message: str, code: str = "INVALID_REQUEST") -> Dict[str, Any]:
        """Fail closed: error status, no query published, field named not valued."""
        emit_trace_event("input_validate_failed", {"reason": reason}, state)
        if code:
            # A value the caller can correct: the run COMPLETES carrying the
            # reason so the request can be sent again on the same conversation.
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": code,
                "error_log": [f"InputValidateNode: {message}"],
            }
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"InputValidateNode: {message}"],
        }

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")
        notes: List[str] = []

        # ------------------------------------------------------------------
        # Channel 1: the question string, optionally a JSON envelope.
        # ------------------------------------------------------------------
        query = ""
        envelope_category: Any = None
        envelope_top_k: Any = None

        if isinstance(raw, str) and raw.strip():
            text = raw.strip()
            payload: Any = None
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append("InputValidateNode: JSON-looking input did not parse - treated as plain text query.")
            if isinstance(payload, dict):
                query = str(payload.get("query") or payload.get("question") or "")
                envelope_category = payload.get("category")
                envelope_top_k = payload.get("top_k")
            else:
                query = text
        else:
            notes.append("InputValidateNode: empty request - no query to search.")

        if query and is_instruction_override(query):
            return self._reject(
                state,
                "instruction_override",
                "query refused - instruction-override content.",
                # Not something the caller corrects by rewording: the refusal
                # terminates the run rather than inviting another attempt.
                code="",
            )

        # ------------------------------------------------------------------
        # Channel 2: structured invocation parameters (input_context).
        # ------------------------------------------------------------------
        input_context = state.get("input_context")
        if input_context is not None and not isinstance(input_context, dict):
            return self._reject(state, "invalid_input_context", "input_context must be an object.")
        context: Dict[str, Any] = input_context if isinstance(input_context, dict) else {}

        # ------------------------------------------------------------------
        # Both channels, one rule set. input_context wins on overlap; an
        # invalid value on EITHER channel rejects the request.
        # ------------------------------------------------------------------
        category: Optional[str] = None
        for source in (envelope_category, context.get("category")):
            value, error = validate_identifier_field("category", source)
            if error:
                return self._reject(state, "invalid_category", error)
            if value is not None:
                category = value

        top_k: Optional[int] = None
        for source in (envelope_top_k, context.get("top_k")):
            parsed, error = validate_top_k(source)
            if error:
                return self._reject(state, "invalid_top_k", error)
            if parsed is not None:
                top_k = parsed

        # ------------------------------------------------------------------
        # Normalise the question. The identifier strip is repeated here so a
        # direct inner-graph call, or an envelope whose `query` field was never
        # seen by the ingest boundary, is covered too.
        # ------------------------------------------------------------------
        search_query, truncation_note = normalise_query(strip_direct_identifiers(query))
        if truncation_note:
            notes.append(f"InputValidateNode: {truncation_note}.")

        filters = {"category": category, "top_k": top_k}

        # Audit: request parsed and normalised.
        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(search_query),
                "has_category_filter": category is not None,
                "has_top_k_override": top_k is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": search_query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
