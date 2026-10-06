"""AgentCore Platform v1.0"""

# RET-C2-006 - PreProcessNode
# Outer backbone pre_process slot: the ingest boundary and the owner of the
# caller-data contract.
#
# Responsibilities:
#   - Enforce VERIFIED_EXTERNAL trust (required_trust_level - matches the
#     manifest's declared required_trust_level in config/agent.yaml)
#   - Reject empty and over-long questions early (fail fast)
#   - Refuse instruction-override payloads before anything downstream runs
#   - Strip direct customer identifiers (card/account/order numbers, telephone
#     numbers, e-mail addresses) from the question AND from every free-text
#     field the caller can put on the context channel
#   - Validate every declared input_context field against explicit bounds and
#     fail CLOSED on a violation, naming the field and never the value
#   - Write validated_input + enriched_context to State
#
# The refusals here are the template's OWN guarantees. A platform input gate
# may reject some of the same payloads first, but this node does not depend on
# that: where such a gate is absent or configured off, an unchecked payload
# would otherwise reach the answer path and come back as a success. Every
# refusal is therefore observable as behaviour - error status, nothing carried
# forward - and holds when execute() is called with no framework wrapper in
# front of it.
#
# Node contract: execute(self, state) -> dict — no extra parameters. This node
# has no tunable config knobs. Returns ONLY the state keys it writes.

import logging
from typing import Any, ClassVar, Dict, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, TOO_LONG
from src.schemas.caller_contract import (
    MAX_QUERY_CHARS,
    is_instruction_override,
    normalise_query,
    strip_direct_identifiers,
    validate_identifier_field,
    validate_top_k,
)

logger = logging.getLogger(__name__)

# Free-text context fields are sanitised rather than rejected: a caller may
# legitimately pass a short note, but it must not smuggle a customer identifier
# into State through a channel the question text does not travel on.
_SANITISED_CONTEXT_FIELDS = ("note",)


def _validate_input_context(input_context: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Validate the caller's input_context against the declared contract.

    Returns (normalised_context, error_message). Error messages name the field
    only - never the rejected value. Accepted fields:

        channel:  str matching ^[a-z0-9_]{1,32}$  (absent -> "unknown")
        category: str matching ^[a-z0-9_]{1,32}$  (absent -> no category filter)
        top_k:    int 1..20                       (absent -> configured default)
        note:     free text, identifier-stripped, length-capped

    Any other key is ignored. A non-mapping input_context is rejected.
    """
    if input_context is None:
        return {"channel": "unknown"}, None
    if not isinstance(input_context, dict):
        return {}, "input_context must be an object"

    normalised: Dict[str, Any] = {}

    channel, channel_error = validate_identifier_field("input_context.channel", input_context.get("channel"))
    if channel_error:
        return {}, channel_error
    normalised["channel"] = channel or "unknown"

    category, category_error = validate_identifier_field("input_context.category", input_context.get("category"))
    if category_error:
        return {}, category_error
    if category is not None:
        normalised["category"] = category

    top_k, top_k_error = validate_top_k(input_context.get("top_k"))
    if top_k_error:
        return {}, f"input_context.{top_k_error}"
    if top_k is not None:
        normalised["top_k"] = top_k

    # Free-text context fields get the same identifier strip as the question.
    # The platform's own input masking covers the question channel only, so a
    # sanitiser that ran only on user_input would leave this route open.
    for field in _SANITISED_CONTEXT_FIELDS:
        value = input_context.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            return {}, f"input_context.{field} must be a string"
        if len(value) > MAX_QUERY_CHARS:
            return {}, f"input_context.{field} exceeds {MAX_QUERY_CHARS} characters"
        cleaned, _note = normalise_query(strip_direct_identifiers(value))
        normalised[field] = cleaned

    return normalised, None


class PreProcessNode(FunctionNode):
    """Ingest validation for RET-C2-006.

    The outer backbone's pre_process slot - the only node declaring
    VERIFIED_EXTERNAL trust, so unauthenticated or anonymous callers are
    rejected here (fail fast; inner domain nodes run behind this boundary and
    are declared ANONYMOUS).

    Input state keys:
        user_input:    str  - the caller's returns/complaint question
        input_context: dict - optional caller parameters (channel, category,
                              top_k, note)

    Output state keys (partial dict):
        validated_input:  str        - normalised, identifier-stripped question
        enriched_context: dict       - request metadata for downstream nodes
        status:           str        - AgentStatus.SUCCESS.value or ERROR
        error_log:        list[str]  - set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context")

        # -- Emptiness --------------------------------------------------------
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            logger.warning("PreProcessNode: user_input is empty or missing")
            emit_trace_event("pre_process_validation_failed", {"reason": "empty_input"}, state)
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        # -- Length bound -----------------------------------------------------
        if len(user_input) > MAX_QUERY_CHARS:
            logger.warning("PreProcessNode: question exceeds %d characters", MAX_QUERY_CHARS)
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "query_too_long", "length": len(user_input)},
                state,
            )
            emit_progress(TOO_LONG)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "QUESTION_TOO_LONG",
                "error_log": [f"PreProcessNode: question exceeds {MAX_QUERY_CHARS} characters"],
            }

        # -- Instruction-override refusal -------------------------------------
        # Refused here, before retrieval or answer assembly, and independently
        # of any platform gate. The rejected text is never echoed back.
        if is_instruction_override(user_input):
            logger.warning("PreProcessNode: question refused - instruction-override content")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "instruction_override"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: question refused - instruction-override content"],
            }

        # -- Caller-data contract (input_context) ------------------------------
        context, context_error = _validate_input_context(input_context)
        if context_error:
            # Name the field, never the value.
            logger.warning("PreProcessNode: input_context validation failed")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "invalid_input_context"},
                state,
            )
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": [f"PreProcessNode: {context_error}"],
            }

        # -- Normalise + strip direct identifiers -------------------------------
        validated_input, truncation_note = normalise_query(strip_direct_identifiers(user_input))
        if truncation_note:
            logger.info("PreProcessNode: %s", truncation_note)

        # Audit: a question was accepted and stripped of direct identifiers.
        logger.info("PreProcessNode: validated question chars=%d", len(validated_input))
        emit_trace_event("pre_process_complete", {"input_chars": len(validated_input)}, state)

        return {
            "validated_input": validated_input,
            "enriched_context": {
                "source": "CustomerReturnsComplaintQAAgent",
                "channel": context.get("channel", "unknown"),
            },
            "status": AgentStatus.SUCCESS.value,
        }
