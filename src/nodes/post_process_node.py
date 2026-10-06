"""AgentCore Platform v1.0"""

# RET-C2-006 - PostProcessNode
# Outer backbone post_process slot: the external-output boundary for the
# returns/complaint answer.
#
# THE OUTPUT INVARIANT this gate enforces (the same statement appears in
# docs/02_design.md, and the answer is RENDERED to satisfy it - this node
# ENFORCES it):
#
#   The external answer is assembled from knowledge-base passages and their
#   citations only. It carries
#     (a) no credential or secret pattern,
#     (b) no direct customer identifier - card/account/order number, telephone
#         number, e-mail address,
#     (c) no verbatim embedding of caller-supplied text.
#
#   This template renders no computed monetary figures: the answer is policy
#   prose plus citations, and the bundled knowledge base contains no monetary
#   amounts. A monetary rounding grid therefore does not apply here, and none
#   is imposed - a numeric transform with nothing to transform can only mangle
#   the structural numbers the policy text depends on ("within 30 days",
#   "5 to 7 business days", "kb-001", "[1]").
#
# LAYER ORDER. The pattern SCANS run before any layer that rewrites the text,
# and the result is scanned AGAIN afterwards. Order matters for a scan-then-
# transform pipeline: a transform that rewrites bytes can destroy the shape a
# later pattern would have matched, and a pattern that only ever ran first
# would then miss it. The final re-scan makes the invariant independent of what
# the middle layers did - if anything is still there, the output is withheld
# entirely rather than shipped.
#
#   1. credential scan          -> any hit withholds the whole answer (ERROR)
#   2. direct-identifier scan   -> hits are redacted, each one audited
#   3. verbatim caller-text     -> embeddings of caller-derived state are
#      redaction                   replaced with the redaction marker
#   4. RE-SCAN of the sanitised bytes -> a residual credential or identifier
#      withholds the whole answer (fail closed)
#
# The domain output gate is the module-level function `_security_gate_output`
# called from inside execute() - NOT an instance method on the node class (the
# framework auto-wraps node instance methods on the real invoke path). The
# agent class exposes the same scanner as the canonical output-gate entry point
# and delegates here, so there is one source of truth.
#
# Node contract: execute(self, state) -> dict — no extra parameters.
# Returns ONLY the state keys this node writes.

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Pattern, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG
from src.schemas.caller_contract import (
    contains_direct_identifier,
    strip_direct_identifiers,
)

logger = logging.getLogger(__name__)

# Credential patterns that must never appear in the answer.
# Each tuple: (name, compiled regex) - order matters (most specific first).
_DISALLOWED_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    # API keys: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # JWT: three base64url segments separated by dots.
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # Bearer token in an authorization-like context.
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]{20,}", re.IGNORECASE)),
    # Credential assignments.
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

# State fields that must never be embedded verbatim in the external response.
# The answer is assembled from retrieved passages and their citations only, so
# caller-derived text reappearing verbatim means caller-controlled content
# reached the external surface.
_BLOCKED_FIELDS = frozenset({"user_input", "validated_input", "search_query"})

# Only substantial values are matched, so a short incidental overlap between a
# question and a policy passage is not mistaken for an embedding.
_MIN_BLOCKED_VALUE_CHARS = 10

_WITHHELD_TEMPLATE = (
    "[ANSWER WITHHELD: the generated answer contained disallowed content ({reason}). "
    "Review the knowledge-base content and retry.]"
)


def _security_gate_output(content: str) -> Optional[str]:
    """Scan output for disallowed credential/secret patterns.

    Returns the first violation name, or None when the output is clean.
    Module-level function (not a node instance method) - the framework
    auto-wraps node instance methods on the real invoke path, so the gate must
    live at module level.
    """
    for name, pattern in _DISALLOWED_PATTERNS:
        if pattern.search(content):
            return name
    return None


def _redact_blocked_fields(result: str, state: AgentState) -> Tuple[str, List[str]]:
    """Replace verbatim embeddings of caller-derived state text with the marker.

    Returns (sanitised_result, redacted_field_names).
    """
    redacted: List[str] = []
    sanitised = result
    for field in sorted(_BLOCKED_FIELDS):
        value = state.get(field)
        if isinstance(value, str) and len(value) > _MIN_BLOCKED_VALUE_CHARS and value in sanitised:
            sanitised = sanitised.replace(value, "[REDACTED]")
            redacted.append(field)
    return sanitised, redacted


# Reason code -> the sentence the caller reads. A code with no entry falls
# back to the generic one rather than leaking the code itself.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Apply the output gate and expose the final returns/complaint answer.

    Outer backbone post_process slot.

    Input state keys:
        result:            str - rendered answer from the inner OutputFormatNode
        returns_kb_answer: str - same rendered answer (fallback source)

    Output state keys (partial dict):
        formatted_output: str
        result:           str        (written on the redaction / withheld paths)
        status:           str        - AgentStatus.SUCCESS.value or ERROR
        error_log:        list[str]  - set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _withhold(self, state: AgentState, reason: str, event: str) -> Dict[str, Any]:
        """Withhold the whole answer: nothing partial reaches the caller."""
        logger.error("PostProcessNode: output withheld - %s", reason)
        emit_trace_event(event, {"violation": reason}, state)
        withheld = _WITHHELD_TEMPLATE.format(reason=reason)
        return {
            "formatted_output": withheld,
            "result": withheld,
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PostProcessNode: output withheld - {reason}"],
        }

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # A run declined upstream has nothing to format. Render the reason as
        # the caller-facing body and carry the marker onward.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
                "formatted_output": message,
                "result": message,
            }
        answer = str(state.get("result") or state.get("returns_kb_answer") or "")

        if not answer.strip():
            # No answer was generated - forward as-is (non-fatal); upstream
            # failures are already recorded in error_log.
            return {
                "formatted_output": answer,
                "status": AgentStatus.SUCCESS.value,
            }

        # -- Layer 1: credential scan (withhold entirely) ----------------------
        violation = _security_gate_output(answer)
        if violation:
            return self._withhold(state, violation, "post_process_credential_violation")

        # -- Layer 2: direct-identifier scan + redaction -----------------------
        sanitised = strip_direct_identifiers(answer)
        if sanitised != answer:
            logger.error("PostProcessNode: direct customer identifier removed from the answer")
            emit_trace_event("post_process_identifier_redaction", {"redacted": True}, state)

        # -- Layer 3: verbatim caller-text redaction ---------------------------
        sanitised, redacted_fields = _redact_blocked_fields(sanitised, state)
        if redacted_fields:
            logger.error(
                "PostProcessNode: caller-derived text embedded verbatim in output - %s",
                ", ".join(redacted_fields),
            )
            emit_trace_event(
                "post_process_blocked_field_redaction",
                {"fields": redacted_fields},
                state,
            )

        # -- Layer 4: re-scan the bytes that are about to be surfaced ----------
        # Independent of what layers 2 and 3 rewrote. Anything still here means
        # a transform exposed or reshaped a pattern, so nothing is shipped.
        residual = _security_gate_output(sanitised)
        if residual:
            return self._withhold(state, residual, "post_process_credential_violation")
        if contains_direct_identifier(sanitised):
            return self._withhold(state, "direct_identifier", "post_process_identifier_violation")

        logger.info("PostProcessNode: output gate passed - length=%d", len(sanitised))
        emit_trace_event("post_process_complete", {"output_chars": len(sanitised)}, state)

        return {
            "formatted_output": sanitised,
            "result": sanitised,
            "status": AgentStatus.SUCCESS.value,
        }
