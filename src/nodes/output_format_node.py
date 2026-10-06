"""AgentCore Platform v1.0"""

# RET-C2-006 - OutputFormatNode
# Domain node 5 (terminal): compose the final formatted answer - the grounded
# answer body, the Sources list, the scope note, and the standing advisory
# disclaimer. These are part of THIS node's domain output contract, not of the
# outer post_process slot (post_process only gates, it does not compose).
#
# The scope note states the contract the answer is RENDERED to satisfy;
# src/nodes/post_process_node.py independently ENFORCES the same statement at
# the external boundary, so a rendering regression cannot quietly widen what
# the answer is allowed to contain.
#
# Node contract: execute(self, state) -> dict — no extra parameters, no config
# knobs.
#
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + status to the outer merge_output().
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json

# Scope note - states the answer's content contract to the reader, matching
# the invariant the output boundary enforces.
_SCOPE_NOTE = (
    "Scope: this answer is assembled only from the knowledge-base passages "
    "cited above. It carries no customer identifiers (order, card, telephone "
    "or e-mail details) and no text quoted back from the request."
)

# Standing advisory line - appended to EVERY answer this template emits.
_ADVISORY_DISCLAIMER = (
    "This answer is generated from the seeded returns & complaints knowledge "
    "base for informational purposes only. Store-specific promotions, "
    "regional regulations, and account-specific exceptions may modify this "
    "policy - confirm with a supervisor or the official policy portal "
    "before finalizing a customer-facing decision."
)


class OutputFormatNode(FunctionNode):
    """Compose the final answer: body + sources + advisory disclaimer.

    Input state keys:
        grounded_answer: answer body with [n] citation markers
        citations:       JSON list [{ref, id, title, source}]

    Output state keys (partial dict):
        formatted_answer: final rendered answer string
        status:           AgentStatus.SUCCESS.value (plain string — never
                          write the bare enum to State)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        grounded_answer = state.get("grounded_answer") or ("No answer is available for this request.")
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []

        lines: List[str] = []
        lines.append("# Customer Returns & Complaint Resolution Search Result")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source = str(citation.get("source", "")).strip()
                suffix = f" ({source})" if source else ""
                lines.append(f"- [{ref}] {title}{suffix}")
        else:
            lines.append("- none (no knowledge-base passage cleared the relevance threshold)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{_SCOPE_NOTE}*")
        lines.append("")
        lines.append(f"*{_ADVISORY_DISCLAIMER}*")

        formatted_answer = "\n".join(lines)

        # Audit: final answer composed (scope note + disclaimer attached).
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
