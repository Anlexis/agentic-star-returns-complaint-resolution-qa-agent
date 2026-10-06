# RET-C2-006 — Unit Tests: OutputFormatNode (inner domain node 5, terminal)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# formatted_answer is a DOMAIN field (not an input-mask scan target); the advisory
# disclaimer is part of THIS node's output contract.
#
# Mirrors docs/03_test_spec.md §2.6 (FMT-01..FMT-05).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import OutputFormatNode
from src.schemas.state import to_json

_DISCLAIMER_FRAGMENT = "confirm with a supervisor or the official policy portal"


def _make_state(grounded_answer, citations, **extra) -> dict:
    state = {
        "grounded_answer": grounded_answer,
        "citations": citations,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestFormattedAnswer:
    def test_fmt_01_composes_header_body_sources_disclaimer(self):
        citations = to_json(
            [
                {
                    "ref": 1,
                    "id": "kb-001",
                    "title": "Standard return eligibility window",
                    "source": "Returns & Exchanges Policy",
                }
            ]
        )
        result = OutputFormatNode()(_make_state("[1] the grounded answer body.", citations))
        answer = result["formatted_answer"]
        assert answer.startswith("# Customer Returns & Complaint Resolution Search Result")
        assert "[1] the grounded answer body." in answer
        assert "## Sources" in answer
        assert "- [1] Standard return eligibility window (Returns & Exchanges Policy)" in answer
        assert _DISCLAIMER_FRAGMENT in answer
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        # An identity check on the TYPE is the point here: AgentStatus values
        # would satisfy isinstance() against str, which is exactly the bug
        # this guards against.
        assert type(result["status"]) is str  # noqa: E721

    def test_fmt_02_source_suffix_omitted_when_blank(self):
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "Standard return eligibility window", "source": ""}])
        answer = OutputFormatNode()(_make_state("body.", citations))["formatted_answer"]
        assert "- [1] Standard return eligibility window\n" in answer + "\n"
        assert "()" not in answer

    def test_fmt_03_disclaimer_present_on_every_answer(self):
        # The disclaimer must ride WITH the substance, never separately.
        for grounded in ("a body.", ""):
            answer = OutputFormatNode()(_make_state(grounded, to_json([])))["formatted_answer"]
            assert _DISCLAIMER_FRAGMENT in answer


class TestDegradedInputs:
    def test_fmt_04_no_citations_renders_explicit_none_line(self):
        answer = OutputFormatNode()(_make_state("no coverage body.", to_json([])))["formatted_answer"]
        assert "- none (no knowledge-base passage cleared the relevance threshold)" in answer

    def test_fmt_05_missing_grounded_answer_uses_fallback_text(self):
        state = _make_state("", to_json([]))
        del state["grounded_answer"]
        result = OutputFormatNode()(state)
        assert "No answer is available for this request." in result["formatted_answer"]
        assert result["status"] == AgentStatus.SUCCESS.value
