# RET-C2-006 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (CustomerReturnsComplaintQAAgent / Graph)
# end-to-end via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# Mirrors docs/03_test_spec.md §3.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

import src.graph.graph
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    CustomerReturnsComplaintQAAgent,
    Graph,
    ReturnsComplaintSearchGraphNode,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

# Byte-equal to the deploy payload's "input" field.
_RETURNS_QUERY = "What is the return window for a defective item, and how do I process " "a refund for the customer?"


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_inherits_agent_base_graph_directly(self):
        assert issubclass(CustomerReturnsComplaintQAAgent, AgentBaseGraph)

    def test_graph_alias(self):
        assert Graph is CustomerReturnsComplaintQAAgent

    def test_state_schema_is_state(self):
        assert CustomerReturnsComplaintQAAgent().state_schema is State

    def test_compile_fills_all_backbone_slots(self):
        agent = CustomerReturnsComplaintQAAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], ReturnsComplaintSearchGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in CustomerReturnsComplaintQAAgent.__dict__


class TestMainSlotGraphNode:
    def test_get_subgraph_returns_the_inner_graph(self):
        subgraph = ReturnsComplaintSearchGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_extract_input_prefers_validated_input(self):
        node = ReturnsComplaintSearchGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_merge_output_maps_the_inner_contract(self):
        node = ReturnsComplaintSearchGraphNode()
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "t", "source": "s"}])
        delta = node.merge_output(
            {},
            {"formatted_answer": "ANSWER", "citations": citations, "status": AgentStatus.SUCCESS.value},
        )
        # The inner formatted_answer surfaces as BOTH returns_kb_answer and
        # result (PostProcessNode's output gate reads state["result"]).
        assert delta == {
            "returns_kb_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "status": AgentStatus.SUCCESS.value,
            # The boundary also carries the reason marker; a run that produced
            # an answer carries no reason, so it is blank here.
            "error_code": "",
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert ReturnsComplaintSearchGraphNode.error_strategy == "propagate"
        assert ReturnsComplaintSearchGraphNode.propagate_hitl is False

    def test_parent_config_falls_back_when_the_runtime_file_is_unreadable(self, monkeypatch):
        # An unreadable runtime file must not crash graph construction: nothing
        # is forwarded and each node keeps its own module default.
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", pathlib.Path("/nonexistent/config.yaml"))
        cfg = ReturnsComplaintSearchGraphNode()._parent_config()
        assert cfg["configurable"] == {"retrieval": {}, "llm": {}}

    def test_parent_config_drops_non_finite_declared_numbers(self, monkeypatch):
        # A NaN relevance floor would compare False against every candidate and
        # silently empty the answer, so a non-finite declared value is dropped
        # rather than forwarded.
        monkeypatch.setattr(
            src.graph.graph,
            "_runtime_config",
            lambda: {
                "retrieval": {"top_k": float("nan"), "score_threshold": float("inf"), "kb_path": 5},
                "llm": {"temperature": "hot", "max_tokens": True},
            },
        )
        cfg = ReturnsComplaintSearchGraphNode()._parent_config()
        assert cfg["configurable"] == {"retrieval": {}, "llm": {}}

    def test_parent_config_bridges_input_context_on_extract_input(self):
        # extract_input is the last hook that sees the outer state before the
        # inner invoke, so it stashes input_context for the inner graph.
        from src.graph.context_bridge import get_caller_input_context

        node = ReturnsComplaintSearchGraphNode()
        node.extract_input({"validated_input": "q", "input_context": {"category": "final_sale"}})
        assert get_caller_input_context() == {"category": "final_sale"}


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_invoke_returns_success(self):
        result = _run(_RETURNS_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_output_is_the_gated_formatted_answer(self):
        output = _run(_RETURNS_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Customer Returns & Complaint Resolution Search Result")
        assert "[1]" in output
        assert "confirm with a supervisor or the official policy portal" in output

    def test_e2e_traverses_the_post_process_gate(self):
        history = _run(_RETURNS_QUERY).get("node_history", [])
        for cls_name in ("PreProcessNode", "ReturnsComplaintSearchGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_no_coverage_query_still_terminates_success(self):
        result = _run("quantum telepathy sandwich recipes")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """Trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        main slot (it sees status=error and skips the inner graph) and routes
        past post_process to finalize — no domain answer is ever produced."""
        result = _run(_RETURNS_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """State helpers: producers to_json() on write, consumers from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-001", "score": 0.69, "title": "return eligibility"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "return_eligibility", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
