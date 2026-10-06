"""AgentCore Platform v1.0"""

# RET-C2-006 - Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Customer Returns & Complaint Resolution Q&A Agent (Cat 2 RAG domain workflow).
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed - do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (RETRY, max 3)
#                                             -> pre_process
#
#   `main` slot is a GraphNode subclass (ReturnsComplaintSearchGraphNode) that
#   delegates the full returns/complaint search domain workflow to
#   DomainWorkflowGraph (inner BaseGraph: input_validate -> retrieve ->
#   rerank_filter -> generate_answer -> output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- input_context hand-off (outer -> inner)
#
# Class-name contract:
#   graph.py class:           CustomerReturnsComplaintQAAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.CustomerReturnsComplaintQAAgent"
#   src/api/server.py import: from src.graph.graph import CustomerReturnsComplaintQAAgent
#
# Rules enforced:
#   - CustomerReturnsComplaintQAAgent inherits AgentBaseGraph (framework base
#     class - direct inheritance)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - ReturnsComplaintSearchGraphNode assigned to self._nodes["main"]
#   - _parent_config() forwards only validated runtime settings (never junk)
#   - merge_output() returns only changed keys
#   - add_edges() NOT overridden on the outer graph
#   - No platform-SDK imports (framework/ and shared/ only)

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, cast

from framework.schemas.agent_status import AgentStatus
from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode, _security_gate_output
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, finite_in_range

if TYPE_CHECKING:
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Runtime-parameter file: src/graph/graph.py -> parents[2] is the repo root.
# config/agent.yaml (the static manifest) holds only registration identity;
# every runtime parameter lives in config/config.yaml.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _runtime_config() -> Dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    This is the same file the platform registry loads and passes as
    Graph(config=...); the standalone server (src/api/server.py) reads it here
    so the deployed agent and a registry-loaded agent see identical
    configuration. Returns an empty dict - never raises - when the file is
    absent, unreadable, not valid YAML, or not a mapping (the graph then runs
    on its built-in defaults).
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return cast(Dict[str, Any], loaded)


# Sentinel that no declared value can equal, so "invalid" is distinguishable
# from "a legitimate value that happens to match the fallback".
_UNSET = float("-1e308")


def _config_number(value: Any, lo: float, hi: float) -> Optional[float]:
    """Validate a declared numeric setting: a real number, finite, within [lo, hi].

    Bools, strings, non-numerics, NaN/Infinity, and OUT-OF-RANGE values return
    None, and the consumer keeps its built-in default - an out-of-range declared
    value is a configuration mistake, not something to clamp silently. The
    shared parser does the type/finiteness work (one implementation, used by the
    nodes too); the range is checked here rather than clamped.
    """
    parsed = finite_in_range(value, _UNSET, -_UNSET, _UNSET)
    if parsed == _UNSET or not lo <= parsed <= hi:
        return None
    return parsed


class ReturnsComplaintSearchGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph RAG pipeline).
    Called by the backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()  - instantiate DomainWorkflowGraph with the validated
                        runtime settings (_parent_config())
      extract_input() - pull validated_input from outer state; bridge input_context
      merge_output()  - map sub_result fields into outer state delta (changed keys only)
      error_strategy  - "propagate": re-raise inner errors as SubgraphError (fail fast)
    """

    # "propagate": re-raise inner graph exceptions as SubgraphError (default - fail fast).
    # "handle": call on_subgraph_error() instead - use for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: HITL interrupts are handled inside the inner graph only.
    # True: surface inner HITL interrupt to the outer caller.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the validated `retrieval` + `llm` settings to the inner graph.

        Reads config/config.yaml (see _runtime_config) and returns the two
        tuning blocks under config["configurable"]. The inner graph republishes
        the `retrieval` block into inner state
        (DomainWorkflowGraph._extra_initial_state()) so RetrieveNode /
        RerankFilterNode read live top_k / score_threshold / kb_path values -
        the node contract is ``execute(self, state) -> dict``, state only, so
        tuning knobs travel through State. The `llm` block is forwarded
        verbatim for the documented answer-synthesis upgrade (unused by the
        current deterministic nodes).

        Every forwarded numeric is validated here (type, finiteness, range) so
        a malformed configuration file can neither crash graph construction nor
        weaken the retrieval relevance floor. Invalid or absent keys are simply
        not forwarded; each node then falls back to its module default.
        """
        cfg = _runtime_config()
        retrieval_raw = cfg.get("retrieval")
        retrieval: Dict[str, Any] = retrieval_raw if isinstance(retrieval_raw, dict) else {}
        llm_raw = cfg.get("llm")
        llm: Dict[str, Any] = llm_raw if isinstance(llm_raw, dict) else {}

        declared_retrieval: Dict[str, Any] = {}

        top_k = _config_number(retrieval.get("top_k"), 1, 20)
        if top_k is not None and top_k == int(top_k):
            declared_retrieval["top_k"] = int(top_k)

        score_threshold = _config_number(retrieval.get("score_threshold"), 0.0, 1.0)
        if score_threshold is not None:
            declared_retrieval["score_threshold"] = score_threshold

        kb_path = retrieval.get("kb_path")
        if isinstance(kb_path, str) and kb_path:
            declared_retrieval["kb_path"] = kb_path

        declared_llm: Dict[str, Any] = {}

        temperature = _config_number(llm.get("temperature"), 0.0, 2.0)
        if temperature is not None:
            declared_llm["temperature"] = temperature

        max_tokens = _config_number(llm.get("max_tokens"), 1, 100_000)
        if max_tokens is not None and max_tokens == int(max_tokens):
            declared_llm["max_tokens"] = int(max_tokens)

        return {"configurable": {"retrieval": declared_retrieval, "llm": declared_llm}}

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time.

        The inner graph receives the validated runtime settings via its
        BaseGraph ctor; its domain NODES still take no constructor arguments
        and read tuning knobs per-call via State.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to act on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates the raw user_input, strips direct customer
        identifiers, and writes the result to validated_input. Prefer that;
        fall back to user_input when validated_input is absent (e.g. a bare
        unit test).

        Also bridges the caller's input_context to the inner graph:
        GraphNode.execute() does not forward input_context on
        subgraph.invoke(), and extract_input is the last hook in this repo's
        code that sees the outer state before the inner invoke - see
        src/graph/context_bridge.py.
        """
        set_caller_input_context(cast(Optional[Dict[str, Any]], state.get("input_context")))
        return cast(str, state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys - never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_answer", "citations", "status", ...
          This merge_output() reads -> sub_result.get("formatted_answer"),
                                       sub_result.get("citations"),
                                       sub_result.get("status")

        returns_kb_answer (str | None): final rendered returns/complaint answer;
          written by OutputFormatNode inside the inner graph.
        result: PostProcessNode (outer post_process slot) reads
          state.get("result") - the inner graph emits the rendered answer under
          "formatted_answer", so map it to "result" as well; otherwise the
          final output surfaced by PostProcessNode (and its output gate) is
          always empty.
        status (str | None): terminal AgentStatus value from the inner graph run.
        """
        return {
            # Outer reason wins: a reason settled before the inner run is the real
            # one, and a plain sub_result.get() would erase it.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
            "returns_kb_answer": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "status": sub_result.get("status"),
        }


class CustomerReturnsComplaintQAAgent(AgentBaseGraph):
    """Outer graph for RET-C2-006 (Cat 2 RAG).

    Inherits AgentBaseGraph directly (framework base class). Domain logic is
    fully encapsulated in ReturnsComplaintSearchGraphNode (main slot), which
    delegates to DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode (ingest validation + caller-data contract)
      - main:         ReturnsComplaintSearchGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate)

    add_edges() is NOT overridden - backbone wiring belongs to the framework.

    Runtime configuration: the platform registry loads config/config.yaml and
    passes it as Graph(config=...); the standalone server does the same via
    _runtime_config(). AgentBaseGraph itself consumes max_retry from that
    config (retry routing), so the declared value is live in both deployments.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the platform registry."""
        return "CustomerReturnsComplaintQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first - it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = ReturnsComplaintSearchGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.

    def _security_gate_output(self, content: str) -> Optional[str]:
        """Canonical output-gate entry point on the agent class.

        Delegates to the credential scanner that PostProcessNode (the
        post_process backbone slot) applies at runtime, so there is a single
        source of truth for the pattern set. Returns the first violation name,
        or None when the output is clean.
        """
        return _security_gate_output(content)


# Back-compat alias - config/agent.yaml declares the
# CustomerReturnsComplaintQAAgent class path, and src/api/server.py imports the
# class directly. Keep both names pointing at the agent.
Graph = CustomerReturnsComplaintQAAgent
