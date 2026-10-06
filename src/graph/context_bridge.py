"""AgentCore Platform v1.0"""

# src/graph/context_bridge.py — carries the caller's input_context across the
# outer→inner graph boundary.
#
# Why this exists: GraphNode.execute() (framework) invokes the inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)` WITHOUT forwarding the
# outer state's input_context, so inner-node reads of state["input_context"]
# (the per-invocation category filter and retrieval-depth override) would always
# see {} through the full nested graph. The sanctioned subclass hooks bridge it:
#
#   ReturnsComplaintSearchGraphNode.extract_input(state)  [BEFORE subgraph.invoke]
#       → set_caller_input_context(state["input_context"])
#   DomainWorkflowGraph._extra_initial_state()            [INSIDE subgraph.invoke]
#       → returns {"input_context": get_caller_input_context()}
#
# A ContextVar keeps the hand-off correct per thread/task, so concurrent
# invocations in one process cannot see each other's context.

from contextvars import ContextVar
from typing import Any, Dict, Optional

_CALLER_INPUT_CONTEXT: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "ret_c2_006_caller_input_context", default=None
)


def set_caller_input_context(input_context: Optional[Dict[str, Any]]) -> None:
    """Stash the outer graph's input_context for the imminent inner-graph invoke."""
    _CALLER_INPUT_CONTEXT.set(dict(input_context) if input_context else {})


def get_caller_input_context() -> Dict[str, Any]:
    """Read (without consuming) the stashed input_context; {} when none was set."""
    return _CALLER_INPUT_CONTEXT.get() or {}
