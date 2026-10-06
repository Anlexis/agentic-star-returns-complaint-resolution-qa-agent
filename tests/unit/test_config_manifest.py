# RET-C2-006 — Unit Tests: manifest / runtime-config consistency
#
# Two files, two jobs:
#   config/agent.yaml  — the static manifest the registry reads for discovery:
#                        identity, entry-point class path, declared trust level,
#                        and the compile-time `requires` declarations. Flat —
#                        every key at root level, no nested `agent:` block.
#   config/config.yaml — the runtime parameters the registry passes as
#                        Graph(config=...). The standalone server reads the
#                        same file, so a declared value is live in both
#                        deployments.
#
# These pin config <-> code consistency so a drift fails fast rather than
# silently changing tuning at runtime.
#
# Mirrors docs/03_test_spec.md §2.8.
# Deterministic — no LLM, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import (
    CustomerReturnsComplaintQAAgent,
    ReturnsComplaintSearchGraphNode,
    _runtime_config,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIsFlat:
    def test_no_nested_agent_block(self):
        """Every key sits at root level — a nested `agent:` block is the
        retired shape and the registry would not read it."""
        assert "agent" not in _MANIFEST
        assert _MANIFEST["id"] == "RET-C2-006"

    def test_identity_fields(self):
        assert _MANIFEST["name"] == CustomerReturnsComplaintQAAgent().name
        assert _MANIFEST["namespace"] == "ret"
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "RET"
        assert _MANIFEST["base_type"] == "RAGAgent"
        assert _MANIFEST["enabled"] is True

    def test_declared_class_is_the_graph_class(self):
        """The manifest carries a single dotted import path, and it must
        resolve to the class the server imports."""
        assert _MANIFEST["class"] == "src.graph.graph.CustomerReturnsComplaintQAAgent"
        module_path, _, class_name = _MANIFEST["class"].rpartition(".")
        assert module_path == CustomerReturnsComplaintQAAgent.__module__
        assert class_name == CustomerReturnsComplaintQAAgent.__name__

    def test_generation_mode_matches_the_implementation(self):
        """The answer is rule-assembled from retrieved passages; no model is
        called, so nothing may be declared that would not be provisioned."""
        assert _MANIFEST["generation_mode"] == "deterministic"

    def test_requires_declares_nothing_unprovisioned(self):
        """A declared secret or extra that the runtime does not provision
        fails the agent at compile time — so both stay empty until the code
        actually needs one."""
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []


class TestManifestSecurity:
    def test_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared

    def test_hitl_is_not_enabled(self):
        """This template declares no human-in-the-loop step, so no
        checkpointer is required."""
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False


class TestRuntimeConfigIsLive:
    def test_runtime_config_reads_the_live_file(self):
        """_runtime_config() must read config/config.yaml — the file the
        registry loads. Reading the manifest instead would silently return
        nothing, and every declared runtime value would go dead."""
        assert _runtime_config() == _RUNTIME

    def test_max_retry_within_framework_ceiling(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # AgentBaseGraph retry ceiling

    def test_timeout_uses_the_current_key(self):
        assert isinstance(_RUNTIME["timeout_s"], int)
        assert "timeout_seconds" not in _RUNTIME

    def test_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror the runtime file — a drift silently
        # changes tuning for a directly-constructed node.
        from src.nodes.rerank_filter_node import _DEFAULT_RETRIEVAL as rerank_defaults
        from src.nodes.retrieve_node import _DEFAULT_RETRIEVAL as retrieve_defaults

        retrieval = _RUNTIME["retrieval"]
        assert retrieval["top_k"] == retrieve_defaults["top_k"] == rerank_defaults["top_k"]
        assert (
            retrieval["score_threshold"] == retrieve_defaults["score_threshold"] == rerank_defaults["score_threshold"]
        )
        assert retrieval["kb_path"] == retrieve_defaults["kb_path"]
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_parent_config_forwards_the_runtime_blocks(self):
        cfg = ReturnsComplaintSearchGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"] == _RUNTIME["retrieval"]
        assert cfg["configurable"]["llm"] == _RUNTIME["llm"]
        assert cfg["configurable"]["retrieval"], "_parent_config() must never forward an empty retrieval block"


class TestSeededKnowledgeBase:
    def test_kb_is_a_well_formed_entry_list(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) >= 5, "seeded knowledge base must carry a usable corpus"
        for entry in entries:
            assert set(entry.keys()) == {"id", "title", "category", "source", "tags", "content"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids))

    def test_kb_categories_are_inert_identifiers(self):
        """A caller selects a category by name, so every shipped category must
        satisfy the identifier alphabet the contract accepts — otherwise a
        legitimate filter would be rejected."""
        from src.schemas.caller_contract import INERT_IDENTIFIER_RE

        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        for entry in entries:
            assert INERT_IDENTIFIER_RE.match(entry["category"]), entry["category"]
