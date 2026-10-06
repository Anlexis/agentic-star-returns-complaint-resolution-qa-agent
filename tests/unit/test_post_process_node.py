# RET-C2-006 — Unit Tests: PostProcessNode (outer post_process slot, output gate)
#
# Invocation: node(state) via BaseNode.__call__ for the behavioural paths;
# execute() directly where the assertion is about THIS node's layers rather
# than about some gate having refused the payload.
#
# The gate's layer ORDER is under test as well as its rules: the pattern scans
# run before the rewriting layers and the sanitised bytes are scanned AGAIN, so
# no transform can destroy or expose a pattern on the way out. The
# whole-answer-withheld path is asserted for both scan families.
#
# Mirrors docs/03_test_spec.md §2.7.
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import PostProcessNode

_CLEAN_ANSWER = (
    "# Customer Returns & Complaint Resolution Search Result\n\n"
    "[1] general merchandise may be returned within 30 days with proof of purchase.\n"
)

# Credential-shaped strings are built at runtime so no credential-shaped
# literal ever sits in the repository.
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12
_FAKE_API_KEY = "sk-" + "ABCDEF0123456789" + "abcdef"
_FAKE_BEARER = "Bearer " + "abcdefghijklmnopqrstuvwxyz0123456789"
_FAKE_ASSIGNMENT = "password=" + "super_secret_value_123"


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestCleanOutput:
    def test_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_ANSWER))
        assert result["status"] == AgentStatus.SUCCESS.value
        # State carries the plain string, never the enum.
        # An identity check on the TYPE is the point here: AgentStatus values
        # would satisfy isinstance() against str, which is exactly the bug
        # this guards against.
        assert type(result["status"]) is str  # noqa: E721
        assert result["formatted_output"] == _CLEAN_ANSWER

    def test_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestCredentialScanWithholdsEntirely:
    def _assert_withheld(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("withheld" in str(e) for e in result["error_log"])
        # The raw secret must not survive into either surfaced field.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert "ANSWER WITHHELD" in result["formatted_output"]

    @pytest.mark.parametrize(
        "secret",
        [_FAKE_API_KEY, _FAKE_JWT, _FAKE_BEARER, _FAKE_ASSIGNMENT],
    )
    def test_credential_pattern_withholds_the_answer(self, secret):
        result = PostProcessNode()(_make_state(f"# Answer\n\ninternal note: {secret}\n"))
        self._assert_withheld(result, secret)


class TestDirectIdentifierBoundary:
    """The stated invariant: no customer identifier on the external surface."""

    @pytest.mark.parametrize(
        "identifier",
        [
            "4111 1111 1111 1111",
            "4111-1111-1111-1111",
            "4111111111111111",
            "090-1234-5678",
            "shopper@example.com",
        ],
    )
    def test_identifier_never_reaches_the_output(self, identifier):
        answer = f"{_CLEAN_ANSWER}\nThe customer reference was {identifier}.\n"
        result = PostProcessNode().execute(_make_state(answer))
        surfaced = result["formatted_output"]
        assert identifier not in surfaced
        assert identifier.replace("-", "").replace(" ", "") not in surfaced

    @pytest.mark.parametrize(
        "structural",
        [
            "returns are accepted within 30 days",
            "card refunds take 5 to 7 business days",
            "see kb-001 and kb-010 for the policy",
            "[1] Standard return eligibility window",
            "the policy was last revised in 2026",
            "reference the 2026-08-25 policy update",
        ],
    )
    def test_structural_tokens_are_byte_identical(self, structural):
        """The other direction: the gate must not touch policy prose."""
        answer = f"{_CLEAN_ANSWER}\n{structural}\n"
        result = PostProcessNode().execute(_make_state(answer))
        assert result["formatted_output"] == answer
        assert result["status"] == AgentStatus.SUCCESS.value


class TestLayerOrderAndRescan:
    """Pattern scans run BEFORE the rewriting layers, and again after.

    A scan-then-transform pipeline is order-sensitive: a rewrite can destroy
    the shape a later pattern would have matched. These assert the finished
    bytes, so the guarantee holds however the middle layers behaved.
    """

    def test_identifier_adjacent_to_a_credential_still_withholds(self):
        answer = f"{_CLEAN_ANSWER}\ncard 4111 1111 1111 1111 and key {_FAKE_API_KEY}\n"
        result = PostProcessNode().execute(_make_state(answer))
        assert result["status"] == AgentStatus.ERROR.value
        assert _FAKE_API_KEY not in result["formatted_output"]
        assert "4111" not in result["formatted_output"]

    def test_verbatim_caller_text_is_redacted(self):
        question = "can I return this defective toaster after 40 days"
        answer = f"{_CLEAN_ANSWER}\nyou asked: {question}\n"
        result = PostProcessNode().execute(_make_state(answer, user_input=question, validated_input=question))
        assert question not in result["formatted_output"]
        assert "[REDACTED]" in result["formatted_output"]

    def test_redaction_does_not_expose_a_new_identifier(self):
        """The re-scan sees the post-redaction bytes, not the original."""
        question = "please refund order 4111 1111 1111 1111 immediately"
        answer = f"{_CLEAN_ANSWER}\nyou asked: {question}\n"
        result = PostProcessNode().execute(_make_state(answer, user_input=question, validated_input=question))
        assert "4111" not in result["formatted_output"]

    def test_result_and_formatted_output_are_gated_together(self):
        answer = f"{_CLEAN_ANSWER}\ncustomer card 4111 1111 1111 1111\n"
        result = PostProcessNode().execute(_make_state(answer))
        assert "4111" not in str(result["result"])
        assert "4111" not in str(result["formatted_output"])


class TestNoMonetaryGridIsImposed:
    """This template renders no computed monetary figures, so no rounding grid
    applies (docs/02_design.md, output schema). The point of this test is that
    the absence is DELIBERATE and stays that way: a grid retro-fitted here
    would rewrite the structural numbers the policy text depends on."""

    @pytest.mark.parametrize(
        "prose",
        [
            "refunds are issued within 30 days",
            "escalate after 3 business days",
            "the item was purchased in 2026",
            "return code 12345 applies",
        ],
    )
    def test_numbers_in_policy_prose_survive_unchanged(self, prose):
        answer = f"{_CLEAN_ANSWER}\n{prose}\n"
        assert PostProcessNode().execute(_make_state(answer))["formatted_output"] == answer
