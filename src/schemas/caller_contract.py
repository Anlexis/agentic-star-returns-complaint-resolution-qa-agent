"""AgentCore Platform v1.0"""

# RET-C2-006 - caller-data contract (single source of truth).
#
# Everything a caller can send is untrusted: the question text, the structured
# parameters, and any identifier a customer-service question happens to carry.
# The rules live here, once, because they are enforced at TWO boundaries and
# the two must never drift apart:
#
#   src/nodes/pre_process_node.py   - the ingest boundary (outer backbone)
#   src/nodes/input_validate_node.py - the inner-graph boundary, so a direct
#                                      inner-graph invocation gets the same
#                                      fail-closed contract
#
# The direct-identifier patterns are also the OUTPUT-side invariant: the same
# set is re-applied by src/nodes/post_process_node.py so a customer identifier
# cannot reach the external surface by any route (retrieved passage, echoed
# text, future model-generated answer).
#
# Two rules run through all of it:
#   - fail CLOSED: an invalid value rejects the request, it is never clamped
#     into a "reasonable" one, because a silently clamped parameter changes
#     the answer without telling anyone;
#   - name the FIELD, never the VALUE: rejected caller data must not round-trip
#     into error logs or the response.

import re
from typing import Any, List, Optional, Pattern, Tuple

# Hard cap on the accepted question length (bounds the input; 2000 characters
# covers any realistic returns/complaint question).
MAX_QUERY_CHARS = 2000

# Bounds for the caller-supplied retrieval-depth override.
TOP_K_MIN = 1
TOP_K_MAX = 20

# Caller strings that select behaviour (the knowledge-base category filter, the
# request channel) must be inert identifiers: lowercase alphanumerics and
# underscore, bounded length. Free text in such a field is caller-controlled
# output/log injection, so it is rejected before it is ever compared or stored.
INERT_IDENTIFIER_RE: Pattern[str] = re.compile(r"^[a-z0-9_]{1,32}$")

# Control characters (tab and newline excepted) are stripped before processing.
CONTROL_CHARS_RE: Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE: Pattern[str] = re.compile(r"\s+")

# Instruction-override phrasings: text addressed to the answering model rather
# than a question addressed to the knowledge base. Deliberately narrow - a
# genuine returns question that happens to use these words ("what instructions
# do I give the customer?") does not match, because every alternative requires
# the imperative override shape.
#
# The template refuses these itself. A platform input gate may refuse them too,
# but this template must not depend on that: where such a gate is absent or
# configured off, an unchecked payload would otherwise reach the answer path
# and come back as a success. Refusal is expressed as BEHAVIOUR - error status,
# nothing carried forward - never as a particular gate's wording.
# Determiners and quantifiers may sit between the verb and the target
# ("ignore the above instructions", "forget all these previous prompts") - the
# attack is identical, so they are absorbed rather than relied on being absent.
_OVERRIDE_FILLER = r"(?:(?:all|any|the|these|those|your|my)\s+)*"
_OVERRIDE_SCOPE = r"(?:previous|prior|above|earlier|preceding)"
_OVERRIDE_TARGET = r"(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions)"
INJECTION_RE: Pattern[str] = re.compile(
    rf"(?:ignore|disregard|forget|discard)\s+{_OVERRIDE_FILLER}{_OVERRIDE_SCOPE}\s+"
    rf"{_OVERRIDE_FILLER}{_OVERRIDE_TARGET}"
    r"|(?:reveal|show|print|repeat|output|disclose)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system\s+prompt|system\s+message|instructions|initial\s+prompt)"
    r"|you\s+are\s+now\s+(?:a|an)\s"
    r"|act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+|an\s+)?(?:developer|admin|root)\s+mode"
    r"|override\s+(?:your|the)\s+(?:instruction|instructions|rules|safety)",
    re.IGNORECASE,
)

# -- Direct-identifier (customer PII) patterns --------------------------------
# A returns or complaint question routinely carries the identifiers of a real
# customer: a card or order number read off a receipt, a callback number, an
# e-mail address. None of them are needed to answer a POLICY question, so they
# are removed at ingest and re-checked at the output boundary.
#
# The digit-run rule is separate from the fixed-shape rules because a card
# number is written in several shapes ("4111111111111111",
# "4111 1111 1111 1111", "4111-1111-1111-1111"); matching a run of digits and
# separators and then COUNTING the digits covers every grouping in one rule,
# where a fixed 4-4-N regex leaves the final group behind.
IDENTIFIER_REPLACEMENT = "[REDACTED]"

_FIXED_SHAPE_IDENTIFIER_PATTERNS: List[Pattern[str]] = [
    # Card / account / order numbers written as 4-4-N groups.
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{2,11}\b"),
    # Telephone numbers: a leading-zero national number, grouped or not.
    re.compile(r"\b0\d{1,4}[- ]?\d{1,4}[- ]?\d{3,4}\b"),
    # E-mail addresses.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
]

# A run of digits, spaces and hyphens; the digit COUNT decides whether it is an
# identifier. 10-19 digits is the card / account / order / phone range. Policy
# prose ("within 30 days", "5 to 7 business days"), citation markers ("[1]"),
# knowledge-base ids ("kb-001") and years ("2026") all fall far below the
# threshold, and letters terminate the run.
_DIGIT_RUN_RE: Pattern[str] = re.compile(r"\b\d[\d -]{8,26}\d\b")
_IDENTIFIER_DIGITS_MIN = 10
_IDENTIFIER_DIGITS_MAX = 19


def _digit_run_is_identifier(match: "re.Match[str]") -> str:
    """Redact a digit run only when its digit count is in the identifier range."""
    token = match.group(0)
    digits = sum(1 for char in token if char.isdigit())
    if _IDENTIFIER_DIGITS_MIN <= digits <= _IDENTIFIER_DIGITS_MAX:
        return IDENTIFIER_REPLACEMENT
    return token


# One redaction pass is applied repeatedly until the text stops changing, up to
# this many rounds. Order matters between the rules and re-running removes the
# need to reason about it: the broad digit-run rule goes first so it can claim a
# whole grouped number, and any remainder a narrower rule exposes is caught on
# the next round rather than shipped. Two rounds are enough for every shape
# tested; the third is slack.
_REDACTION_ROUNDS = 3


def _redaction_pass(text: str) -> str:
    """One full sweep: the digit-run rule first, then the fixed-shape rules.

    The digit-run rule runs FIRST deliberately. A fixed 4-4-N card regex
    consumes only the first three groups of "4111 1111 1111 1111" and leaves
    the fourth behind as an unrecognisable fragment - a narrower rule running
    first destroys the shape the broader rule would have matched whole.
    """
    text = _DIGIT_RUN_RE.sub(_digit_run_is_identifier, text)
    for pattern in _FIXED_SHAPE_IDENTIFIER_PATTERNS:
        text = pattern.sub(IDENTIFIER_REPLACEMENT, text)
    return text


def strip_direct_identifiers(text: str) -> str:
    """Replace every direct-identifier token in *text* with the redaction marker.

    Applied at ingest (so raw identifiers never reach a domain node or a
    checkpoint) and again at the output boundary (so none can reach the
    external surface by any route). Sweeps until the text is stable, so the
    result does not depend on which rule happened to match first. Idempotent:
    running it on already-redacted text is a no-op.
    """
    for _ in range(_REDACTION_ROUNDS):
        swept = _redaction_pass(text)
        if swept == text:
            return text
        text = swept
    return text


def contains_direct_identifier(text: str) -> bool:
    """True when *text* still carries a direct-identifier token.

    Used as the output boundary's re-scan: the sanitised bytes that are about
    to be surfaced are checked again, so the invariant holds whatever the
    earlier layers did to the string.
    """
    return strip_direct_identifiers(text) != text


def is_instruction_override(text: str) -> bool:
    """True when *text* carries an instruction-override (prompt-injection) payload."""
    return bool(INJECTION_RE.search(text))


def validate_top_k(value: Any) -> Tuple[Optional[int], Optional[str]]:
    """Validate the untrusted retrieval-depth override. Returns (value, error).

    Fail-closed: only a plain integer within [1, 20] is accepted. Booleans
    (which are ints in Python), floats - including NaN and +/-Infinity, which
    parse through float() and arrive intact through a raw JSON body - numeric
    strings, and out-of-range integers all return a field-naming error. A
    non-finite value is the dangerous one: every comparison against NaN is
    False, so it would silently pass a range check written as an inequality.
    The offending value itself is never echoed.
    """
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"top_k must be an integer between {TOP_K_MIN} and {TOP_K_MAX}"
    if not TOP_K_MIN <= value <= TOP_K_MAX:
        return None, f"top_k must be an integer between {TOP_K_MIN} and {TOP_K_MAX}"
    return value, None


def validate_identifier_field(name: str, value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Validate an inert-identifier caller field. Returns (value, error).

    *name* is used only to build the error message - the rejected value is
    never included.
    """
    if value is None:
        return None, None
    if not isinstance(value, str) or not INERT_IDENTIFIER_RE.match(value):
        return None, f"{name} must be a lowercase identifier (a-z, 0-9, _; 1-32 chars)"
    return value, None


def normalise_query(text: str) -> Tuple[str, Optional[str]]:
    """Strip control characters, collapse whitespace, cap the length.

    Returns (normalised_text, note). The note is set when the text had to be
    truncated, so the caller can record that the answer was built from a
    shortened question.
    """
    cleaned = CONTROL_CHARS_RE.sub("", text)
    normalised = WHITESPACE_RE.sub(" ", cleaned).strip()
    if len(normalised) > MAX_QUERY_CHARS:
        return normalised[:MAX_QUERY_CHARS], f"query truncated to {MAX_QUERY_CHARS} characters"
    return normalised, None
