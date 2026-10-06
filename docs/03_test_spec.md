# Test Specification — RET-C2-006

**Template ID:** RET-C2-006
**Template Name:** CustomerReturnsComplaintQAAgent
**Category:** Cat 2 (nested retrieval pipeline)

This document is the contract the shipped tests implement. Every file named
below exists in this repository; every file in `tests/` is described here.

## 1. Scope and invocation conventions

- Per-node unit tests for the five inner domain nodes and the two outer gate slots.
- The caller-data contract module, tested directly.
- Manifest / runtime-config consistency and corpus integrity.
- Retrieval quality (golden queries over the bundled corpus).
- Inner-graph and outer-graph composition.
- Proof-of-boundary: import isolation, State msgpack safety, invoke order,
  HITL propagation (conditional), server boot, and end-to-end business
  behaviour through `POST /invoke`.

**Trust-gate invocation.** Behavioural per-node tests invoke the node via
`node(state)` — through `BaseNode.__call__`, which runs the trust gate, the
framework input mask, `execute()`, and the framework output scan. The state
builder sets `caller_trust_level` to `TrustLevel.VERIFIED_EXTERNAL.value` for
the two outer gate slots (the manifest's declared caller level) and
`TrustLevel.ANONYMOUS.value` for the five inner domain nodes.

**When a test calls `execute()` directly, and why.** One class of assertion
requires it: the refusals this template OWNS — instruction-override content,
the caller-data contract, the identifier strip, the output invariant. Asserting
those only through `__call__` would prove that *some* gate refused the payload,
which is a different statement and would keep passing after the template's own
check was removed — and would not hold in a deployment where the platform gate
is absent or configured off. Those tests therefore call `execute()` with no
framework wrapper in front of it, and assert BEHAVIOUR, never any gate's
wording.

**Two rejection outcomes, asserted apart.** The behaviour a rejection test
asserts depends on which half of the contract it exercises
(`docs/02_design.md` → *How a rejected request ends*), and the two are never
asserted with the same expectation:

- **Correctable** — empty / over-long input, a caller-data contract violation
  on either channel. The run COMPLETES: `status=SUCCESS`, with nothing
  published (no `validated_input`, no `search_query`) and the reason recorded
  in `error_log`. End to end, the caller reads the sentence naming what to
  correct and receives no citations.
- **Terminating** — instruction-override content, a credential pattern in the
  rendered answer, a direct identifier surviving the output re-scan, a
  trust-gate denial. The run ENDS: `status=ERROR`, nothing carried forward, and
  end to end no output at all (or, at the output boundary, the withheld-answer
  marker in place of the answer).

A test that accepted either status for either half would pass whichever way the
boundary behaved, which is precisely the regression these assertions exist to
catch.

**Both directions, always.** Every refusal rule is tested with the hostile form
AND the legitimate form: attack payloads are refused and ordinary questions
using the same words are unaffected; identifier shapes are redacted and policy
prose is byte-identical. A one-directional test proves a rule fires; it does not
prove the rule is the right shape.

**Node `execute()` contract.** Every `FunctionNode` here implements exactly
`def execute(self, state) -> dict` — no second parameter, no constructor
arguments. `RetrieveNode` / `RerankFilterNode` tuning knobs (`kb_path`,
`top_k`, `score_threshold`) are exercised by seeding
`state["retrieval_config"]`, the JSON field
`DomainWorkflowGraph._extra_initial_state()` republishes at runtime.

**Input-mask expectations.** The framework input gate masks
`user_input` / `validated_input` / `llm_response` (e-mail, phone/digit groups,
Title-Case name bigrams) to `[MASKED]` before `execute()` runs. Positive-path
payloads are therefore lowercase, identifier-free phrasing so the assertions are
about the template. Domain fields (`grounded_answer`, `formatted_answer`,
`retrieved_documents`, …) are not input-mask targets.

**Audit muting.** `shared.*` is never sys.modules-stubbed (the framework imports
`shared.security` at load time). The domain emitter is muted via an autouse
fixture patching `src.nodes.<mod>.emit_trace_event`; the audit assertion
re-patches the same attribute with a spy and asserts on `call.args[1]` (the
event payload).

## 2. Unit tests

### 2.1 `PreProcessNode` — `tests/unit/test_pre_process_node.py`

| Case | Input | Expected |
|------|-------|----------|
| Valid query | lowercase returns question | `status=SUCCESS`, `validated_input` set, `enriched_context` carries channel/source |
| Whitespace normalisation | ragged spacing / newlines | collapsed to single spaces |
| Empty / whitespace-only / missing / non-string input | — | correctable: `status=SUCCESS`, `error_log` non-empty, no `validated_input` |
| Over-long question | > 2000 chars | correctable: `status=SUCCESS`, no `validated_input` |
| Instruction-override refusal (via `execute()`) | 8 attack phrasings | terminating: `status=ERROR`, nothing carried forward, rejected text not echoed |
| Ordinary questions using the same words | 4 real questions | `status=SUCCESS`, `validated_input` set |
| Identifier strip (via `execute()`) | card group, hyphenated card, phone, e-mail | identifier absent from `validated_input`, marker present |
| Policy prose | "within 30 days … 5 to 7 business days … 2026" | `validated_input` byte-identical |
| Context-channel strip | `input_context.note` carrying a card number | identifier absent from the returned state |
| Caller contract fails closed | 14 malformed field values (incl. the non-finite matrix) | correctable: `status=SUCCESS`, no `validated_input`, value never echoed |
| Non-mapping `input_context` | a list | correctable: `status=SUCCESS` |
| Valid contexts | 7 shapes incl. absent/empty | `status=SUCCESS` |
| Audit | valid query / a refusal | `pre_process_complete` with `input_chars`; `pre_process_validation_failed` on refusal |

### 2.2 `InputValidateNode` — `tests/unit/test_input_validate_node.py`

| Case | Input | Expected |
|------|-------|----------|
| Plain text | free-text query | whole string becomes `search_query`; filters `{category: None, top_k: None}` |
| Whitespace | ragged spacing | collapsed |
| JSON envelope | `{"query","category","top_k"}` | all parsed; `question` alias accepted |
| Malformed JSON | `{`-prefixed non-JSON | treated as plain text + parse note |
| Bad `top_k`, envelope channel | 8 values incl. NaN/Infinity | correctable: `status=SUCCESS`, no `search_query` published |
| Bad `top_k`, context channel | same 8 values | correctable: `status=SUCCESS`, no `search_query` published |
| Bad `category`, either channel | free text, uppercase, over-length, punctuation | correctable: `status=SUCCESS` on both channels |
| Rejected value echo | distinctive marker string | absent from `error_log` |
| Channel precedence | both channels set | `input_context` wins |
| Non-mapping `input_context` | a string | correctable: `status=SUCCESS` |
| Instruction-override refusal (via `execute()`) | 3 attack phrasings | terminating: `status=ERROR`, no `search_query` / `query_filters` |
| Oversize query | > 2000 chars | truncated + note |
| Empty request | `""` | `search_query=""` + note (non-fatal) |
| Envelope `query` identifier strip | card number inside the envelope | redacted in `search_query` |
| State convention | any | `query_filters` is a JSON string, never a bare dict |

### 2.3 The caller-data contract — `tests/unit/test_caller_contract.py`

The rules in `src/schemas/caller_contract.py` are enforced at two boundaries and
re-used by the output gate, so they get direct tests: a rule that drifts here
drifts everywhere at once.

| Case | Expected |
|------|----------|
| `top_k` accepted | 1, 4, 20, absent |
| `top_k` rejected | out of range, bools, numeric strings, floats (incl. integral), NaN, ±Infinity, containers |
| `top_k` error text | names the field, never the value |
| Raw JSON non-finite | bare `NaN` / `Infinity` / `-Infinity` in a body still rejects |
| Identifier fields accepted / rejected | inert slugs pass; free text, mixed case, over-length, punctuation, non-strings reject |
| Instruction-override detection | 10 attack forms flagged; 7 ordinary questions not flagged |
| Identifier forms removed COMPLETELY | 10 shapes; no residual survives the sweep |
| Structural text byte-identical | 12 tokens (`kb-001`, `[1]`, "30 days", years, versions, section numbers, scores) |
| Multiple identifiers in one string | all removed |
| Idempotence | a second sweep is a no-op |
| The shipped corpus | every entry, rendered as an answer, survives the sweep unchanged |
| Query normalisation | control characters stripped, whitespace collapsed, truncation noted |
| `finite_in_range` (the shared numeric parser used by the graph boundary AND the nodes) | real numbers parsed and clamped; NaN, ±Infinity, numeric strings, bools, containers and None all take the supplied default |

### 2.4 `RetrieveNode` — `tests/unit/test_retrieve_node.py`

| Case | Expected |
|------|----------|
| Happy path / ordering | top-1 is `kb-001`; scores strictly descending and > 0 |
| Entry shape | keys `{id,title,category,source,score,excerpt}`; excerpt ≤ 400 chars |
| Category filter | only matching entries; warranty filter → `kb-010` |
| Empty query | no candidates |
| `retrieval_config` precedence | unreadable corpus degrades with a note and `[]`; a valid override is honoured |
| No `retrieval_config` seeded | falls back to the module defaults |
| Notes accumulation | appended, never clobbered |

### 2.5 `RerankFilterNode` — `tests/unit/test_rerank_filter_node.py`

| Case | Expected |
|------|----------|
| Relevance floor | below-threshold candidates dropped |
| Threshold / `top_k` overrides | state-seeded values honoured |
| Category boost and cap | +0.1, re-ranked ahead, capped at 1.0 |
| Caller `top_k` | a stricter value wins; a looser one does not widen |
| Garbage entries | non-dict skipped; uncoercible score → 0.0 and dropped |
| Tie-break | deterministic id-ascending order |
| Non-finite seeded `score_threshold` | 8 values → the declared default floor applies; the passages the corpus covers still survive (clamping NaN would silently install the strictest possible floor and answer "insufficient coverage" for everything) |
| Non-finite seeded `top_k` | 5 values → the declared default applies |
| Boolean caller `top_k` | does not collapse the result set to one passage |
| Non-finite candidate score | treated as zero and dropped |

### 2.6 `GenerateAnswerNode` — `tests/unit/test_generate_answer_node.py`

| Case | Expected |
|------|----------|
| Citation markers | `[1]`/`[2]` markers with titles |
| Static lead sentence | a distinctive marker planted in the query is absent from the answer |
| Lead sentence invariance | identical with and without a query |
| Citations list | refs 1..n mirror ranked order; id/title/source carried |
| Groundedness | the answer body traces to ranked passages only |
| No coverage | empty `ranked_documents` → escalation answer; `citations=[]` |

### 2.7 `OutputFormatNode` — `tests/unit/test_output_format_node.py`

| Case | Expected |
|------|----------|
| Full compose | header, body, `## Sources` rows, scope note, advisory disclaimer; `status=SUCCESS` |
| Blank source | no `()` suffix |
| Disclaimer | rides with every answer |
| No citations | explicit "- none (…)" sources line |
| Missing body | fallback text; `status=SUCCESS` |

### 2.8 `PostProcessNode` — `tests/unit/test_post_process_node.py`

| Case | Input (`result`) | Expected |
|------|------------------|----------|
| Clean output | normal answer | `formatted_output=result`, `status=SUCCESS` |
| Empty result | `""` | forwarded as-is, `status=SUCCESS` (non-fatal) |
| Credential scan | API key / JWT / Bearer / assignment (all built at runtime) | terminating: whole answer WITHHELD, `status=ERROR`, raw secret absent from both surfaced fields |
| Direct identifier | 5 shapes | absent from the output, in grouped and ungrouped form |
| Structural tokens | 6 policy-prose forms | output byte-identical, `status=SUCCESS` |
| Identifier next to a credential | both present | terminating: `status=ERROR`, withheld; neither survives |
| Verbatim caller text | the question embedded in the answer | redacted |
| Redaction does not expose a new identifier | identifier inside the redacted span | absent from the final bytes (the re-scan sees the post-redaction text) |
| `result` and `formatted_output` | any gated path | gated together |
| No monetary grid | 4 numeric policy phrasings | unchanged — the absence of a rounding grid is deliberate and pinned |

### 2.9 Manifest / runtime config — `tests/unit/test_config_manifest.py`

| Case | Expected |
|------|----------|
| Manifest is flat | no nested `agent:` block; `id` at root |
| Identity | `name`, `namespace`, `category`, `industry`, `base_type`, `enabled` |
| Class path | `class` is a single dotted path resolving to the graph class the server imports |
| `generation_mode` | `deterministic` — matches an implementation that calls no model |
| `requires` | `secrets` and `extras` both empty — nothing declared that the runtime does not provision |
| Trust level | manifest `VERIFIED_EXTERNAL` == both outer slots' `required_trust_level` |
| HITL | not enabled |
| Runtime config is live | `_runtime_config()` returns the contents of `config/config.yaml` — reading the manifest instead would return nothing and every declared runtime value would go dead |
| `max_retry` / `timeout_s` | int within the framework ceiling; the retired `timeout_seconds` key is absent |
| Retrieval block | mirrors the node module defaults; `kb_path` exists |
| `_parent_config()` | forwards the runtime blocks; never `{}` |
| Corpus integrity | JSON list ≥ 5 entries; unique ids; required keys per entry |
| Corpus categories | every shipped category satisfies the identifier alphabet the contract accepts |

### 2.10 Trust matrix — `tests/unit/test_trust_gate.py`

| Case | Expected |
|------|----------|
| ANONYMOUS on an inner node | allowed; the node produces a normalised query |
| ANONYMOUS on either outer slot | denial dict RETURNED (never raised), `status=ERROR`, "trust gate denied" in `error_log`, execute-only keys ABSENT |
| VERIFIED_EXTERNAL on either outer slot | passes; the node writes its output |
| Empty input after the gate | the gate passes, then the node's own validation declines it — correctable: `status=SUCCESS`, the reason recorded in `error_log` |
| Full 7-node matrix | outer slots VERIFIED_EXTERNAL, all five inner nodes ANONYMOUS |

### 2.11 Retrieval quality — `tests/unit/test_retrieval_quality.py`

| Case | Expected |
|------|----------|
| Golden domain queries (one per corpus entry) | the expected entry is top-1 |
| Relevance floor | every survivor ≥ 0.25 |
| Citation integrity | every survivor id exists in the corpus |
| Precision | a front-line-complaint query keeps only `kb-007` |
| Category filter | warranty filter → warranty entries, top-1 `kb-010` |
| No coverage | out-of-domain query → zero survivors, explicit escalation answer, no citations |

### 2.12 Framework compliance — `tests/unit/test_framework_compliance_tc06_tc07.py`

| Case | Expected |
|------|----------|
| TC-06 / TC-07 | overriding the framework's default input or output gate raises at class definition; domain nodes extend only through the sanctioned `_extra_*` hooks |

## 3. Integration and composition

### 3.1 Inner graph — `tests/unit/test_domain_workflow_graph.py`

| Case | Expected |
|------|----------|
| Composition | inherits `BaseGraph`; registers exactly the five domain nodes; no initialize/finalize |
| Config forwarding | `_extra_initial_state()` republishes the retrieval block as the JSON-string `retrieval_config` |
| Caller-context seeding | the same hook returns the bridged `input_context` |
| Output shape | `get_output()` emits the merge contract; `route()` → END on error |
| Inner end-to-end | full inner `invoke()` → SUCCESS; formatted answer, disclaimer, `kb-001` citation; inner `node_history` = the five domain nodes in order |

### 3.2 Outer graph — `tests/unit/test_graph_composition.py`

| Case | Expected |
|------|----------|
| Outer composition | inherits the framework base class directly; `Graph` alias; `add_edges()` NOT overridden |
| Backbone slots | `compile()` fills all five with the expected classes |
| `get_subgraph()` | returns the inner graph carrying the forwarded config |
| `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| `merge_output()` | inner `formatted_answer` → outer `returns_kb_answer` AND `result`; changed keys only; the delta also carries the reason marker, blank for a run that produced an answer |
| Unreadable runtime file | `_parent_config()` forwards nothing and does not raise; nodes keep their defaults |
| Non-finite declared numbers | dropped rather than forwarded |
| Context bridge | `extract_input()` stashes `input_context` for the inner graph |
| End-to-end happy path | VERIFIED_EXTERNAL invoke → SUCCESS; `output` is the gated answer; post_process traversed |
| End-to-end trust denial | ANONYMOUS invoke → terminating: `status=ERROR`; empty `output`; post_process NOT traversed |
| State helpers | `to_json`/`from_json` round-trip; None/malformed handling |

## 4. Proof-of-boundary

| File | Expected |
|------|----------|
| `test_import_isolation.py` | no platform-SDK import anywhere under `src/` |
| `test_state_safety.py` | `State` has no credential-named fields and no `BaseModel` / `InvocationContext` annotations |
| `test_pb_invoke_order.py` | full `Graph().invoke()` with `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` (never `for_internal()`) over the payload byte-equal to `deploy/invoke_payload.json`'s `input` → SUCCESS with outer `node_history` exactly `[InitializeNode, PreProcessNode, ReturnsComplaintSearchGraphNode, PostProcessNode, FinalizeNode]` |
| `test_pb7_hitl_interrupt_propagation.py` | **Auto-waived — non-HITL** (no `hitl.enabled: true`; no graph class declares `propagate_hitl=True`); the conditional skip stub is retained |
| `test_server_boot.py` | `import src.api.server` does not raise; the module-level agent is this template's class, compiled; a fresh ctor→`compile()` fills the five slots; `/health` reports the agent |
| `test_invoke_e2e.py` | end-to-end business behaviour through the real ASGI `POST /invoke` — see below |

### 4.1 End-to-end through `POST /invoke` — `tests/proof_of_boundary/test_invoke_e2e.py`

Unlike `test_server_boot.py` (which stubs the agent to isolate the auth
boundary), these run the REAL compiled agent: every request crosses the
entry-point auth, the outer trust and input gates, the `input_context` bridge
into the inner graph, all five domain nodes, and the output gate. The app is
driven through its real ASGI interface (no TestClient — httpx is only a
transitive dependency).

| Case | Expected |
|------|----------|
| Grounded question | `status=success`; header, `[1]` marker, `## Sources`; not the no-coverage text |
| Scope note and disclaimer | both present in the answer |
| No coverage | `status=success` with the explicit insufficient-coverage answer and the "none" sources line — not a fabrication |
| Category filter | narrows the corpus to the filtered entry |
| `top_k` override | changes the citation count |
| Unknown category | no coverage, not a silently unfiltered answer |
| `top_k` rejection | 9 values incl. the non-finite matrix → correctable: `status=success`, the answer body is the sentence naming what to correct, no citation lines |
| Raw JSON `NaN` body | refused, not silently compared — HTTP 200 with `status=success` and no answer produced |
| Malformed context field | 5 shapes → correctable: `status=success`, the correction sentence, no citation lines |
| Oversize `input_context` | HTTP 413 at the adapter |
| Instruction-override | 3 attack payloads → terminating: `status=error`, no output |
| Ordinary question with the same words | `status=success` with an answer |
| Customer identifiers in the question | absent from the answer (card, phone, e-mail) |
| Question not quoted back | a planted fake citation marker and instruction do not appear in the answer |
| Citation integrity | every rendered `[n]` marker has a matching Sources entry |

> **Gate checklist:** import isolation, State safety, invoke order, server boot
> and the end-to-end invoke suite are mandatory. The HITL boundary applies only
> to HITL-enabled templates — this template is non-HITL, so it is auto-waived
> and its skip must not block the gate.

## 5. Execution summary

- Runner: the real `agenticstar-agentcore==1.0.1` wheel — the version CI installs.
- Total collected: 382 (`tests/` full tree, including `tests/proof_of_boundary/`).
- Pass 381 / Fail 0 / Skip 1 (the HITL conditional stub — auto-waived, non-HITL).
- `tests/proof_of_boundary/` alone: 41 pass / 0 fail / 1 skip (42 collected).
- Determinism: no model call, no network; retrieval and answer assembly are
  rule-based, so a given question returns a byte-identical answer.
