# Template Design Specification — RET-C2-006

**Template ID:** RET-C2-006
**Template Name:** CustomerReturnsComplaintQAAgent
**Category:** Cat 2 (multi-step domain workflow — retrieval pattern)
**Industry:** RET

## Position in the AgentCore architecture

| Role | Class |
|---|---|
| Agent class | `CustomerReturnsComplaintQAAgent` (alias `Graph`) |
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Inner graph base | `BaseGraph` — `DomainWorkflowGraph` |

- **Pattern:** two-layer nested architecture — the outer fixed five-node backbone
  plus a `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain
  workflow.
- **Three-layer separation:**
  - State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
    structured fields stored as JSON strings via `to_json()` / `from_json()`
  - Node: `FunctionNode` subclasses overriding `execute(self, state) -> dict`
    ONLY — no extra parameters such as `config`, and no constructor arguments
  - Graph: composition (`register_nodes()` for node substitution); the outer
    `add_edges()` is NOT overridden

## Purpose

Customer returns and complaint-resolution Q&A. Service-desk staff, store
associates, or a customer-facing chat ask natural-language questions about
return eligibility, refund procedures, exchange policy, or complaint
escalation; the agent retrieves from a returns-and-complaints policy knowledge
base and returns a grounded, cited, policy-consistent answer — reducing manual
policy lookup and cross-store answer inconsistency.

The agent answers policy questions. It does not approve or deny a return, call
an order-management or refund system, or decide an individual case.

The current build is deterministic end-to-end: keyword retrieval plus
rule-based answer assembly, with no model call (see *Answer synthesis* below).

## Configuration files

| File | Read by | Contents |
|---|---|---|
| `config/agent.yaml` | the registry, at discovery | Static manifest: identity, the dotted entry-point class path, declared trust level, and the compile-time `requires` block. Flat — every key at root level. |
| `config/config.yaml` | the registry (passed as `Graph(config=...)`) and `_runtime_config()` in the standalone server | Runtime parameters: `max_retry`, `timeout_s`, and the `retrieval` / `llm` tuning blocks. |

Both deployments read the same runtime file, so a declared value is live in
both. `requires.secrets` and `requires.extras` are empty because the code
constructs no client and requires no secret — declaring one that the runtime
does not provision fails the agent at compile time.

## Architecture overview

### Outer backbone (`AgentBaseGraph`)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max_retry)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | `InitializeNode` (framework default) | session_id, trust_level, schema_version | — (framework) |
| pre_process | `PreProcessNode` | Ingest boundary: settle a correctable rejection (empty / over-long question, caller-data contract violation) by publishing a reason marker, refuse instruction-override content outright, strip direct customer identifiers → `validated_input` | `TrustLevel.VERIFIED_EXTERNAL` |
| main | `ReturnsComplaintSearchGraphNode` (`GraphNode`) | Skips the inner run when a reason marker is already set; otherwise delegates to the inner `DomainWorkflowGraph`, bridges `input_context`, maps inner `formatted_answer` → outer `result` | — (delegation) |
| post_process | `PostProcessNode` | Output boundary: renders the reason marker as the caller-facing sentence when one is set; otherwise credential scan, direct-identifier scan, verbatim caller-text redaction, then a re-scan of the sanitised bytes | `TrustLevel.VERIFIED_EXTERNAL` |
| finalize | `FinalizeNode` (framework default) | response_metadata, total_time_ms | — (framework) |

### Inner graph (`DomainWorkflowGraph` — `BaseGraph`, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner domain nodes declare `required_trust_level =
TrustLevel.ANONYMOUS`: the external trust gate lives on the outer backbone, and
a stricter inner level would deny a real VERIFIED_EXTERNAL invoke at runtime.

| Node | Responsibility | Input State | Output State |
|------|----------------|-------------|--------------|
| `InputValidateNode` | Re-validate the caller contract on both channels (a violation settles a reason marker), refuse instruction-override content outright, normalise and length-cap the question | `validated_input` \| `user_input`, `input_context` | `search_query`, `query_filters`, `intake_notes` |
| `RetrieveNode` | Deterministic keyword retrieval over `config/kb/returns_complaints_kb.json`: tokenise, score title/tags/content overlap, apply the category filter | `search_query`, `query_filters`, `retrieval_config` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Rerank (category-match boost), drop entries below `score_threshold`, cap at `top_k` | `retrieved_documents`, `query_filters`, `retrieval_config` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based answer assembly from the ranked passages only, with numbered citation markers | `ranked_documents` | `grounded_answer`, `citations` |
| `OutputFormatNode` | Compose the final answer: body, Sources list, scope note, advisory disclaimer | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Data flow

```
user_input + input_context
  → PreProcessNode                                → validated_input
  → ReturnsComplaintSearchGraphNode.extract_input → inner DomainWorkflowGraph.invoke(validated_input)
        → input_validate                          → search_query / query_filters
        → retrieve                                → retrieved_documents
        → rerank_filter                           → ranked_documents
        → generate_answer                         → grounded_answer / citations
        → output_format                           → formatted_answer
     get_output() → {formatted_answer, citations, status, error_code, ...}
  → ReturnsComplaintSearchGraphNode.merge_output  → result = formatted_answer, returns_kb_answer
  → PostProcessNode                               → formatted_output (gated)
```

## How a rejected request ends

Not every rejection is the same kind of event, so they do not end the same way.

**A request the caller can fix completes.** Empty, whitespace-only, missing or
non-string input; a question over the length bound; a value that violates the
caller-data contract on either channel — each of these is a correctable
mistake. The run completes with `status = SUCCESS` and a reason marker
(`error_code`) naming what went wrong, and the answer body is the sentence from
`src/services/failure_message.py` that tells the caller what to correct. No
knowledge-base passage is retrieved and no citation is rendered — the request
was not carried out; it was declined in a form the caller can act on.

Terminating instead would end the calling surface's turn and surface an
exception type, leaving the reason reachable only from the audit trail.
Completing lets the caller correct the request and send it again on the same
conversation.

**A refusal the agent owns terminates.** Instruction-override content is not a
mistake the caller corrects by rewording, so `PreProcessNode` and
`InputValidateNode` end the run with `status = ERROR` and publish nothing. The
output boundary behaves the same way: a credential pattern in the rendered
answer, or a direct customer identifier still present in the sanitised bytes,
withholds the whole answer with `status = ERROR`. A trust-gate denial at either
outer slot likewise returns `status = ERROR` without `execute()` ever running.
None of these degrade into a partial answer.

| Condition | `status` | `error_code` | What the caller receives |
|---|---|---|---|
| Empty / whitespace-only / missing / non-string question | `SUCCESS` | `EMPTY_INPUT` | the sentence asking for a question |
| Question over `MAX_QUERY_CHARS` | `SUCCESS` | `QUESTION_TOO_LONG` | the sentence asking for a shorter request |
| `input_context` contract violation (bad `channel` / `category` / `top_k` / `note`, non-mapping context) | `SUCCESS` | `INVALID_REQUEST` | the sentence pointing at the documented format |
| Instruction-override content | `ERROR` | — | nothing |
| Credential pattern in the rendered answer | `ERROR` | — | the withheld-answer marker |
| Direct customer identifier surviving the output re-scan | `ERROR` | — | the withheld-answer marker |
| Trust gate denied at an outer slot | `ERROR` | — | nothing |

**How the marker travels.** `error_code` is set once, at the boundary that
settled the reason, and every node after it passes it through untouched rather
than doing work on input that was already declined: the `main` slot returns
early instead of running the inner graph, and `RetrieveNode`,
`RerankFilterNode`, `GenerateAnswerNode` and `OutputFormatNode` each return the
marker unchanged if one reaches them. Running the pipeline anyway would produce
a second, vaguer reason for the same rejection and overwrite the specific one.
`merge_output()` prefers the outer marker over anything the subgraph reports,
for the same reason.

**The marker is internal.** The invocation envelope carries `output`, `status`,
`trace_id`, `correlation_id` and `node_history` — `error_code` is deliberately
not among them. The reason reaches the caller as the message text and nothing
else: a machine-readable code on the external surface would become a contract
this template has not agreed to keep stable, and the field paths and gate
wording behind it stay in `error_log`, the internal audit channel.

## The caller-data contract

A caller supplies the question as `input` and structured parameters as
`input_context`. The rules live in one module, `src/schemas/caller_contract.py`,
because they are enforced at two boundaries and re-used by the output gate — a
rule that drifted in one place would silently open a hole in another.

| Field | Accepted | Absent |
|---|---|---|
| `channel` | `^[a-z0-9_]{1,32}$` | `"unknown"` |
| `category` | `^[a-z0-9_]{1,32}$` | no category filter |
| `top_k` | integer, 1–20 | the configured default |
| `note` | free text, identifier-stripped, ≤ 2000 chars | omitted |

Three rules run through all of it:

1. **Fail closed.** An invalid value rejects the request; it is never clamped
   into a "reasonable" one, because a silently adjusted parameter changes the
   answer without telling anyone. Rejecting here means the run completes
   carrying `INVALID_REQUEST` and answers nothing (see *How a rejected request
   ends*) — the request is not partially honoured with a substituted value.
2. **Name the field, never the value.** Rejected caller data does not
   round-trip into error logs or the response.
3. **Numbers must be finite.** `top_k` is accepted only as a plain integer in
   range. Bools (which are ints in Python), floats, numeric strings, and
   NaN/±Infinity are all refused. The non-finite case is the dangerous one:
   Python's `json` parses a bare `NaN` in a request body, and every comparison
   against NaN is False — so a range check written as an inequality would pass
   it.

Behaviour-selecting strings are locked to an inert identifier alphabet rather
than merely escaped: free text in a field that reaches a comparison, a log, or
the rendered output is caller-controlled injection.

Two channels carry the structured parameters — `input_context`, and a JSON
envelope inside the question string (`{"query": ..., "category": ...,
"top_k": ...}`) for callers that can only send text. **Both are validated with
the same rules**, `input_context` wins where they overlap, and an invalid value
on either rejects the request — completing with `INVALID_REQUEST` and no query
published — so neither is a way round the other.

The adapter (`src/api/server.py`) additionally caps the serialized
`input_context` size before it reaches the graph at all.

### Carrying `input_context` into the inner graph

`GraphNode.execute()` invokes the inner graph without forwarding the outer
state's `input_context`, so an inner-node read of `state["input_context"]` would
always see `{}` through the full nested graph. `src/graph/context_bridge.py`
bridges it with a `ContextVar`:

```
ReturnsComplaintSearchGraphNode.extract_input(state)   [BEFORE subgraph.invoke]
    → set_caller_input_context(state["input_context"])
DomainWorkflowGraph._extra_initial_state()             [INSIDE subgraph.invoke]
    → {"input_context": get_caller_input_context()}
```

A `ContextVar` keeps the hand-off correct per thread/task, so concurrent
invocations in one process cannot see each other's context. This is proven
end-to-end (a category filter and a `top_k` override changing the answer through
`POST /invoke`), not at node level — a node-level test would pass even with the
bridge removed.

## Runtime settings forwarding

```
config/config.yaml `retrieval` / `llm` blocks
  → ReturnsComplaintSearchGraphNode._parent_config()   (type/range/finiteness validated)
  → DomainWorkflowGraph ctor config
  → DomainWorkflowGraph._extra_initial_state()          (seeds State["retrieval_config"] as JSON)
  → RetrieveNode / RerankFilterNode read state["retrieval_config"] via from_json(),
    falling back to module defaults that mirror the file when unseeded
    (e.g. a bare unit test that constructs the node directly).
```

Every forwarded number is validated where it is read, so a malformed
configuration file can neither crash graph construction nor weaken the
relevance floor. An invalid or absent key is simply not forwarded. The
non-finite case matters here too: a NaN `score_threshold` would compare False
against every candidate and silently empty the answer instead of applying the
declared floor.

## State definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `Optional[str]` | identifier-stripped query payload | outer |
| `returns_kb_answer` | `Optional[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `search_query` | `Optional[str]` | normalised search query | inner |
| `query_filters` | `Optional[str]` (JSON) | validated structured params (`category`, `top_k`) | inner |
| `retrieval_config` | `Optional[str]` (JSON) | forwarded `retrieval` block | inner |
| `retrieved_documents` | `Optional[str]` (JSON) | scored candidates | inner |
| `ranked_documents` | `Optional[str]` (JSON) | reranked + threshold-filtered passages | inner |
| `grounded_answer` | `Optional[str]` | rule-assembled answer body | inner |
| `citations` | `Optional[str]` (JSON) | `[{ref, id, title, source}]` | inner |
| `formatted_answer` | `Optional[str]` | final answer + sources + scope note + disclaimer | inner |
| `intake_notes` | `Optional[str]` (JSON) | validation / parse notes (no personal data) | inner |
| `error_code` | `Optional[str]` | reason marker for a correctable rejection; set once at the boundary that settled it, passed through unchanged afterwards, never surfaced in the invocation envelope | both |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

**State constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serialisable types).
- Structured fields (dict / list[dict]) stored as JSON STRINGS via `to_json()` /
  `from_json()` — used consistently by every producer AND consumer, so a
  checkpointed State stays msgpack-safe.
- `formatted_output` is NOT re-declared (the backbone field stays
  framework-owned).
- No tokens, keys, credentials, or raw personal identifiers in State.
- `InvocationContext` via `config["configurable"]` only, never in State.
- No Pydantic models, dataclasses, or arbitrary Python objects.

## Security design

### Trust

Every node declares `required_trust_level`. `PreProcessNode` requires
VERIFIED_EXTERNAL and declines empty / non-string / over-long `user_input`
before the domain workflow runs — a correctable rejection, so the run completes
carrying the reason rather than terminating. A caller below the required trust
level never reaches that check: the gate denies the call with `status = ERROR`
and `execute()` does not run. The standalone server elevates an
authenticated Bearer caller to VERIFIED_EXTERNAL via `INVOKE_AUTH_TOKEN`.

### The template owns its refusals

Instruction-override payloads — text addressed to the answering model rather
than a question addressed to the knowledge base — are refused by
`PreProcessNode` and again by `InputValidateNode`. A platform input gate may
refuse some of the same payloads first, but this template does not depend on
that: where such a gate is absent or configured off, an unchecked payload would
otherwise reach the answer path and come back as a success.

This is the terminating half of the rejection contract above: unlike a
malformed parameter, an instruction-override payload is not something the
caller rewords into an acceptable request, so the run ends with `status = ERROR`
and no reason marker, and nothing is carried forward. The refusal is therefore
observable as behaviour, and the tests prove it by calling `execute()` directly,
with no framework wrapper in front, asserting behaviour rather than any gate's
wording.

The rule is narrow on purpose and tested in both directions: attack forms are
refused, and ordinary questions that happen to use the same words ("what
instructions should I give the customer?") are unaffected.

### Direct customer identifiers

A returns or complaint question routinely carries a real customer's identifiers
— a card or order number read off a receipt, a callback number, an e-mail
address. None are needed to answer a policy question, so they are removed at
ingest and re-checked at the output boundary. The strip covers the question AND
the free-text field on the context channel: the framework's own input masking
covers the question channel only, so a sanitiser that ran on `user_input` alone
would leave the context route open.

The rule matches a run of digits and separators and then COUNTS the digits
(10–19 = card / account / order / phone), rather than matching a fixed 4-4-N
shape that would consume the first three groups of `4111 1111 1111 1111` and
leave the fourth behind as an unrecognisable fragment.

### The output invariant

The external answer is assembled from knowledge-base passages and their
citations only. It carries:

1. **no credential or secret pattern** — any hit withholds the whole answer;
2. **no direct customer identifier** — card/account/order number, telephone
   number, e-mail address;
3. **no verbatim embedding of caller-supplied text** — the question SELECTS
   passages, it is never rendered. Echoing it back would put caller-controlled
   free text on the external surface, where it could carry a fake citation
   marker or an instruction aimed at a downstream reader.

`OutputFormatNode` RENDERS the answer to satisfy this (and prints the scope note
that states it to the reader); `PostProcessNode` independently ENFORCES it, so a
rendering regression cannot quietly widen what the answer may contain.

**Layer order.** The pattern scans run before any layer that rewrites the text,
and the result is scanned AGAIN afterwards:

```
1. credential scan          → any hit withholds the whole answer (ERROR)
2. direct-identifier scan   → hits redacted, audited
3. verbatim caller-text     → embeddings of caller-derived state replaced
4. RE-SCAN of the sanitised bytes → a residual withholds the whole answer
```

Order matters for a scan-then-transform pipeline: a transform that rewrites
bytes can destroy the shape a later pattern would have matched, and a pattern
that only ever ran first would then miss it. The final re-scan makes the
invariant independent of what the middle layers did.

**No monetary precision grid applies to this template.** Some templates in this
family round monetary aggregates onto a published grid at the output boundary.
This one renders no computed monetary figures: the answer is policy prose plus
citations, and the bundled corpus contains no monetary amounts. A numeric
transform with nothing to transform could only mangle the structural numbers the
policy text depends on — "within 30 days", "5 to 7 business days", "kb-001",
"[1]". The absence is deliberate and is covered by a test, so it stays that way.

### Audit

Every node's `execute()` emits exactly ONE domain event
(`emit_trace_event("<node>_complete", {small non-personal payload}, state)`) on
the path where it does its work, plus a failure event on each refusal path. A
node that returns early because a reason marker was already settled emits
nothing: the boundary that settled the reason already recorded it, and a second
event per skipped node would report the same rejection five times. Nodes do NOT emit
`node_start` / `node_complete` / `node_error` — `BaseNode.__call__()` emits
those. Domain event names:

- `pre_process_complete`, `pre_process_validation_failed`
- `input_validate_complete`, `input_validate_failed`
- `retrieve_complete`
- `rerank_filter_complete`
- `generate_answer_complete`
- `output_format_complete`
- `post_process_complete`, `post_process_degraded`,
  `post_process_credential_violation`,
  `post_process_identifier_redaction`, `post_process_identifier_violation`,
  `post_process_blocked_field_redaction`

`post_process_degraded` records the run that completed carrying a reason marker
instead of an answer, so a declined request is as visible in the audit trail as
a terminated one — the caller-facing surface only shows a sentence.

No `_extra_security_gate_input` / `_extra_security_gate_output` instance methods
are defined on any node; the domain output gate is the module-level
`_security_gate_output()` called from inside `execute()`, and the agent class
exposes the same scanner as the canonical entry point.

## Advisory disclaimer

Every answer carries the standing advisory line (informational only;
store-specific promotions, regional regulations, and account-specific exceptions
may modify the policy; confirm with a supervisor or the official policy portal
before finalizing a customer-facing decision). It is appended by
`OutputFormatNode` as part of the domain output contract — post_process only
gates, it does not compose.

## Answer synthesis

The current build is deterministic: retrieval is keyword scoring over the
bundled corpus and `GenerateAnswerNode` assembles the answer from the ranked
passages (a static lead sentence plus cited passage excerpts). There is no model
call and no client dependency, which is why the manifest declares
`generation_mode: deterministic` and an empty `requires` block.

The `llm` block is forwarded through `_parent_config()` for forward
compatibility but is consumed by no node. The upgrade seam is documented in
`config/prompts/answer_synthesis_prompt.md`: a later `GenerateAnswerNode` swaps
the rule-based assembly for a model call over the same `ranked_documents` input
and emits the same `grounded_answer` / `citations` contract, so no other node
changes — and the output boundary keeps enforcing the same invariant over
whatever that node produces.

## Composition pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error propagation:** `propagate` — inner errors re-raised as `SubgraphError`.
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; the outer pre/post_process
  slots run at `TrustLevel.VERIFIED_EXTERNAL`.

## Import isolation

- [x] The template does not import the platform SDK.
- [x] Import targets: `framework/` and `shared/` only.
- [x] The agent class inherits the framework base class directly.

## Design decision record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base type | `AgentBaseGraph` | `AutonomousBaseGraph` | **`AgentBaseGraph`** | Fixed multi-step retrieval workflow, no autonomous loop |
| Composition | Standalone slots | `GraphNode` → inner `BaseGraph` | **`GraphNode` → inner `BaseGraph`** | A five-step domain workflow exceeds a single `main` node; nesting keeps the outer backbone untouched |
| Answer synthesis | Rule-based assembly | Model call | **Rule-based** | Deterministic and testable; a model call swaps in at the documented seam without a contract change |
| Corpus storage | External vector store | Bundled JSON corpus | **Bundled JSON corpus** | Self-contained and deterministic; the retrieval contract (`retrieved_documents` JSON) is store-agnostic for a later upgrade |
| Invalid caller parameter | Clamp to the nearest valid value | Reject the request | **Reject** | A clamped parameter changes the answer without telling the caller |
| Reporting a correctable rejection | Terminate with an error status | Complete carrying a reason marker and a caller-facing sentence | **Complete carrying the reason** | Terminating ends the calling surface's turn and leaves the reason only in the audit trail; completing lets the caller fix the request and retry on the same conversation |
| Reporting an owned refusal (instruction-override, output-gate hit) | Complete with an explanatory sentence | Terminate with an error status | **Terminate** | These are not requests a caller rewords into acceptance; completing would present a refusal as an outcome and invite a retry that must also fail |
| Surfacing the reason marker | Return `error_code` in the invocation envelope | Message text only | **Message text only** | An exported code becomes an external contract to keep stable; the marker stays internal and field paths stay in `error_log` |
| Question in the answer body | Echo it back for context | Static lead sentence | **Static lead sentence** | Echoing puts caller-controlled free text on the external surface |
