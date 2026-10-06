# Answer Synthesis Prompt — RET-C2-006 (upgrade seam)

> **The current build does NOT use this prompt at runtime.**
> `GenerateAnswerNode` assembles answers deterministically from
> `ranked_documents`; no node reads this file. It documents the synthesis
> contract for the model-driven upgrade described in `docs/02_design.md`
> ("Answer synthesis"), so the swap changes only the inside of
> `GenerateAnswerNode.execute()`.

## Contract (a model-driven GenerateAnswerNode)

- **Input:** the same `ranked_documents` JSON (id / title / category / source /
  score / excerpt) the current node reads. The caller's question is available
  as `search_query` for retrieval and prompting, but must NOT be echoed into
  the answer body — the output boundary redacts verbatim caller text.
- **Output:** the same state contract — `grounded_answer` (str, with numbered
  `[n]` citation markers) and `citations` (JSON list of
  `{ref, id, title, source}`). `PostProcessNode` enforces the output
  invariant over whatever this node produces, so a model-generated answer is
  held to the same rules as the rule-assembled one.
- **Grounding rule:** every factual statement in the answer must be traceable
  to one of the supplied passages via a `[n]` marker; content not present in
  the passages must not be asserted (never invent a store policy exception).
- **No-coverage rule:** when no passage supports the question, say so and
  recommend refining the query or escalating to a store supervisor — never
  answer from parametric knowledge.
- **Tone:** neutral, service-appropriate, no promises of an outcome the
  policy does not support (the advisory disclaimer is appended downstream by
  `OutputFormatNode`).

## Prompt template

```
You answer customer returns-and-complaint questions strictly from the
returns/complaint-resolution knowledge-base passages provided below.

Question:
{search_query}

Passages (each with a reference number):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   knowledge base has insufficient coverage and recommend escalating to a
   store supervisor.
2. Mark every factual statement with the [n] reference of its passage.
3. Do not promise a refund, exchange, or exception amount that the passages
   do not support.
4. Keep the answer under 300 words.
```

## Configuration coupling

The `llm` block in `config/config.yaml` (`temperature`, `max_tokens`) is
already validated and forwarded to the inner graph via
`ReturnsComplaintSearchGraphNode._parent_config()` under
`config["configurable"]["llm"]`; a model-driven node reads it from there.
Adopting it also means declaring the client extra and any required secret in
`config/agent.yaml`'s `requires` block and switching `generation_mode` to
`llm` — the current build declares neither because it needs neither.
