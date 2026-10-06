# Returns & Complaint Resolution Q&A Agent

AI agent for answering customer returns and complaint resolution questions, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Retail
> **Template ID**: RET-C2-006

## Overview

Answers natural-language questions about retail returns, exchanges and complaint
resolution — whether an item is still inside its return window, how a refund is
issued, what a receipt-less return requires, when a complaint should be escalated —
with a retrieval pipeline over a returns-and-complaints policy knowledge base.
Each question is validated and normalised, matching passages are retrieved,
reranked and filtered by a relevance threshold, and the answer is assembled
**only** from passages that clear that threshold, with numbered `[n]` citations
and a standing advisory note. When no passage is relevant enough, the agent says
the knowledge base has insufficient coverage instead of inventing policy.

The agent answers policy questions; it does not approve or deny a return, call an
order-management or refund system, or decide an individual case — those stay with
the system of record and the person handling the customer.

Two properties are enforced rather than assumed. Direct customer identifiers
(order and card numbers, telephone numbers, e-mail addresses) that a real question
carries are stripped on the way in and re-checked on the way out, so they never
reach a stored answer. And the question itself is never quoted back into the
answer: it selects passages, it is not rendered.

The bundled knowledge base is a small sample of returns-policy material so the
pipeline runs and tests end-to-end out of the box; a real deployment replaces it
with its own policy corpus behind the same node contract.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent
fails at graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Invoking

`POST /invoke` takes the question as `input` and optional structured parameters
as `input_context`:

```json
{
  "input": "What is the return window for a defective item?",
  "input_context": { "category": "defective_items", "top_k": 3, "channel": "web" }
}
```

`category` filters the knowledge base, `top_k` (1–20) narrows retrieval depth, and
`channel` records where the question came from. Every field is validated against
explicit bounds; an out-of-range or malformed value rejects the request rather than
being quietly adjusted.

## Project Structure

```
src/          agent implementation (nodes, graph, schemas)
tests/        unit and boundary tests
config/       agent manifest, runtime parameters, and the sample knowledge base
docs/         design and test documentation
```

See `docs/` for the design specification and test specification.

## Customising

1. Adjust `config/config.yaml` for your own retrieval settings.
2. Replace the sample knowledge base (`config/kb/`) with your own policy corpus.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
