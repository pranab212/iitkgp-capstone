# Supply Chain Disruption Monitor

An agentic AI system that monitors news articles for supply chain disruptions, identifies affected suppliers across a multi-tier network, scores risk with human-readable reasoning, and recommends actions — with a human-in-the-loop review, AI self-critique, and RAG-enhanced analysis.

Based on: *AlMahri, Xu & Brintrup — "Automating Supply Chain Disruption Monitoring via an Agentic AI Approach"*

## What It Does

Give it a **news article** and a **grocer**. The system:

1. **RAG Retrieval** (optional) — BM25 search finds 3 similar solved articles + entity grounding verifies company names against the knowledge graph
2. **Monitoring Agent** (LLM + RAG) — extracts disruption type, affected countries, industries, companies using Chain of Thought reasoning, enhanced by RAG examples and grounded entities
3. **Knowledge Graph Query** — BFS traversal finds which suppliers (Tier-1 → Tier-4) in the grocer's network are hit
4. **Risk Scoring with Reasoning** — ranks exposed Tier-1 suppliers using a weighted 5-factor formula and generates human-readable explanations for each score
5. **Network Visualisation** — interactive tiered diagram with connection reasoning (hover any node/edge for details on why it's connected and why it's flagged)
6. **CSCO Summary** (LLM + RAG) — executive summary enhanced by RAG context from similar past disruptions and verified entities
7. **AI Self-Critique** — reviews its own extraction and flags potential mistakes before human review
8. **Human Review** — approve, edit, send feedback for re-analysis, or discard
9. **Alternative Sourcing** — for HIGH-risk suppliers, identifies replacement candidates from undisrupted regions

Three steps use an LLM (extraction + summary + self-critique). Everything else — supplier matching, risk scores, actions, alternatives — is deterministic Python.

## Pipeline

```
                        ┌──── feedback ───────────────────────────────────────┐
                        │                                                     │
                        ▼                                                     │
[RAG Retrieval] → Monitoring (LLM+RAG) → KG Query → Risk Scoring → Network Viz
                                            ▲              │
                                            │              ▼
                                            │     CSCO Summary (LLM+RAG)
                                            │              │
                                            │       AI Self-Critique
                                            │              │
                                            └── edit ── Human Review ──→ Alternatives → Done
                                                           │
                                                     discard → END
```

**RAG flows through two LLM steps:**
- **Monitoring Agent** — gets similar solved examples + grounded entity records
- **CSCO Summary** — gets the same RAG context to write more informed executive summaries

## Quick Start

```bash
# Clone and setup
git clone <repo-url> && cd iitkgp-capstone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure LLM (optional — without this, uses offline mock)
cp .env.example .env
# Fill in QWEN_API_KEY, QWEN_API_BASE, QWEN_CA_CERT

# Run the dashboard
python app.py                        # http://localhost:8080

# Run with RAG enabled
RAG=1 python app.py                  # enables BM25 examples + entity grounding

# Or use the CLI
python cli.py run --text "War in Ukraine halts grain exports." --auto approve
python cli.py demo                   # tools-only demo, no LLM needed
```

## Dashboard

The web dashboard at `http://localhost:8080` provides:

- **Scenario picker** — pre-built articles (Ukraine conflict, plant fire, port strike, false positive) or paste your own
- **Live pipeline timeline** — each step appears as it finishes with timing and token usage
- **Bento grid KPIs** — disrupted supplier count, tier breakdown, risk level distribution
- **Risk table with reasoning** — each supplier shows WHY it scored as it did (e.g., "directly disrupted; 80% of inputs disrupted")
- **Network graph with connection reasoning** — hover nodes for risk reasoning, hover edges for product flow + disruption status
- **RAG-enhanced CSCO badge** — shows when the executive summary was written with RAG context
- **Human review panel** — editable summary, action dropdowns, AI suggestions, feedback box, edit extraction
- **History view** — when re-analysis is used, shows diff badges comparing analysis rounds
- **Analytics overlay** — retry rate, feedback history, outcome distribution
- **KG Impact panels** — live baseline-vs-grounding comparison metrics

## Human Review Options

| Action | What Happens |
|---|---|
| **Approve** | Accept analysis → find replacement suppliers |
| **Approve with Edits** | Modify summary/actions → find replacements |
| **Send Feedback & Re-analyse** | AI re-runs extraction using reviewer's corrections (up to 3 rounds) |
| **Edit Extraction Directly** | Fix countries/industries/companies → re-run downstream only (no LLM) |
| **Discard Analysis** | Stop — analysis is discarded |

See [AI_FEEDBACK_LOOP.md](AI_FEEDBACK_LOOP.md) for details on the feedback features.

## RAG Integration

Retrieval-Augmented Generation enhances **both** LLM steps:

```bash
RAG=1 python app.py                  # both parts enabled
RAG_EXAMPLES=1 python app.py         # similar solved examples only
GROUND_ENTITIES=1 python app.py      # company name grounding only
```

### How RAG works in each step:

| Step | RAG provides | Why it helps |
|---|---|---|
| **Monitoring Agent** | 3 similar solved articles (BM25) + grounded company records | Prevents over-generalization (one company ≠ whole industry), buyer/producer confusion |
| **CSCO Summary** | Same RAG context | Executive summary references verified entities and past patterns from similar disruptions |

- **BM25 keyword search** — 37 labeled exemplar articles, pure Python, no vector DB
- **Entity grounding** — company names matched against 2,900 nodes in the knowledge graph
- Adds ~500 tokens to prompts; deterministic and reproducible

## Risk Scoring with Reasoning

Each Tier-1 supplier gets a risk score + human-readable explanation:

```
score = 0.35 × breadth + 0.25 × dependency + 0.20 × criticality
      + 0.10 × centrality + 0.10 × depth

HIGH ≥ 0.60 → Replace supplier
MEDIUM 0.45–0.59 → Increase monitoring  
LOW < 0.45 → Maintain standard operations
```

**Example reasoning output:**
> Sakri Packaging 020: HIGH (0.613) — directly disrupted (is itself in the affected region/industry); high criticality — a critical node in the network (PageRank/degree); high centrality among Tier-1 suppliers

The network diagram also embeds this reasoning — hover any node to see why it's flagged, hover any edge to see what product flows through that connection and whether the link is at risk.

## LLM Support

The system picks the first available, in order:

| LLM | When |
|---|---|
| **Qwen3-8B** | `QWEN_API_KEY` is set (course API with CA certificate) |
| **Azure OpenAI** | `AZURE_OPENAI_*` variables are set |
| **Offline Mock** | No keys — keyword-based, good for testing the flow |

## Evaluation

20 hand-written articles with expected answers (15 disruptions + 5 false positives):

```bash
python cli.py eval                                 # full eval with real LLM
python cli.py eval --mock                          # quick check with mock
python cli.py eval --with-retry --max-retries 2    # measure feedback loop improvement
python cli.py eval --rag                           # eval with RAG enabled

# Compare configurations side by side
python cli.py eval-compare outputs/eval_results.json outputs/eval_results_rag-examples.json
```

Reports extraction precision/recall/F1, network query matches, risk scoring accuracy, false positive handling, time, and token cost.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v                            # 31 tests, no API key needed
```

| Suite | Tests | Covers |
|---|---|---|
| `test_pipeline.py` | 12 | Approve, reject, edit, retry, max retries, self-critique, partial edit, early exit |
| `test_rag.py` | 12 | Tokenisation, BM25 ranking, corpus validity, entity grounding, prompt formatting |
| `test_tools.py` | 7 | Entity resolution, BFS, risk scoring, alternatives, network diagram |

## Project Structure

```
app.py                  Web server + JSON API
dashboard.html          Dashboard UI (dark theme)
cli.py                  CLI: run, eval, demo, data
architecture.html       Interactive pipeline architecture diagram
AI_FEEDBACK_LOOP.md     Feature documentation for the feedback loop
scpoc/
  pipeline.py           LangGraph pipeline with human-in-the-loop + RAG in both LLM steps
  schemas.py            Pydantic models (State, MonitoringOutput, CSCOOutput, SelfCritiqueOutput)
  prompts.py            All LLM prompts (monitoring, CSCO, self-critique, RAG context)
  llm.py                Qwen3-8B, Azure OpenAI, MockLLM
  tools.py              Deterministic tools (KG query, risk scoring with reasoning, alternatives, visualisation)
  rag.py                BM25 exemplar search + entity grounding
  evaluate.py           Evaluation framework with retry metrics
  graph.py              NetworkX graph loading
  evalset.py            20 evaluation articles
  datagen.py            Dummy network generator
data/
  companies.csv         ~2,900 dummy companies across 3 grocers
  supplies.csv          Supply relationships
  eval_articles.json    Evaluation set with ground truth
  rag/exemplars.json    37 labeled examples for RAG
tests/                  31 pytest tests
```

## Configuration

See [README.txt](README.txt) for detailed setup instructions including virtual environment setup, Qwen API configuration, Azure OpenAI, RAG options, and troubleshooting.
