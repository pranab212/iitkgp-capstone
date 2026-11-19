# Supply Chain Disruption Monitor

An agentic AI system that monitors news articles for supply chain disruptions, identifies affected suppliers across a multi-tier network, scores risk, and recommends actions — with a human-in-the-loop review and AI-assisted feedback cycle.

Based on: *AlMahri, Xu & Brintrup — "Automating Supply Chain Disruption Monitoring via an Agentic AI Approach"*

## What It Does

Give it a **news article** and a **grocer**. The system:

1. **Reads the article** (LLM) — extracts disruption type, affected countries, industries, and companies
2. **Queries a knowledge graph** — finds which suppliers (Tier-1 through Tier-4) in the grocer's network are hit
3. **Scores risk** — ranks exposed Tier-1 suppliers using a weighted formula (breadth, dependency, criticality, centrality, depth)
4. **Visualises the network** — generates an interactive tiered diagram of the disrupted supply chain
5. **Writes an executive summary** (LLM) — with key recommendations per supplier
6. **AI self-critique** — the AI reviews its own extraction and flags potential mistakes before human review
7. **Pauses for human review** — the reviewer can approve, edit, send feedback for re-analysis, or discard
8. **Finds replacement suppliers** — for any HIGH-risk supplier, identifies alternatives from undisrupted regions

Two steps use an LLM (extraction + summary). Everything else — supplier matching, risk scores, actions, alternatives — is deterministic Python, so results are repeatable.

## Pipeline

```
[RAG retrieval] → Monitoring (LLM) → Knowledge Graph Query → Risk Scoring
                       ↑                                          ↓
                       │                              Network Visualisation
                       │                                          ↓
                       │                                CSCO Summary (LLM)
                       │                                          ↓
                       │                                  AI Self-Critique
                       │                                          ↓
                       └── feedback ──── Human Review ──→ Alternative Sourcing
                                            │
                                      edit extraction → KG (skips LLM)
```

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

# Or use the CLI
python cli.py run --text "War in Ukraine halts grain exports." --auto approve
python cli.py demo                   # tools-only demo, no LLM needed
```

## Dashboard

The web dashboard at `http://localhost:8080` provides:

- **Scenario picker** — pre-built articles (Ukraine conflict, plant fire, port strike, false positive) or paste your own
- **Live pipeline timeline** — each step appears as it finishes with timing and token usage
- **Bento grid KPIs** — disrupted supplier count, tier breakdown, risk level distribution
- **Network graph** — interactive tiered diagram with color-coded risk levels
- **Human review panel** — editable summary, action dropdowns, AI suggestions, feedback box
- **History view** — when re-analysis is used, shows diff badges comparing analysis rounds
- **Analytics overlay** — retry rate, feedback history, outcome distribution

## Human Review Options

| Action | What Happens |
|---|---|
| **Approve** | Accept analysis → find replacement suppliers |
| **Approve with Edits** | Modify summary/actions → find replacements |
| **Send Feedback & Re-analyse** | AI re-runs extraction using reviewer's corrections (up to 3 rounds) |
| **Edit Extraction Directly** | Fix countries/industries/companies → re-run downstream only (no LLM) |
| **Discard Analysis** | Stop — analysis is discarded |

See [AI_FEEDBACK_LOOP.md](AI_FEEDBACK_LOOP.md) for details on the feedback features.

## LLM Support

The system picks the first available, in order:

| LLM | When |
|---|---|
| **Qwen3-8B** | `QWEN_API_KEY` is set (course API with CA certificate) |
| **Azure OpenAI** | `AZURE_OPENAI_*` variables are set |
| **Offline Mock** | No keys — keyword-based, good for testing the flow |

The dashboard header shows which LLM is active.

## RAG (Optional)

Retrieval-Augmented Generation for the Monitoring step, off by default:

```bash
RAG=1 python app.py                  # both parts
RAG_EXAMPLES=1 python app.py         # similar solved examples only
GROUND_ENTITIES=1 python app.py      # company name grounding only
```

- **Similar examples**: BM25 search picks 3 labeled articles as few-shot examples
- **Entity grounding**: Companies named in the article are looked up in the knowledge graph and listed in the prompt

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
dashboard.html          Dashboard UI
cli.py                  CLI: run, eval, demo, data
scpoc/
  pipeline.py           LangGraph pipeline with human-in-the-loop
  schemas.py            Pydantic models (State, MonitoringOutput, CSCOOutput, SelfCritiqueOutput)
  prompts.py            All LLM prompts
  llm.py                Qwen3-8B, Azure OpenAI, MockLLM
  tools.py              Deterministic tools (KG query, risk scoring, alternatives, visualisation)
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

## Risk Score Formula

```
score = 0.35 × breadth + 0.25 × dependency + 0.20 × criticality
      + 0.10 × centrality + 0.10 × depth

HIGH ≥ 0.60 → Replace supplier
MEDIUM 0.45–0.59 → Increase monitoring  
LOW < 0.45 → Maintain standard operations
```

## Configuration

See [README.txt](README.txt) for detailed setup instructions including virtual environment setup, Qwen API configuration, Azure OpenAI, RAG options, and troubleshooting.
