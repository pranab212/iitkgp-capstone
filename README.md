# Supply Chain Disruption Monitor — Human-AI Feedback Loop

> Agentic POC extending [AlMahri, Xu & Brintrup (2024)](https://arxiv.org/abs/2401.12345) with a **human-in-the-loop feedback cycle**, **AI self-critique**, and **partial retry** capabilities.

## What's New (My Contributions)

The base pipeline had a one-shot human review: approve, edit, or reject.  
**Reject was a dead end** — the analysis was discarded with no way for the AI to learn from the reviewer's corrections.

I added five features that close this gap:

### 1. Human Feedback → AI Re-analysis Loop

When a reviewer isn't satisfied with the LLM's extraction, they can send specific feedback (e.g., *"Japan is only a buyer, not a disrupted country. Remove it."*) and the AI **re-analyses the article from scratch**, incorporating those corrections.

**Implementation:** A LangGraph graph cycle — `human_review` routes back to `monitoring` with the reviewer's feedback injected into the LLM prompt via `MONITORING_RETRY_CONTEXT`. Configurable max retries (default 3) prevent infinite loops.

```
monitoring → KG → risk → viz → CSCO → self_critique → human_review
     ↑                                                      │
     └──────────── retry (with feedback) ───────────────────┘
```

**Files:** `pipeline.py` (graph cycle + routing), `schemas.py` (retry state fields), `prompts.py` (`MONITORING_RETRY_CONTEXT`)

### 2. AI Self-Critique

A new LLM step runs **automatically before human review**. The AI examines its own extraction and flags potential mistakes — common errors like listing buyer countries as disrupted, or applying broad industry labels to single-company events.

- Outputs structured `SelfCritiqueOutput` (suggestions + confidence level)
- Suggestions appear in the review panel as **clickable items** — click one to auto-fill the feedback box
- Works with both real LLM (Qwen3-8B) and the offline mock

**Files:** `schemas.py` (`SelfCritiqueOutput`), `prompts.py` (`SELF_CRITIQUE_SYSTEM/USER`), `pipeline.py` (new node wired between `csco` and `human_review`), `llm.py` (MockLLM handler)

### 3. Edit Extraction Directly (Partial Retry)

Sometimes you don't need the LLM to redo everything — you just want to fix one wrong country or company. The reviewer can **directly edit** the extracted countries, industries, or companies and re-run only the downstream steps.

- Routes to `kg_query` instead of `monitoring` — **skips the LLM entirely**
- Only re-runs: KG query → Risk → Visualization → CSCO → Self-critique → Review
- Much faster than a full re-analysis

**Files:** `pipeline.py` (new `edit_extraction` routing to `kg_query`), `schemas.py` (new approval literal), `app.py` (API support)

### 4. Quantitative Evaluation of the Feedback Loop

Extended the evaluation framework to **measure whether re-analysis actually helps**:

- `_generate_feedback()` compares extraction vs. ground truth and generates specific feedback (simulating a human reviewer)
- `run_one_with_retry()` runs the article, checks extraction, retries with simulated feedback
- Reports: articles retried, fixed by retry, still wrong after retry, mean retries per article

```bash
python cli.py eval --with-retry --max-retries 2         # full eval with retry
python cli.py eval --mock --with-retry                   # quick plumbing check
```

**Files:** `evaluate.py` (new functions + retry metrics), `cli.py` (new CLI args)

### 5. Retry Analytics

Tracks retry patterns across all jobs via `/api/analytics`:

- Total jobs, retry rate, avg retries, max retries used
- Edit-extraction count, self-critique usage
- Outcome distribution (approved/edited/rejected/no_exposure)
- Recent feedback log — what reviewers told the AI to fix

Accessible via the **📊 Analytics** button in the dashboard header.

**Files:** `app.py` (new endpoint), `dashboard.html` (analytics overlay)

---

## Updated Pipeline Flow

```
[retrieve] → monitoring (LLM) → kg_query → [no_exposure? → END]
                 ↓
             risk → visualizer → csco (LLM) → self_critique (LLM)
                 ↓
            human_review (interrupt)
             ↓        ↓        ↓              ↓                ↓
          approve    edit    discard    feedback+retry    edit_extraction
             ↓        ↓        ↓              ↓                ↓
        alternatives  alt.    END      → monitoring        kg_query
                                        (graph cycle)    (skip LLM)
```

## Review Actions

| Action | What happens |
|---|---|
| **Approve** | Accept as-is → alternative sourcing runs |
| **Approve with Edits** | Edit summary/actions → alternative sourcing runs |
| **Send Feedback & Re-analyse** | AI re-runs monitoring with your corrections |
| **Edit Extraction Directly** | Fix countries/industries/companies → re-run KG onwards (no LLM) |
| **Discard Analysis** | Hard stop — analysis is thrown away |

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure LLM (copy and fill in your API key)
cp .env.example .env

# Run dashboard
python app.py                    # http://localhost:8080

# Run tests (no API key needed)
pip install -r requirements-dev.txt
pytest -v                        # 31 tests

# Evaluation
python cli.py eval --mock                          # quick check with mock LLM
python cli.py eval                                 # full eval with real LLM
python cli.py eval --with-retry --max-retries 2    # eval with feedback loop
python cli.py eval --rag --with-retry              # eval with RAG + retry
```

## Tests

31 tests covering all flows:

| Test | What it verifies |
|---|---|
| `test_retry_reruns_monitoring_then_approves` | Feedback loop works end-to-end |
| `test_retry_max_retries_reached` | Max retry limit enforced |
| `test_self_critique_runs_before_review` | Self-critique produces suggestions between CSCO and review |
| `test_edit_extraction_skips_llm` | Partial edit skips monitoring, re-runs KG onwards |
| + 27 original tests | Approve, reject, edit, no exposure, alternatives, RAG, tools |

## File Structure

```
app.py                  Web server + JSON API (loads dashboard.html)
dashboard.html          Production UI (dark theme, glassmorphism)
cli.py                  CLI: run, eval, demo, data
scpoc/
  pipeline.py           LangGraph pipeline (graph cycle for retry)
  schemas.py            State, MonitoringOutput, CSCOOutput, SelfCritiqueOutput
  prompts.py            All LLM prompts (including retry context, self-critique)
  llm.py                Qwen3-8B, Azure OpenAI, MockLLM
  tools.py              Deterministic tools (KG, risk, alternatives, network viz)
  evaluate.py           Evaluation framework (with retry metrics)
  rag.py                BM25 exemplar search + entity grounding
tests/
  test_pipeline.py      12 pipeline tests (4 new)
  test_rag.py           12 RAG tests
  test_tools.py         7 tools tests
```

## Configuration

See `README.txt` for detailed setup instructions (virtual environment, Qwen API configuration, Azure OpenAI, RAG options, troubleshooting).

The `vm05-ca.crt` file is the **public** CA certificate for the course Qwen API server (not a private key) — safe to keep in the repo.
