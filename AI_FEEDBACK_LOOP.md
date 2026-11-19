# AI-Assisted Human Review: Closing the Feedback Loop

## The Problem

The base pipeline has a one-shot human review: approve, edit, or reject. If the reviewer rejects, the analysis is discarded — the AI learns nothing from the correction and the reviewer starts over from scratch.

- A small LLM mistake (e.g., listing a buyer country as disrupted) forces a full rejection
- The reviewer's domain knowledge is lost — they see the problem but can't tell the AI what to fix
- There's no way to iteratively refine the analysis

## The Solution

A **human-AI feedback loop** where the reviewer's corrections guide the AI to produce a better result.

## Features

### 1. Send Feedback & AI Re-analysis

The reviewer writes what's wrong (e.g., *"Japan is only a buyer, not a disrupted country. Remove it."*). The AI receives this as context in its prompt and re-analyses the article from scratch, correcting the flagged issues.

- Up to 3 rounds of feedback before auto-rejection
- History view shows each round with the reviewer's feedback quoted
- Diff badges highlight what changed between rounds (added / removed / unchanged)

**How it's built:** A LangGraph graph cycle — `human_review` routes back to `monitoring` with the feedback injected via `MONITORING_RETRY_CONTEXT` in the prompt. State tracks `retry_count`, `retry_feedback`, and `previous_monitoring`.

### 2. AI Self-Critique

Before the reviewer sees the analysis, the AI **reviews its own extraction** and flags potential issues:

- Buyer countries listed as disrupted
- Broad industry labels applied to single-company events
- Missing entities

Suggestions appear as clickable items in the review panel — click one to auto-fill the feedback box. Structured output: `SelfCritiqueOutput` with `suggestions` (list) and `confidence` (high/medium/low).

**How it's built:** A new LLM node (`self_critique`) wired between `csco` and `human_review` in the graph.

### 3. Edit Extraction Directly (Partial Retry)

For fine-grained fixes: the reviewer edits countries, industries, or companies directly. This **skips the LLM entirely** and re-runs only the downstream steps (KG → Risk → Visualization → Summary → Self-critique → Review).

**How it's built:** A new `edit_extraction` decision that routes to `kg_query` instead of `monitoring`, updating the `MonitoringOutput` in-place.

### 4. Quantitative Evaluation

Measures whether feedback actually improves extraction:

- Simulates human feedback from ground truth: compares extraction to expected answer and generates specific corrections (e.g., *"Remove countries: Germany — not actually disrupted"*)
- Reports: articles retried, fixed by retry, still wrong after retry, mean retries per article

```bash
python cli.py eval --with-retry --max-retries 2
```

### 5. Retry Analytics

Tracks feedback patterns across all analysis jobs:

- Retry rate, avg retries, max retries used
- Self-critique usage, direct edit count
- Outcome distribution, recent feedback log

Accessible via the **📊 Analytics** button in the dashboard header, backed by `/api/analytics`.

## Updated Pipeline Flow

```
monitoring → KG → risk → viz → CSCO → self_critique → human_review
     ↑                                       │               │
     └────── feedback + re-analyse ──────────┘               │
                                                              │
                                           edit_extraction → KG (skips LLM)
```

## Review Actions

| Action | What happens |
|---|---|
| **Approve** | Accept as-is → alternative sourcing runs |
| **Approve with Edits** | Edit summary/actions → alternative sourcing |
| **Send Feedback & Re-analyse** | AI re-runs with reviewer's corrections |
| **Edit Extraction Directly** | Fix entities manually → re-run downstream only |
| **Discard Analysis** | Hard stop |

## Running It

```bash
python app.py                                      # dashboard at http://localhost:8080
python cli.py eval --with-retry --max-retries 2    # evaluation with feedback loop
pytest -v                                          # 31 tests (4 for feedback features)
```

## Tests

| Test | Verifies |
|---|---|
| `test_retry_reruns_monitoring_then_approves` | Full feedback loop works |
| `test_retry_max_retries_reached` | Max retry limit enforced |
| `test_self_critique_runs_before_review` | Self-critique runs and produces suggestions |
| `test_edit_extraction_skips_llm` | Partial edit skips LLM, re-runs KG onwards |
