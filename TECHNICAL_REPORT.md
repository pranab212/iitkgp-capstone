# Supply Chain Disruption Monitor — Technical Report

**Executive Post Graduate Programme in Generative AI and Agentic AI**
**Track B — API-Side Capstone: Multi-Agent Operations Copilot**

**Date:** November 2026
**Team:** Krishika Singh, Kuldeep, Rohan Dawar, Pranab, and others

---

## Table of Contents

1. [Overview](#1-overview)
2. [Task Decomposition & Scope](#2-task-decomposition--scope)
3. [Architecture](#3-architecture)
4. [Data Model](#4-data-model)
5. [Prompt Engineering](#5-prompt-engineering)
6. [RAG — Retrieval-Augmented Generation](#6-rag--retrieval-augmented-generation)
7. [Agentic Layer](#7-agentic-layer)
8. [Core Algorithms](#8-core-algorithms)
9. [Human-AI Feedback Loop](#9-human-ai-feedback-loop)
10. [Evaluation & Observability](#10-evaluation--observability)
11. [Deployment & Interface](#11-deployment--interface)
12. [Responsible AI](#12-responsible-ai)
13. [Limitations & Future Work](#13-limitations--future-work)

---

## 1. Overview

The Supply Chain Disruption Monitor is a multi-agent operations copilot that takes a news article and a grocer, then determines which of the grocer's suppliers — from Tier-1 to Tier-4 — are affected by the reported event. It produces a risk score with human-readable reasoning for each exposed Tier-1 supplier, a recommended action, and replacement candidates for HIGH-risk ones. A human reviewer approves, corrects, or refines the analysis before any downstream action is taken.

The system implements AlMahri, Xu & Brintrup, *"Automating Supply Chain Disruption Monitoring via an Agentic AI Approach"*, as an IIT Kharagpur capstone. It runs on a synthetic network of **2,953 companies** and **4,438 supply links** across three fictional grocers.

### Key Design Principle

A strict split between **language understanding** and **calculation**. Only three steps call an LLM: extraction (Monitoring Agent), the executive summary (CSCO), and a self-critique. Name matching, the graph search, risk scores, actions, the network diagram, and alternative sourcing are **deterministic Python**. For the same extracted criteria the tools always return the same answer, so any error traces back to the extraction.

### What the Small Model Breaks and How We Compensate

Since the system uses a ≤8B model (Qwen3-8B), the following failure modes were identified and addressed:

| Small-Model Failure | Compensation Strategy |
|---|---|
| Lists buyer countries as disrupted (e.g., Germany is a buyer, not a disruption source) | Chain of Thought prompting + few-shot examples that explicitly show buyer/producer distinction |
| Over-generalises: names one company but lists the entire industry | Entity grounding from the knowledge graph tells the LLM the exact company identity |
| Hallucinates company names not in the article | Pydantic strict JSON validation rejects malformed output; entity resolution checks names against the graph |
| Produces malformed JSON | Schema appended to system prompt; `<think>` tag stripping; one automatic repair retry |
| Inconsistent extraction on retries | Temperature fixed at 0; CoT forces reasoning before extraction |
| Misses sector-wide disruptions phrased without the industry keyword | RAG BM25 retrieval surfaces similar solved examples that demonstrate the pattern |

### Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Orchestration | LangGraph (StateGraph) | Pipeline as a directed graph with cycles, interrupts, persistent state |
| LLM | Qwen3-8B (course API, TLS with CA cert) | Extraction, summary, self-critique (3 LLM nodes) |
| LLM Fallbacks | Azure OpenAI, MockLLM | Alternative backends; mock for offline testing |
| Graph | NetworkX + SciPy | Supply chain knowledge graph, PageRank, BFS |
| Data Validation | Pydantic v2 | Strict JSON schema enforcement on all LLM outputs |
| RAG | BM25 (pure Python, Okapi) | Exemplar retrieval; no vector database or embeddings API |
| State Persistence | MemorySaver (LangGraph) | Checkpointing for interrupt/resume across human review |
| Server | Python `http.server` (stdlib) | Zero-dependency web server and JSON API |
| Frontend | Single-file HTML + vanilla JS | Dark-themed dashboard, no build step |

---

## 2. Task Decomposition & Scope

### What the Copilot Does

Given a news article and a grocer identity:
1. **Classify** whether the article reports a real supply chain disruption
2. **Extract** the disruption type, affected countries, industries, and companies
3. **Map** the extraction to the grocer's supplier network (Tier-1 through Tier-4)
4. **Score risk** for each exposed Tier-1 supplier with a 5-factor weighted formula
5. **Generate reasoning** explaining why each supplier is flagged
6. **Visualise** the disrupted network as an interactive diagram with connection reasoning
7. **Summarise** the impact for the Chief Supply Chain Officer (RAG-enhanced)
8. **Self-critique** the extraction and flag potential errors
9. **Pause for human review** with 5 possible decisions
10. **Find alternative suppliers** for HIGH-risk ones from undisrupted regions

### What It Must Refuse

- Articles that are not supply chain disruptions (opinion pieces, forecasts, routine business news) must be classified as `is_disruption: false` and the pipeline must exit at "No Exposure" without scoring or recommending actions
- The system must never invent suppliers, risk scores, or actions not grounded in the knowledge graph
- Alternative suppliers must not come from the disrupted area

### Evaluation Set (Written Before Code)

20 hand-written articles with gold answers, written to cover the main failure modes:
- 5 country-level disruptions (Turkey, Ukraine, Vietnam, Egypt, Iceland)
- 4 single-company events (at Tier-1 through Tier-4)
- 3 industry-wide disruptions
- 3 country-and-industry combinations
- 5 false positives that must end with "no exposure"

Each article has expected extraction (countries, industries, companies) and expected pipeline outputs (disrupted count, top supplier, top HIGH supplier) computed deterministically from the gold extraction using the same tools the pipeline uses.

---

## 3. Architecture

### Pipeline Graph

The pipeline is a LangGraph `StateGraph` built in `scpoc/pipeline.py` by `build_app()`. Each node reads one shared Pydantic `State` and returns a partial update. Two conditional edges decide the route: one after the graph query, one after human review.

```
                        ┌──── feedback ───────────────────────────────────────┐
                        │                                                     │
                        ▼                                                     │
[RAG Retrieval] → Monitoring (LLM+RAG) → KG Query → Risk Scoring → Network Viz
                                            ▲              │
                                            │              ▼
                                            │     CSCO Summary (LLM+RAG)
                                            │              │
                                            │       AI Self-Critique (LLM)
                                            │              │
                                            └── edit ── Human Review ──→ Alternatives → Done
                                                           │
                                                     discard → END
```

**10 nodes** (9 without optional retrieval), **2 conditional decision points**.

### Routing Rules

1. **After `kg_query`**: go to `risk` only if `bfs.disrupted_count > 0` AND the extraction agrees with itself (`is_disruption` is true and `disruption_type` is not `none`). Otherwise go to `no_exposure` → END.

2. **After `human_review`**:
   - `rejected` → END
   - `retry` → loops back to `monitoring` (LLM re-runs with feedback)
   - `edit_extraction` → loops back to `kg_query` (no LLM call, skips directly to graph query)
   - `approved` or `edited` → `alternatives` → END

3. **`retrieve`** is added only when `rag_examples` or `ground_entities` is on. With both off, the graph starts directly at `monitoring`.

### Mechanics

- `human_review` calls LangGraph's `interrupt()`; a `MemorySaver` checkpointer keeps state per `thread_id` until `resume_review()` sends `Command(resume=decision)`
- A `_timed` decorator appends each node's seconds and tokens to `state.trace` (an `operator.add` reducer, so entries accumulate across loops)
- The knowledge graph loads once (`lru_cache`); PageRank and degree centrality are computed once per graph object and cached with a `WeakKeyDictionary`

### Module Map

| Module | Purpose |
|---|---|
| `scpoc/pipeline.py` | Nodes, routing, `run_article()`, `resume_review()` |
| `scpoc/schemas.py` | Pydantic state and LLM output models |
| `scpoc/tools.py` | Entity resolution, BFS, risk scoring with reasoning, alternatives, network HTML |
| `scpoc/graph.py` | CSV loading and tier computation |
| `scpoc/rag.py` | BM25 exemplar search + entity grounding |
| `scpoc/llm.py` | Qwen3-8B, Azure OpenAI, and MockLLM backends |
| `scpoc/prompts.py` | All LLM prompts (monitoring, CSCO, self-critique, RAG context) |
| `scpoc/evaluate.py` | Evaluation framework with retry simulation |
| `scpoc/evalset.py` | 20 evaluation articles with ground truth |
| `scpoc/datagen.py` | Synthetic network generator |
| `app.py` | Web server + JSON API |
| `dashboard.html` | Dashboard UI (dark theme, single-file) |
| `cli.py` | Command-line interface |

---

## 4. Data Model

### Knowledge Graph

The knowledge graph is a NetworkX `DiGraph` whose edges point **supplier → customer**. A supplier's tier is its shortest upstream distance from the grocer being viewed, so the same company can be Tier-1 for one grocer and Tier-3 for another.

| File | Contents |
|---|---|
| `data/companies.csv` | 2,953 rows: `company_id`, `name`, `country`, `industry`, `is_focal` |
| `data/supplies.csv` | 4,438 rows: `supplier_id`, `customer_id`, `product` |
| `data/rag/exemplars.json` | 37 labelled articles for RAG retrieval |
| `data/eval_articles.json` | 20 evaluation articles with ground truth |

Generated by `datagen.py` with seed 7. Uses **40 countries**, **26 industries**, and a rule table of which industries each one buys from (e.g., Packaging buys from Plastics, Paper & Pulp, and Metals). Each grocer gets 35/140/420/900 links at Tiers 1–4, and 35% of new links reuse an existing supplier, so suppliers are shared across grocers.

| Grocer | Tier-1 | Tier-2 | Tier-3 | Tier-4 |
|---|---|---|---|---|
| Northwind Grocers (UK) | 30 | 157 | 648 | 994 |
| Alpenmarkt (Germany) | 24 | 161 | 684 | 1,184 |
| Maple Fresh Markets (Canada) | 32 | 168 | 693 | 1,291 |

### Pipeline State

`schemas.State` (Pydantic `BaseModel`) carries everything between nodes:

| Group | Fields |
|---|---|
| **Inputs** | `article`, `focal_name` |
| **Retrieval** | `retrieved` (dict with `examples` and `entities` lists) |
| **Extraction** | `monitoring` (`MonitoringOutput`: reasoning, is_disruption, disruption_type, summary, countries, industries, companies) |
| **Graph Query** | `focal_id`, `criteria` (graph-resolved values), `unresolved`, `bfs` |
| **Risk** | `risk`, `actions`, `network_html` |
| **Review** | `csco` (`CSCOOutput`), `approval`, `retry_count`, `retry_feedback`, `previous_monitoring`, `ai_suggestions`, `ai_confidence` |
| **Outcome** | `alternatives`, `status`, `message`, `trace` (timing/tokens) |

`disruption_type` is a closed `Literal`: geopolitical, economic_crisis, natural_disaster, cybersecurity, industrial_accident, labour_action, regulatory, other, none. Pydantic rejects any other value, which is what makes the LLM output strict.

---

## 5. Prompt Engineering

### Three Prompting Strategies Compared

The system was tested with three prompting strategies on the Monitoring extraction task:

#### Strategy 1: Direct JSON (baseline)
> "Read this article. Return JSON with is_disruption, countries, industries, companies."

**Result with Qwen3-8B**: Frequent over-generalization (listing whole industries for single-company events), buyer countries included as disrupted. The model jumps to extraction without reasoning.

#### Strategy 2: Chain of Thought + Strict Schema (adopted)
> "Think step by step: 1. What happened, where, to whom? 2. Is it a disruption? 3. Which entities are affected?"

The `MonitoringOutput` schema forces a `reasoning` field as the **first** field, so the LLM must write its reasoning before listing entities.

**Result**: Significant reduction in buyer/producer confusion. The reasoning step forces the model to distinguish "Germany imports grain from Ukraine" (Germany = buyer, not disrupted) before committing to a country list.

#### Strategy 3: Few-Shot with Solved Examples (adopted, via RAG)
Four static examples in the system prompt demonstrate:
- Company fire → list company only, not industry
- Country event → list source country only, not buyer
- Sector-wide ban → list industry
- Non-disruption → `is_disruption: false`

Additionally, when RAG is enabled, 3 dynamically selected similar examples are injected into the user prompt via BM25 retrieval.

**Result**: Combining CoT + few-shot + RAG examples gave the best extraction quality, particularly for edge cases.

### Qwen3-8B Thinking Mode

Qwen3-8B has a native `<think>...</think>` mode. We **disable** it (`enable_thinking: False`) because we control reasoning through the structured `reasoning` field in JSON output. As a safety net, any leaked `<think>` tags are stripped: `re.sub(r"<think>.*?</think>", "", text, flags=re.S)`.

### Prompt Catalogue

| Step | Prompt | Key Instructions |
|---|---|---|
| **Monitoring** (extraction) | `MONITORING_SYSTEM` + `MONITORING_USER` | CoT steps, 4 few-shot examples, disruption type enum, industry whitelist, buyer/producer distinction rules |
| **Monitoring** (retry) | + `MONITORING_RETRY_CONTEXT` appended | Previous extraction + human feedback injected; "correct the issues, do NOT repeat mistakes" |
| **CSCO Summary** | `CSCO_SYSTEM` + `CSCO_USER` | Write for executives; use only provided numbers; do not invent suppliers or change actions |
| **CSCO Summary** (RAG) | + `CSCO_RAG_CONTEXT` prepended | Similar past disruptions and verified entities from the knowledge graph |
| **Self-Critique** | `SELF_CRITIQUE_SYSTEM` + `SELF_CRITIQUE_USER` | Check 4 common mistakes: buyer countries, over-generalized industries, passively-mentioned companies, missing entities |

### JSON Enforcement per Backend

| Backend | Method |
|---|---|
| **Qwen3-8B** | JSON schema appended to system prompt; reply stripped of `<think>` tags and code fences; parsed and validated with Pydantic; one automatic repair retry |
| **Azure OpenAI** | Native structured output (`beta.chat.completions.parse`); one retry |
| **MockLLM** | Rule-based keyword matching; always produces valid schema |

Temperature is 0 on both real backends. The Qwen API enforces max_tokens ≤ 1,024, which is sufficient for all three output schemas.

---

## 6. RAG — Retrieval-Augmented Generation

RAG enhances **two** LLM steps (Monitoring Agent and CSCO Summary) with the same context. It is optional and controlled by environment variables: `RAG=1` enables both parts, or `RAG_EXAMPLES=1` and `GROUND_ENTITIES=1` individually.

### 6.1 BM25 Exemplar Search

A pure-Python Okapi BM25 implementation (k1=1.5, b=0.75) indexes 37 labelled exemplar articles stored in `data/rag/exemplars.json`.

**Tokenisation**: lowercase alphanumerics minus stop-words, with light plural stripping (`"strikes"` → `"strike"`, `"companies"` → `"company"`).

**Search**: For each new article, the top 3 exemplars by BM25 score are selected, with at most 2 of one kind (company, country, industry, mixed, false_positive) so the prompt shows a contrast between different scenario types.

**Corpus design**: The 37 exemplars are written separately from the 20 evaluation articles. A test (`test_corpus_does_not_leak_the_evaluation_set`) enforces that no evaluation article appears in the exemplar corpus — this prevents data leakage.

### 6.2 Entity Grounding

`EntityIndex` builds an inverted index of each company name's distinctive tokens (shared by ≤25 names in the graph). For each article:

1. Article words are matched against the inverted index (exact + fuzzy at ≥0.85 to survive typos)
2. Each candidate company name is compared with every same-length word window in the article using `SequenceMatcher`
3. A ratio ≥ 0.85 is a hit

Matched companies get their **verified industry** and **country** from the knowledge graph injected into the prompt, with specific instructions:
- Focal grocers are flagged: *"This is the grocery retailer we monitor (the customer, not a supplier). Do not list it under companies."*
- Suppliers are flagged: *"If affected, list under companies; do not also list its industry or country unless the whole sector/country is disrupted."*

**Critical safeguard**: Retrieval never shows the model the grocer's supplier network, so it cannot be nudged toward "finding" exposure.

### 6.3 RAG in the Monitoring Agent

The monitoring prompt receives:
- **Similar solved examples** showing correct extraction for articles like the current one
- **Grounded entity records** so the LLM knows "Sakri Packaging 020 is a Packaging company in Turkey" before extracting

This prevents the two most common small-model failures:
1. **Over-generalization**: "Sakri Packaging 020 had a fire" → without grounding the LLM might say `industries: ["Manufacturing"]`; with grounding it knows the exact industry and lists only the company
2. **Buyer/producer confusion**: examples demonstrate that destination countries should be excluded

### 6.4 RAG in the CSCO Summary

The CSCO step also receives the RAG context (similar solved examples and grounded entities) when retrieval is on. This lets the executive summary:
- Reference verified company details from the knowledge graph
- Reference patterns from similar past disruptions
- Produce more specific, grounded summaries rather than generic ones

The dashboard displays a **📚 RAG-Enhanced Summary** badge when this is active.

### 6.5 RAG Impact Measurement

When the server starts with ground-entities on, it automatically runs a background comparison:
- Baseline eval (no RAG) vs. ground-entities eval (with RAG) — against whichever LLM is active
- Results displayed in live KG Impact panels under the Retrieval, Risk Scoring, and Network Diagram steps
- `/api/kg-impact` serves the comparison; `/api/kg-impact/refresh` triggers recomputation

---

## 7. Agentic Layer

### 7.1 Multi-Agent Architecture

The system implements a **planner → executor → critic → human** loop using LangGraph:

| Agent | Role | LLM? |
|---|---|---|
| **Monitoring Agent** | Planner/Extractor — reads article, extracts structured disruption info | Yes (Qwen3-8B + RAG) |
| **KG Query Agent** | Executor — maps extraction to the knowledge graph, finds disrupted suppliers | No (deterministic) |
| **Risk Scorer** | Executor — computes 5-factor risk score with reasoning | No (deterministic) |
| **CSCO Agent** | Executor/Summariser — writes executive summary | Yes (Qwen3-8B + RAG) |
| **Self-Critique Agent** | Critic — reviews extraction, flags potential errors | Yes (Qwen3-8B) |
| **Human Reviewer** | Final authority — approves, corrects, or rejects | Human |

### 7.2 Persistent State Across Turns

LangGraph's `MemorySaver` checkpointer persists the full `State` object per `thread_id`. This enables:
- **Interrupt/resume**: the pipeline pauses at human review and resumes with the reviewer's decision
- **Multi-turn feedback**: up to 3 rounds of human feedback, with the full history of previous extractions maintained in state (`retry_count`, `retry_feedback`, `previous_monitoring`)
- **Accumulating trace**: `operator.add` reducer on `state.trace` accumulates timing and token data across loops

### 7.3 Tool Design

Five deterministic tools, each with strict input/output contracts:

| Tool | Input | Output |
|---|---|---|
| `resolve_entity` | Free-text company name | Canonical `company_id`, score, alternatives |
| `supply_chain_bfs` | Focal + criteria (countries, industries, companies) | Disrupted suppliers grouped by tier, with chain paths and products |
| `tier1_risk_scores` | BFS result + graph | Risk scores with reasoning per Tier-1 supplier |
| `recommend_actions` | Risk scores | HIGH → replace, MEDIUM → monitor, LOW → maintain |
| `find_alternatives` | HIGH-risk supplier + disrupted set | Replacement candidates from undisrupted regions |

### 7.4 Error Handling

| Failure Mode | Handling |
|---|---|
| **Malformed JSON from LLM** | Pydantic validation + one automatic repair retry |
| **LLM API rate limiting (429/503)** | Exponential backoff: 2, 4, 8 seconds |
| **Company name not in graph** | Stored in `state.unresolved`, shown to reviewer |
| **Self-critique failure** | Exception caught; returns empty suggestions (never blocks pipeline) |
| **Maximum retries exceeded** | Auto-rejection with message "Maximum retries (3) reached" |
| **LLM thinking mode leakage** | `<think>` tags stripped via regex |
| **API concurrency** | Class-level lock serialises requests (course API: 1 request per key) |

---

## 8. Core Algorithms

All algorithms live in `scpoc/tools.py` and `scpoc/rag.py`, use no LLM, and return plain JSON-serialisable dicts.

### 8.1 Name Resolution (`kg_query` node)

Countries and industries from the extraction are matched case-insensitively against the values in the graph. Company names go through `resolve_entity()`:
- **Exact** lower-case match scores 1.0
- **Fuzzy**: `difflib.get_close_matches` (cutoff 0.6) returns up to 4 candidates
- Pipeline accepts a company only if its best score ≥ 0.8
- Anything that fails is kept in `state.unresolved`, so the reviewer sees what the LLM named that the network does not contain

### 8.2 Disruption BFS (`supply_chain_bfs`)

The graph is reversed so a breadth-first search walks upstream from the grocer, with a cutoff of 4 hops. A supplier is **disrupted** if it matches any criterion (OR logic):

```python
disrupted = [n for n in tier_of
             if node.country in countries
             or node.industry in industries
             or n in companies]
```

For each disrupted supplier, the shortest chain `grocer → Tier-1 → … → supplier` is recorded with the **product** carried on each link. Chains are grouped under `tier_1` through `tier_4`.

### 8.3 Risk Scoring with Reasoning (`tier1_risk_scores`)

Every Tier-1 supplier whose upstream network contains a disrupted company is exposed and scored:

```
risk = 0.35 × B + 0.25 × D + 0.20 × C + 0.10 × T + 0.10 × E
```

| Component (weight) | Definition | Range |
|---|---|---|
| **B — Exposure Breadth** (0.35) | w = Σ 1/(hops+1) over disrupted nodes upstream; B = w/(w+3). Saturates: w=3 gives 0.5 | 0–1 |
| **D — Dependency Ratio** (0.25) | 1.0 if the Tier-1 itself is disrupted; else share of its direct suppliers that are disrupted or reach one within 2 more hops | 0–1 |
| **C — Downstream Criticality** (0.20) | Max over its disrupted nodes of max(normalised PageRank, normalised degree centrality) | 0–1 |
| **T — Tier-1 Centrality** (0.10) | Its degree centrality divided by the highest among the grocer's Tier-1 suppliers | 0–1 |
| **E — Exposure Depth** (0.10) | Deepest disrupted tier relative to the grocer, divided by 4 | 0–1 |

**Score → Level → Action:**
- HIGH ≥ 0.60 → Replace — source alternative supplier
- MEDIUM 0.45–0.59 → Increase monitoring
- LOW < 0.45 → Maintain standard operations

**Human-Readable Reasoning**: For each scored supplier, a plain-English reasoning string is generated by checking which components exceed their thresholds. For example:

> *Sakri Packaging 020: HIGH (0.613) — directly disrupted (is itself in the affected region/industry); high criticality — a critical node in the network (PageRank/degree); high centrality among Tier-1 suppliers — many connections*

This reasoning is:
- Displayed in the dashboard risk table ("Why" column)
- Shown in the network diagram reasoning panel below the SVG
- Included in SVG node tooltips on hover

### 8.4 Network Diagram with Connection Reasoning (`plot_network`)

Generates a self-contained dark-themed HTML/SVG diagram:
- Grocer on the left, Tier-1 through Tier-4 in columns
- **Node colours**: focal (purple), HIGH risk (red), MEDIUM risk (amber), LOW risk (green), not exposed (grey), disrupted (red with glow)
- **Edge tooltips**: `"X supplies [product] to Y"` + ⚠ warning if the supplier is disrupted

Below the diagram, two visible panels (no hovering required):
1. **"Why are these suppliers flagged?"** — each supplier with risk level, industry, country, product supplied, score, and reasoning text
2. **"Supply Chain Paths (product flow)"** — the actual chain showing products at each link

### 8.5 Alternative Sourcing (`find_alternatives`)

Runs only for HIGH-risk suppliers, after approval. Scans the whole graph for companies that:
1. Are in the same industry as the at-risk supplier
2. Are not a grocer
3. Do not match the disruption criteria themselves

For each candidate, walks 3 hops upstream and computes exposure = Σ 1/(hops+1) over disrupted upstream nodes. Candidates are sorted by (exposure, upstream_size, name). Fully clear candidates (`verified_clear = true`) are returned first. Top 5 returned.

### Worked Example: Sakri Packaging 020 Fire

| Step | What happens |
|---|---|
| **RAG Retrieval** | BM25 finds 3 similar articles; entity grounding identifies "Sakri Packaging 020" as a Packaging company in Turkey |
| **Monitoring** | Extracts: `is_disruption: true`, `disruption_type: industrial_accident`, `companies: ["Sakri Packaging 020"]` |
| **KG Query** | Resolves to company C00020. BFS finds 1 disrupted supplier at Tier-1 |
| **Risk Scoring** | Sakri: B=0.250, D=1.000, C=0.750, T=1.000, E=0.250 → **0.613 HIGH**. Reasoning: "directly disrupted; high criticality; high centrality" |
| **CSCO Summary** | RAG-enhanced executive summary referencing verified entity details |
| **Self-Critique** | Confidence: medium, no corrections needed |
| **Human Review** | Reviewer approves |
| **Alternatives** | 48 candidates considered, all 48 clear. Top 3: Kanor Packaging 065 (Canada), Miri Packaging 067 (Indonesia), Love Packaging 003 (Mexico) |

**Contrast — country-wide event (Ukraine blockade):** Extracts `countries: ["Ukraine"]`, finds 48 disrupted suppliers (none at Tier-1; 5/17/26 at Tiers 2–4), 30 exposed Tier-1s. Breadth drives ranking: Taka Beverages 003 scores 0.627 HIGH with 11 disrupted upstream suppliers.

---

## 9. Human-AI Feedback Loop

### The Problem

The base pipeline has a one-shot human review: approve, edit, or reject. If the reviewer rejects, the analysis is discarded — the AI learns nothing from the correction.

### The Solution

A **human-AI feedback loop** where the reviewer's corrections guide the AI to produce a better result.

### Five Review Decisions

| Decision | Extra Fields | Effect | Next Node |
|---|---|---|---|
| **approve** | — | Accept as-is | alternatives |
| **edit** | csco, actions | Merge reviewer's edited summary and actions | alternatives |
| **retry** | reason (feedback text) | Save feedback and previous extraction; +1 retry count | monitoring (3 LLM calls: extraction + CSCO + self-critique) |
| **edit_extraction** | monitoring (edited countries/industries/companies) | Overwrite extraction fields; no LLM call | kg_query (re-runs: KG → risk → viz → CSCO → self-critique → review) |
| **reject** | reason | status = rejected, pipeline ends | END |

### AI Self-Critique

Before the reviewer sees the analysis, a separate LLM call reviews the extraction using 4 checks:
1. Country listed that is only a buyer/destination, not where the disruption happened
2. Entire industry listed when only one company was affected
3. Company listed that the article only mentions in passing
4. Countries, industries, or companies that should be listed but were missed

Output: `SelfCritiqueOutput` with `suggestions` (list of strings) and `confidence` (high/medium/low). Suggestions are **clickable** in the dashboard — clicking one auto-fills the feedback textbox.

### Retry Mechanics

When the reviewer sends feedback:
1. `state.retry_feedback` stores the feedback text
2. `state.previous_monitoring` saves the current extraction
3. `state.retry_count` increments
4. `after_review` routes to `monitoring` (graph cycle)
5. The monitoring prompt receives `MONITORING_RETRY_CONTEXT` with the previous extraction + feedback
6. The LLM re-runs with: original article + RAG context + feedback
7. All downstream steps re-run (KG → risk → viz → CSCO → self-critique → review)
8. Dashboard shows "Analysis #2" with diff badges comparing extractions

Maximum 3 retries (configurable via `max_retries`). After max retries, auto-rejection.

### Edit Extraction (Partial Retry)

For fine-grained fixes: the reviewer directly edits countries, industries, or companies. This:
- **Skips the LLM entirely** — routes to `kg_query`
- Re-runs only downstream steps (KG → risk → viz → CSCO → self-critique → review)
- Guarantees the exact fix the reviewer wanted

Implementation: `MonitoringOutput(**{**existing.model_dump(), **edits})` merges reviewer's edits into the existing extraction.

---

## 10. Evaluation & Observability

### Evaluation Framework

```bash
python cli.py eval [--mock] [--rag] [--with-retry] [--max-retries N] [--ids a,b,c]
python cli.py eval-compare file1.json file2.json
```

### Metrics Reported

| Category | Metrics |
|---|---|
| **Extraction quality** | Precision, recall, F1 per field (countries, industries, companies) + micro-averaged overall |
| **Exact match** | Articles where extraction perfectly matches ground truth |
| **Network query** | Disrupted count match, exposure detected, disrupted supplier count match |
| **Risk scoring** | Top Tier-1 supplier match, top HIGH-risk supplier match |
| **False positives** | Articles correctly ended at "no exposure" (5/5 expected) |
| **Cost** | Time per article, prompt tokens, completion tokens |

### Mock LLM Baseline

| Measure | Result |
|---|---|
| Extraction F1 (overall / companies / countries / industries) | 0.703 / 1.000 / 0.700 / 0.444 |
| Exact extraction / disrupted count matches / top Tier-1 matches | 12/20 · 7/15 · 8/15 |
| False positives ended "no exposure" | 5/5 |

These figures measure the keyword mock, not a real LLM. The mock's typical errors are ones the prompts are built to prevent: listing buyer countries and missing sector-wide events phrased without the industry name.

### With-Retry Evaluation

`--with-retry` mode simulates human feedback from the ground truth:
- Compares extraction to expected answer
- Generates specific feedback (e.g., *"Remove countries: Germany — not actually disrupted"*)
- Sends feedback to the LLM and measures improvement
- Reports: articles retried, fixed by retry, still wrong after retry, mean retries per article

### Comparative Evaluation

`eval-compare` takes two eval result files and shows per-article deltas: which articles improved, which regressed, and by how much.

```bash
# Compare baseline vs RAG
python cli.py eval --mock                        # outputs eval_results.json
python cli.py eval --mock --ground-entities      # outputs eval_results_ground-entities.json
python cli.py eval-compare outputs/eval_results.json outputs/eval_results_ground-entities.json
```

### Observability

| Feature | Implementation |
|---|---|
| **Per-node timing** | `_timed` decorator records seconds per node in `state.trace` |
| **Token counting** | Prompt and completion tokens per LLM call tracked in `state.trace` |
| **Dashboard timeline** | Each step card shows timing and token count |
| **Analytics overlay** | `/api/analytics` aggregates: total jobs, retry rate, avg retries, outcome distribution, feedback log |
| **KG Impact dashboard** | `/api/kg-impact` shows live baseline-vs-grounding comparison |

### Failure Taxonomy

| Failure Type | How Detected | Example |
|---|---|---|
| **Extraction error** | F1 < 1.0 on extraction fields | Listed Germany as disrupted (buyer confusion) |
| **Retrieval miss** | Entity not found in graph | Misspelled company name below 0.8 fuzzy threshold |
| **Over-generalization** | Industry listed + company listed for same event | "Fire at one factory" → lists entire Packaging industry |
| **False negative** | `is_disruption: false` for a real disruption | Model classifies disruption as "forecast" |
| **Tool error** | Pipeline error status | Company name resolves to wrong entity |

---

## 11. Deployment & Interface

### Web API

| Method & Path | Body / Result |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /api/status` | `{llm, mock}` — which LLM is active |
| `GET /api/scenarios` | Pre-built scenario list + grocer list |
| `POST /api/run` | `{article, focal}` → `{job_id}` |
| `GET /api/job/{id}` | Finished steps with timing/tokens, status, review payload |
| `POST /api/review` | `{job_id, decision, csco?, actions?, reason?, monitoring?}`; 409 unless awaiting review |
| `GET /api/network/{id}` | Network diagram HTML |
| `GET /api/analytics` | Retry stats, outcome distribution, feedback log |
| `GET /api/kg-impact` | Baseline vs grounding comparison |
| `POST /api/kg-impact/refresh` | Trigger recomputation |

A standard-library `ThreadingHTTPServer` runs each job on a background thread and streams LangGraph updates into an in-memory `JOBS` dict. The dashboard polls `/api/job/{id}` every 500ms for progress.

### Dashboard Features

- **Scenario picker** — 4 pre-built articles + custom paste
- **Live pipeline timeline** — each step appears as it finishes with timing and token counts
- **Bento grid KPIs** — disrupted count, tier breakdown, risk level distribution
- **Risk table with "Why" column** — human-readable reasoning per supplier
- **Network diagram** — interactive SVG with colour coding
- **Reasoning panel** — "Why are these suppliers flagged?" + supply chain paths with products (visible, no hover needed)
- **RAG-enhanced CSCO badge** — shows when executive summary used RAG context
- **AI self-critique suggestions** — clickable to auto-fill feedback box
- **Edit extraction** — directly modify countries/industries/companies
- **History view** — diff badges comparing analysis rounds
- **Analytics overlay** — retry rate, feedback log, outcome distribution
- **KG Impact panels** — live baseline-vs-grounding metrics

### CLI

| Command | What it does |
|---|---|
| `run --text "..." [--focal NAME] [--auto approve] [--rag]` | One article through the full pipeline |
| `demo [--country X] [--industry Y]` | Tools only, no LLM needed |
| `eval [--mock] [--rag] [--with-retry]` | 20-article evaluation |
| `eval-compare file1 file2` | Side-by-side delta of two eval runs |

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env              # optional: QWEN_* or AZURE_OPENAI_* keys
RAG=1 python app.py               # dashboard on http://localhost:8080
pytest -q                          # 31 tests, no key needed
```

---

## 12. Responsible AI

### Human-in-the-Loop Boundary

No extraction, risk assessment, or supplier replacement proceeds without **explicit human approval**. The pipeline pauses with `interrupt()` and cannot resume programmatically — only a human decision (via the API) moves it forward.

### PII and Data

- The knowledge graph is fully synthetic (seed-generated names, countries, industries)
- No real company data, supplier relationships, or trade secrets
- `.env` is git-ignored; no API keys are committed
- The CA certificate (`vm05-ca.crt`) is in the repository — it is not a secret, but each course group needs its own

### Prompt Injection Protection

- All LLM outputs are validated against strict Pydantic schemas — arbitrary text cannot change the pipeline's behaviour
- The extraction output must conform to a closed enum for `disruption_type` and specific list types for countries/industries/companies
- The pipeline's routing logic is in Python code (conditional edges), not in LLM output

### Failure Transparency

- Unresolved entity names are shown to the reviewer (not silently dropped)
- Self-critique suggestions are shown before the reviewer decides
- Retry history with diff badges shows exactly what changed between rounds
- Risk reasoning explains why each supplier is scored, not just the number

### Limitations of Human-in-the-Loop

- State is in memory (`MemorySaver`) — lost on server restart
- Dashboard binds to `127.0.0.1` by default with no authentication
- No audit trail beyond the in-memory `JOBS` dict

---

## 13. Limitations & Future Work

| Limitation | Impact | Suggested Fix |
|---|---|---|
| **Criteria combine with OR** | "Packaging makers across India halted" flags every Indian company AND every packaging company: 64 suppliers in that eval case | Add an optional AND/intersection scope for "this industry in this country" |
| **Exposure exceeds disrupted_count** | Sakri is 1 disrupted supplier but exposes 3 Tier-1s, because tier uses only the shortest path | Report exposed Tier-1 count alongside disrupted count (done in dashboard KPIs) |
| **Fuzzy match on numbered names** | "Zenkri Packaging 012" vs "… 009" scores 0.789, just under the 0.8 cut | Match numeric suffix exactly, then fuzzy-match the words |
| **Alternatives match industry only** | A can maker can be offered to replace a box maker; capacity and cost are ignored | Filter by the product on the at-risk edge |
| **State is in memory** | `MemorySaver` and `JOBS` are lost on restart | Use a SQLite or Postgres checkpointer |
| **Breadth constant K=3 is a project choice** | The paper does not define this value | Calibrate on labelled cases or make it configurable |
| **Synthetic data only** | Real supply chains are orders of magnitude more complex | Pilot on one real supplier list |
| **Self-critique errors are swallowed** | Exception in self-critique returns empty suggestions silently | Log critique failures for observability |
| **No caching of LLM calls** | Same article re-analysed hits the API again | Add response caching keyed on (prompt hash, temperature) |
| **Single-threaded LLM calls** | Course API lock serialises all requests | Parallelise where the API allows multiple keys |

---

## Appendix A: Test Suite

All **31 tests** pass in ~2 seconds with no API key.

| Suite | Tests | Covers |
|---|---|---|
| `test_pipeline.py` | 12 | Approve, reject, edit, retry, max retries, self-critique, partial edit extraction, early exits (no disruption, no exposure, self-contradicting) |
| `test_rag.py` | 12 | Tokenisation, BM25 ranking, exemplar kind limits, format validation, corpus validity, corpus vs eval isolation, entity grounding (exact + fuzzy), focal grocer flagging, entity format, RAG flags off/on, state serialisation |
| `test_tools.py` | 7 | Entity resolution, tiers and lists, BFS chain structure, risk component values, Tier-1-itself-disrupted case, no-disruption case, actions and plot generation |

---

## Appendix B: Risk Scoring Worked Example

**Sakri Packaging 020 (fire scenario):**

| Component (weight) | Sakri Packaging 020 | Lohel Meat 026 |
|---|---|---|
| B — Breadth (0.35) | w = 1/(0+1) = 1 → 1/(1+3) = **0.250** | w = 1/(1+1) = 0.5 → 0.5/3.5 = **0.143** |
| D — Dependency (0.25) | Itself disrupted → **1.000** | 1 of 4 inputs → **0.250** |
| C — Criticality (0.20) | **0.750** | **0.750** |
| T — Centrality (0.10) | Highest-degree Tier-1 → **1.000** | **0.208** |
| E — Depth (0.10) | Tier-1 → 1/4 = **0.250** | Tier-2 → 2/4 = **0.500** |
| **Score** | 0.088 + 0.250 + 0.150 + 0.100 + 0.025 = **0.613 HIGH** | 0.050 + 0.063 + 0.150 + 0.021 + 0.050 = **0.333 LOW** |

**Reasoning generated:**
- Sakri: *"directly disrupted (is itself in the affected region/industry); high criticality — a critical node in the network (PageRank/degree); high centrality among Tier-1 suppliers — many connections"*
- Lohel: *"high criticality — a critical node in the network (PageRank/degree); disruption reaches deep (tier 2)"*
