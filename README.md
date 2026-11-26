# Supply Chain Disruption Monitor

An agentic AI proof of concept that reads a news article, decides whether it is a real supply chain disruption, finds which suppliers (Tier-1 to Tier-4) of a grocer are hit, scores the risk, and recommends actions. A person reviews and approves before anything is acted on.

Based on: *AlMahri, Xu & Brintrup, "Automating Supply Chain Disruption Monitoring via an Agentic AI Approach"*.

- **Author:** Rohan Dawar
- **Python:** 3.10 or newer (tested on 3.14)
- **Data:** dummy only (made-up companies), in-memory NetworkX graph, no database

## Contents

1. [What it does](#what-it-does)
2. [Quick start](#quick-start)
3. [Configuration](#configuration)
4. [Using the dashboard](#using-the-dashboard)
5. [Fake-news (credibility) check](#fake-news-credibility-check)
6. [Alternative suppliers](#alternative-suppliers)
7. [Retrieval (RAG)](#retrieval-rag)
8. [Risk score](#risk-score)
9. [Langfuse tracing](#langfuse-tracing-optional)
10. [Command line](#command-line)
11. [Evaluation](#evaluation)
12. [Tests](#tests)
13. [Project structure](#project-structure)
14. [Known limitations](#known-limitations)
15. [Troubleshooting](#troubleshooting)

## What it does

Give it a **news article** and a **grocer**. The pipeline:

1. **Retrieval (RAG, optional):** finds similar solved articles and checks company names against the knowledge graph.
2. **Credibility check:** extracts the article's claims and checks them, giving a Genuine / Suspicious / Likely fake verdict.
3. **Monitoring agent (LLM):** extracts disruption type, countries, industries and companies as strict JSON.
4. **Knowledge-graph query:** a breadth-first search finds every disrupted supplier (Tier-1 to Tier-4) and its chain down to the grocer.
5. **Risk scoring:** scores each exposed Tier-1 supplier with a weighted five-factor formula and a plain-language reason.
6. **Network diagram:** an interactive tiered diagram (hover nodes and edges for details).
7. **CSCO summary (LLM):** an executive summary written for the Chief Supply Chain Officer. It only narrates; the actions come from fixed thresholds.
8. **AI self-critique (LLM):** flags likely extraction mistakes before the human sees them.
9. **Alternative sourcing preview:** replacement candidates for the top suppliers, shown during review.
10. **Human review:** approve, edit, send feedback for re-analysis, or discard.
11. **Alternative sourcing:** replacement candidates for each HIGH-risk supplier after approval.

Up to four steps use an LLM (credibility claims, monitoring, CSCO summary, self-critique). Everything that produces numbers or decisions (supplier matching, risk scores, actions, alternatives, the fake-news verdict) is plain Python, so results are repeatable.

```
[retrieve] -> [credibility] -> monitoring (LLM) -> kg_query -> [nothing disrupted? stop: "no exposure"]
   -> risk -> visualizer -> csco (LLM) -> self_critique (LLM) -> preview_alternatives
   -> human review (pause) -> alternatives -> done
        |-- send feedback --> monitoring (up to 3 rounds)
        |-- edit extraction --> kg_query
        '-- discard --> end
```

The run also stops with "no exposure" when the model's own verdict is "not a disruption", even if some names matched. The human review is a LangGraph `interrupt()`, so the run pauses until someone decides.

## Quick start

```bash
# 1. Create a virtual environment OUTSIDE OneDrive (see the note below)
mkdir -p ~/venvs
python3 -m venv ~/venvs/capstone
source ~/venvs/capstone/bin/activate

# 2. Install
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the dashboard (works with no keys: it uses an offline mock LLM)
python app.py                        # http://localhost:8080
```

Windows PowerShell:

```powershell
python -m venv $HOME\venvs\capstone
$HOME\venvs\capstone\Scripts\Activate.ps1
```

If PowerShell blocks the activate script, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. With conda instead: `conda create -n capstone python=3.12 -y && conda activate capstone`.

> **Keep the virtual environment out of OneDrive.** OneDrive syncing can block or badly slow `pip` inside a synced folder ("Operation not permitted"). Keep the code where you like, but put the environment in your home folder. Your prompt should show `(capstone)` when it is active; leave it with `deactivate`.

Check the install:

```bash
python -c "import pandas, networkx, scipy, pydantic, langgraph; print('ok')"
python cli.py demo                   # tools only, no LLM needed
```

Other useful settings:

```bash
PORT=9000 python app.py              # another port (PowerShell: $env:PORT=9000; python app.py)
HOST=0.0.0.0 python app.py           # let other machines connect (default is 127.0.0.1 only)
RAG=1 python app.py                  # turn on retrieval
CREDIBILITY=0 python app.py          # turn off the fake-news check
```

Any of these can also go in `.env`. Stop the app with Ctrl+C.

## Configuration

Copy the template and fill it in (put each setting on its own line, with no spaces before it and no comment after the value):

```bash
cp .env.example .env                 # Windows: copy .env.example .env
```

The app picks the first LLM available, in this order:

| LLM | When |
|---|---|
| **Qwen3-8B** | `QWEN_API_KEY` is set (course Track B API, with a CA certificate) |
| **Azure OpenAI** | all four `AZURE_OPENAI_*` variables are set |
| **Offline mock** | no keys; keyword-based, fine for trying the flow but not a measure of extraction quality |

The top right of the dashboard shows which one is active.

### Qwen3-8B (course API)

1. Put your group's CA certificate in the project folder (for example `vm05-ca.crt`).
2. Fill in `.env`:

```
QWEN_API_BASE=https://103.42.50.41:1033/v1
QWEN_API_KEY=<your-group-key>
QWEN_CA_CERT=vm05-ca.crt
QWEN_MODEL=Qwen/Qwen3-8B
```

| Groups | Port in the URL | Certificate |
|---|---|---|
| 001-015 | `:1031` | `vm07-ca.crt` |
| 016-030 | `:1032` | `vm06-ca.crt` |
| 031-045 | `:1033` | `vm05-ca.crt` |
| 046-065 | `:1034` | `vm04-ca.crt` |

The base URL already ends in `/v1`; do not add `/chat/completions`. Check the connection:

```bash
curl --cacert vm05-ca.crt -H "Authorization: Bearer <key>" https://103.42.50.41:1033/v1/models
```

API limits, handled by the app: one active request per key (calls are queued; do not run two things with the same key at once), 60 requests per minute, 1024 output tokens per call, 8192 tokens input plus output. On 429 or 503 it retries after 2, 4 and 8 seconds, then reports the error.

### Azure OpenAI (optional)

Used only if `QWEN_API_KEY` is not set. Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` and `AZURE_OPENAI_API_VERSION`, and install the `openai` package. This path is written but has not been tested against a live deployment.

`.env` holds secrets. It is in `.gitignore`: do not commit it or paste keys into screenshots, tickets or chat.

## Using the dashboard

Open http://localhost:8080.

1. Pick a ready-made scenario (Ukraine conflict, plant fire at a packaging supplier, port strike in Vietnam, two fake-news examples, or the storm-in-Iceland false positive) or paste your own article. Choose the grocer.
2. Click **Run analysis**. Each step appears as it finishes, with its time and token use. With RAG on, a "Retrieval (RAG)" card comes first.
3. **Human review.** The panel shows the fake-news verdict, the AI self-critique, an editable summary, an action dropdown for each supplier, and the alternative suppliers for the top suppliers.
4. After approval, **Alternative sourcing** lists replacement candidates for each HIGH-risk supplier. "Clear = no" means the candidate's own suppliers are also exposed, which is common on this dummy data.
5. If nothing in the network is hit, or the article is not a disruption, the run stops with "no exposure".

Other features: a KPI bar (disrupted suppliers, tier breakdown, risk levels), a risk table with a reason for each score, a network diagram with connection reasoning, diff badges when re-analysis is used, an **Analytics** overlay (retry rate, feedback history, outcomes), live baseline-versus-grounding comparison panels, and a **Light / Dark** button (your choice is remembered; the first time the page follows your system setting). Runs are kept in memory, so restarting the app clears them.

### Human review options

| Action | What happens |
|---|---|
| **Approve** | Accept the analysis, then find replacement suppliers |
| **Approve with edits** | Change the summary or any action, then find replacements |
| **Send feedback and re-analyse** | The AI re-runs the extraction using your corrections (up to 3 rounds) |
| **Edit extraction directly** | Fix countries, industries or companies and re-run only the downstream steps (no LLM) |
| **Discard analysis** | Stop; nothing is acted on |

The AI self-critique suggestions can be clicked to fill in the feedback box. See [AI_FEEDBACK_LOOP.md](AI_FEEDBACK_LOOP.md) for details.

## Fake-news (credibility) check

The first step reads the article and extracts its claims and any warning signs (the only LLM part). The checks are plain Python:

- do the named companies exist in the supplier network, and is a company really in the country and industry the article says;
- do claims contradict each other;
- are there warning signs such as sensational wording, calls to share or panic, or vague sourcing ("sources say");
- with RAG examples on, how many of the closest labelled historic articles were false positives.

The verdict is **Genuine**, **Suspicious** or **Likely fake**, with the reasons and the evidence for each claim. There is no live fact-checking, so it cannot prove an event happened; a person decides. By default a Likely fake verdict only warns and the run continues.

```bash
CREDIBILITY=0          # turn the check off in the dashboard
CREDIBILITY_BLOCK=1    # stop the run when the article looks fake
python cli.py run --credibility [--block-fake] --text "..."
```

The scenarios "Fake news: unknown company" and "Fake news: real company, wrong country" show it. It is off in `cli.py eval` unless added to the code.

## Alternative suppliers

Shown in the review panel before you approve (top 5 suppliers by risk; HIGH ones open by default) and again after approval. For each supplier being replaced:

1. **Same industry** as that supplier.
2. **Outside the disruption:** the candidate's country, industry and name do not match what the article says is hit.
3. **Clean upstream:** the candidate's own suppliers down to Tier-4 (it would become your Tier-1, so its Tier-2 to Tier-4) include no disrupted company. These are "clear".
4. **Exposure score**, for candidates that are not clear: each disrupted upstream supplier adds 1/(hops+1), so nearer ones count more. Clear = 0.
5. **Ranking:** lowest exposure, then smaller upstream network (less can go wrong), then name.
6. **Fallback:** exposed candidates are listed only when no clear one exists, and are marked not clear. The top 5 are shown, with how many were considered.

Only the industry is matched. Product, capacity, price and lead time are not considered, so check them before switching. The code is `find_alternatives()` in `scpoc/tools.py`.

## Retrieval (RAG)

Optional, **off by default**. With it off, the pipeline and results are exactly as before.

**Why:** the baseline evaluation shows the Monitoring step (Qwen3-8B) mostly fails by listing too much: it adds a named company's whole industry (one company becomes hundreds of disrupted suppliers), sectors an article only mentions in passing, or countries that merely buy the goods. Retrieval gives the model two kinds of help before it answers, each with its own switch:

1. **Similar solved examples** (`--rag-examples`). `data/rag/exemplars.json` holds 37 short labelled articles (single-company, country, industry and country-and-industry events, plus false positives such as forecasts, events that are over, and routine news). A BM25 keyword search picks the 3 most similar (at most 2 of one kind, so the model sees a contrast) and adds them, with their correct JSON answers, to the prompt. The same context also goes to the CSCO summary.
2. **Company grounding** (`--ground-entities`). Companies from the knowledge graph that the article names (exact name or a near miss) are listed in the prompt with their industry and country, plus a reminder that listing the company already covers it. A grocer named in the article is marked as the retailer, not a supplier. Only companies the article names are shown, never the grocer's supplier network.

The search is plain Python (no extra packages, no vector database, no embeddings API) and deterministic. It adds about 500 prompt tokens per article.

```bash
RAG=1 python app.py                       # both parts (also RAG_EXAMPLES=1 or GROUND_ENTITIES=1)
python cli.py run --rag --text "..." --auto approve
```

CLI switches (for `run` and `eval`): `--rag-examples`, `--ground-entities`, `--rag` (both), `--k 3` (number of examples). In code: `build_app(rag_examples=True, ground_entities=True, k_examples=3)`.

**Adding or editing examples.** Each entry in `data/rag/exemplars.json` has an `id`, a `kind` (`company`, `country`, `industry`, `country_industry` or `false_positive`), an `article`, and an `answer` that matches the Monitoring output schema in `scpoc/schemas.py` (industries and countries must exist in the graph). Rules, checked by `tests/test_rag.py` (run `pytest` after editing):

- never copy or paraphrase an evaluation article; each example must share less than 40% of its words with every evaluation article;
- never use an evaluation company or any real company from the graph: use made-up names;
- ids must be unique.

Keep examples short and make the reasoning say which rule applies ("Germany is only a buyer", "one firm is not a whole sector"). Tuning knobs are in `scpoc/rag.py` (`ExemplarStore.search(k, max_per_kind)`, `EntityIndex(min_score, max_df)`).

**Caveats:** the examples were written after looking at the baseline failures, so an improvement may partly fit this evaluation set; confirm it on a few new articles. BM25 matches words, not meaning. Grounding only finds companies that are in the graph.

## Risk score

```
score = 0.35 x breadth + 0.25 x dependency + 0.20 x criticality
      + 0.10 x centrality + 0.10 x depth          (each part is 0 to 1)

HIGH >= 0.60  -> replace         MEDIUM 0.45-0.59 -> increase monitoring
LOW  <  0.45  -> maintain standard operations
```

| Part | Meaning |
|---|---|
| breadth | disrupted suppliers upstream of the Tier-1; closer ones count more (1/(hops+1) each); the total x is squashed to x/(x+3) |
| dependency | share of the Tier-1's direct suppliers that lead to a disruption (1 if the Tier-1 itself is disrupted) |
| criticality | highest PageRank or degree centrality among its disrupted suppliers |
| centrality | how connected the Tier-1 is, compared with the grocer's other Tier-1 suppliers |
| depth | deepest disrupted tier / 4 |

The paper gives the weights and thresholds but not the formulas for breadth and dependency; those two are this POC's own choices. Each supplier also gets a short reason, for example: *"directly disrupted; high criticality; high centrality among Tier-1 suppliers"*.

## Langfuse tracing (optional)

Off unless both keys are set. Add to `.env` (see `.env.example`):

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com        # EU; US is https://us.cloud.langfuse.com
```

`langfuse` is in `requirements.txt`. Restart the app after changing `.env`; the startup line then ends with "Langfuse tracing". In Langfuse you get:

- one trace per run, kept as one trace across the human-review pause, grouped by session (the dashboard shows a "View this run in Langfuse" link);
- a span per step and a generation (prompt, output, tokens) per LLM call; Qwen retries and JSON repairs appear as events;
- scores: `human_decision`, `retry_count`, `edited_actions`, `credibility_verdict`, `credibility_score`, `risk_top_score`, `disrupted_suppliers`, `final_status`;
- the alternatives spans carry the selection rule and the counts per supplier.

**Prompts.** `python cli.py langfuse-push-prompts` creates the 4 system prompts in Langfuse (skipping existing ones). After that, edit them in Langfuse and give a version the label `production`; the app picks it up within about a minute. If Langfuse is unreachable or a prompt is missing, the text in `scpoc/prompts.py` is used. Keep the `{industries}` placeholder in `monitoring_system`.

**Evaluation.** `python cli.py langfuse-sync-dataset` uploads the 20 articles as a dataset. `python cli.py eval --langfuse-run NAME` puts a whole run in one session with per-article scores (extraction F1, matches, tokens); compare runs by filtering on the session or tag. Traces are not attached to the dataset items as dataset runs.

**Privacy.** Prompts, articles and outputs are stored in Langfuse. Set `LANGFUSE_MASK_ARTICLES=1` to send only names, timings, tokens and scores. Tracing never stops a run: if Langfuse is down, calls are skipped with a one-time warning (each step may wait a few seconds the first time it fetches a prompt).

## Command line

```bash
python cli.py data                                   # regenerate the dummy network
python cli.py demo                                   # tools only (no LLM), Ukraine, Northwind
python cli.py demo --focal alpenmarkt --industry Packaging
python cli.py run --text "War in Ukraine halts grain exports." --auto approve
python cli.py run --article article.txt              # asks you to approve
python cli.py run --rag --credibility --text "..."   # retrieval and fake-news check
```

Generated files go to `outputs/`.

## Evaluation

Needs the real LLM to be meaningful. There are 20 hand-written articles with expected answers: 15 that hit the network (4 single-company events at Tier-1 to Tier-4, 5 country events, 3 industry events, 3 country-and-industry events) and 5 false positives that must end with "no exposure". Each has a hand-written truth; the expected disrupted count, top Tier-1 supplier and top HIGH supplier are computed from that truth with the deterministic tools. Because the tools are deterministic, any mismatch traces back to what the LLM extracted.

```bash
python cli.py eval                          # all articles, baseline
python cli.py eval --ids co_t1,fp_chips     # only some
python cli.py eval --mock                   # plumbing check with the offline mock
python cli.py eval --rag                    # also --rag-examples, --ground-entities
python cli.py eval --with-retry --max-retries 2   # simulate reviewer feedback and measure improvement
python cli.py eval-build                    # regenerate data/eval_articles.json
```

It reports extraction precision, recall and F1 (countries, industries, companies; places or companies named but not in the network count as false positives), whether the network query and risk results match, false-positive handling, and time and tokens per article. Pass `--price-in` and `--price-out` (USD per 1K tokens) for cost; the course API has no price, so cost shows as 0. Each configuration writes its own files in `outputs/` (`eval_report*.txt`, `eval_results*.json`), so runs do not overwrite each other. Compare them side by side (the first file is the reference):

```bash
python cli.py eval-compare outputs/eval_results_baseline.json outputs/eval_results_rag-examples.json
```

The LLM can give slightly different answers between runs even at temperature 0. `eval --mock --rag` only checks that everything runs; the mock ignores the retrieved context.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                               # 53 tests, no API key or internet needed
```

The tests use a 6-node hand-built network where every number can be checked by hand, and a scripted LLM.

| Suite | Tests | Covers |
|---|---|---|
| `test_pipeline.py` | 12 | approve, reject, edit, retry, max retries, self-critique, partial edit, early exit |
| `test_rag.py` | 12 | tokenisation, BM25 ranking, corpus validity, entity grounding, prompt formatting |
| `test_tools.py` | 7 | entity resolution, BFS, risk scoring, alternatives, network diagram |
| `test_credibility.py` | 13 | claim checks and verdicts, warn versus block, alternatives shown before approval |
| `test_langfuse.py` | 9 | off without keys, spans and generations, scores, masking, a failing Langfuse never breaks a run |

## Project structure

```
app.py                  Web server + JSON API (serves dashboard.html)
dashboard.html          Dashboard UI (light and dark)
cli.py                  CLI: data | demo | run | eval | eval-build | eval-compare | langfuse-*
architecture.html       Interactive pipeline architecture diagram
AI_FEEDBACK_LOOP.md     Documentation for the feedback loop
.env.example            Template for the settings (copy to .env)
requirements.txt        Dependencies to run
requirements-dev.txt    Extra dependency for the tests (pytest)
scpoc/
  pipeline.py           LangGraph pipeline with human review, RAG and credibility
  schemas.py            Pydantic models (State, MonitoringOutput, CSCOOutput, claims, credibility)
  prompts.py            All LLM prompts
  llm.py                Qwen3-8B client, Azure OpenAI client, offline mock LLM
  tools.py              Deterministic tools (entity resolution, BFS, risk, actions, alternatives, diagram)
  credibility.py        Evidence checks and verdict for the fake-news step
  rag.py                BM25 exemplar search and entity grounding
  observability.py      Optional Langfuse tracing, prompts, scores, dataset
  evaluate.py           Evaluation framework
  evalset.py            The 20 evaluation articles and their expected answers
  graph.py              Load the network, tiers, find suppliers
  datagen.py            Makes the dummy network
data/
  companies.csv         About 2,900 dummy companies across 3 grocers
  supplies.csv          Supply relationships
  eval_articles.json    Evaluation set with ground truth
  rag/exemplars.json    37 labelled examples for RAG
tests/                  pytest tests
outputs/                Generated files (network diagram, eval reports)
```

## Known limitations

- Dummy data: 3 made-up grocers and about 2,900 made-up companies.
- The offline mock LLM is keyword-based; use Qwen for real extraction.
- Qwen3-8B is a small model: it may add an industry or a place the article only mentions in passing (places not in the network show as "not found in the graph"). Check the Monitoring card; RAG is aimed at this.
- The fake-news check has no live fact-checking: it rests on the supplier network, internal consistency and warning signs, so it can miss subtle fakes.
- Alternatives are matched on industry only.
- The Azure code path has not been tested against a live deployment.
- Chains shown are one shortest path per supplier; risk scoring looks at all upstream suppliers.
- Very few HIGH-risk suppliers appear (the 0.60 threshold is hard to reach), and country-wide disruptions leave few or no fully clear replacements.
- The dashboard is a local demo: no login, runs kept in memory.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` (scipy, langfuse, ...) | The install did not finish, or you are in the wrong environment. Activate `capstone` and re-run `pip install -r requirements.txt`. |
| pip says "Operation not permitted", or hangs | The virtual environment is inside OneDrive. Recreate it in your home folder. |
| "No matching distribution found" | Your Python may be too new for that package. Try Python 3.12 or 3.13. |
| Dashboard says "offline mock" but you set up Qwen or Azure | Check `.env` is next to `app.py` and the values are set, with each setting on its own line and nothing after the value. Restart the app. |
| "CA certificate not found" | `QWEN_CA_CERT` must be the exact path to your `.crt` file (a bare file name is looked up in the project folder). Make sure the line has no trailing comment. |
| "Certificate verify failed" | Use the certificate that matches your group's URL (vm04 to vm07). Do not turn verification off. |
| Qwen API error 401 / 404 / 429 / 503 | 401: wrong key or the URL for another group. 404: the base URL must end in `/v1`. 429: someone else is using the same key; wait. 503: server busy (already retried three times); try later. |
| Port already in use | `PORT=9000 python app.py` |
| A run shows an error card | The message is the exception text. Check the key, URL and certificate (or the Azure settings). |
| No "Retrieval (RAG)" card | The variable was not set for the app process. Use `RAG=1 python app.py` (or put `RAG=1` in `.env`) and restart. |
| `FileNotFoundError ... data/rag/exemplars.json` | `--rag-examples` needs that file; restore it. `--ground-entities` alone does not need it. |
| `ValidationError` with `--rag-examples` | An entry in `exemplars.json` does not match the Monitoring schema (for example a `disruption_type` that is not allowed). The error names the field. |
| `tests/test_rag.py` fails after editing examples | The message names the example: too close to an evaluation article, names a graph company, or uses an industry or country not in the graph. |
| No traces in Langfuse | The log says `[langfuse] disabled: ...`: install `langfuse` in the active environment, check both keys and `LANGFUSE_HOST`, and restart the app. |
