================================================================================
  SUPPLY CHAIN DISRUPTION MONITOR - README
  Agentic POC of AlMahri, Xu & Brintrup: "Automating Supply Chain Disruption
  Monitoring via an Agentic AI Approach"
================================================================================

  Author      : Rohan Dawar
  Python      : 3.10 or higher (tested on 3.14)
  Data        : dummy only (made-up companies), in-memory NetworkX graph, no database

--------------------------------------------------------------------------------
TABLE OF CONTENTS
--------------------------------------------------------------------------------

  1.  What is this?
  2.  Files in this package
  3.  Prerequisites
  4.  Setting up a virtual environment
  5.  Installing dependencies
  6.  Configuration (Qwen3-8B API, or Azure OpenAI)
  7.  Running the dashboard
  8.  Using the dashboard
  9.  Running from the command line
  10. Running the tests
  11. Retrieval (RAG) for the Monitoring step
  12. How it works
  13. Risk score
  14. Known limitations
  15. Troubleshooting
  16. Stopping the app

--------------------------------------------------------------------------------
1. WHAT IS THIS?
--------------------------------------------------------------------------------

Give it a news article and a grocer. It decides whether the article reports a supply
chain disruption, finds which suppliers (Tier-1 to Tier-4) of that grocer are hit,
scores the Tier-1 suppliers, writes an executive summary, waits for a person to approve
it, then suggests replacement suppliers for the HIGH-risk ones.

Two steps use an LLM (reading the article, writing the summary). Everything that
produces numbers (supplier matching, risk scores, actions, alternatives) is plain
Python, so results are repeatable.

app.py is a single file, standard library only: it embeds the dashboard HTML and a
small JSON API, the same structure as the Hermes dashboard.

--------------------------------------------------------------------------------
2. FILES IN THIS PACKAGE
--------------------------------------------------------------------------------

  app.py               The dashboard (web server + embedded page)
  cli.py               Command line: data | demo | run
  scpoc/               The library
    graph.py             load the network, tiers, find suppliers
    tools.py             the deterministic tools (entity resolution, BFS, risk,
                         actions, alternatives, network diagram)
    schemas.py           shared State and the strict-JSON outputs of the LLM steps
    prompts.py           prompts for the Monitoring and CSCO steps
    llm.py               Qwen3-8B client, Azure OpenAI client, offline mock LLM
    pipeline.py          the LangGraph pipeline (human review via interrupt())
    rag.py               optional retrieval for Monitoring (BM25 examples, entity grounding)
    datagen.py           makes the dummy network
    evalset.py           the 20 evaluation articles and their expected answers
    evaluate.py          runs them and reports precision/recall/F1, matches, time, tokens
  tests/               pytest tests (no API key needed)
  data/                companies.csv, supplies.csv (dummy network, 3 grocers),
                       eval_articles.json (evaluation set),
                       rag/exemplars.json (labeled examples for --rag-examples)
  outputs/             generated files (network diagram, demo JSON)
  requirements.txt     dependencies to run
  requirements-dev.txt extra dependency for the tests (pytest)
  .env.example         template for the LLM settings (copy to .env)

--------------------------------------------------------------------------------
3. PREREQUISITES
--------------------------------------------------------------------------------

  - Python 3.10 or newer      python3 --version
  - Internet access for pip
  - For the real LLM: your group's API key and CA certificate for the course
    Qwen/Qwen3-8B inference API (Track B guide). Without a key the app uses an
    offline keyword-based mock LLM. (An Azure OpenAI deployment also works.)

--------------------------------------------------------------------------------
4. SETTING UP A VIRTUAL ENVIRONMENT
--------------------------------------------------------------------------------

IMPORTANT: create the virtual environment OUTSIDE OneDrive. OneDrive syncing can
block or badly slow pip inside a synced folder ("Operation not permitted"). Keep
the code here, put the environment in your home folder.

  macOS / Linux
      cd path/to/Capstone
      mkdir -p ~/venvs
      python3 -m venv ~/venvs/capstone
      source ~/venvs/capstone/bin/activate
      which python        # should point to ~/venvs/capstone/bin/python

  Alternative with conda (macOS / Linux / Windows)
      conda create -n capstone python=3.12 -y
      conda activate capstone

  Windows (PowerShell)
      cd path\to\Capstone
      python -m venv $HOME\venvs\capstone
      $HOME\venvs\capstone\Scripts\Activate.ps1

If PowerShell blocks the activate script, run once:
      Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Your prompt should now start with (capstone). Next time you only need the
"activate" line. Leave the environment with:  deactivate

--------------------------------------------------------------------------------
5. INSTALLING DEPENDENCIES
--------------------------------------------------------------------------------

With the environment active:

      python -m pip install --upgrade pip
      pip install -r requirements.txt

For the tests also run:

      pip install -r requirements-dev.txt

Check the install:

      python -c "import pandas, networkx, scipy, pydantic, langgraph; print('ok')"

--------------------------------------------------------------------------------
6. CONFIGURATION (QWEN3-8B API, OR AZURE OPENAI)
--------------------------------------------------------------------------------

Skip this to use the offline mock (fine for trying the flow, but it is a crude
keyword matcher, not a measure of extraction quality).

Qwen3-8B (course Track B API):

  1. Put your group's CA certificate in the project folder (for example
     vm05-ca.crt). The app needs its exact path and verifies HTTPS with it.
  2. Copy the template and fill it in:

         cp .env.example .env          (Windows: copy .env.example .env)

         QWEN_API_BASE=https://103.42.50.41:1033/v1    <- the URL for YOUR group
         QWEN_API_KEY=<your-group-key>
         QWEN_CA_CERT=vm05-ca.crt                       <- file name or full path
         QWEN_MODEL=Qwen/Qwen3-8B

     Group -> URL and certificate: 001-015 :1031 vm07-ca.crt, 016-030 :1032
     vm06-ca.crt, 031-045 :1033 vm05-ca.crt, 046-065 :1034 vm04-ca.crt.
     The base URL already ends in /v1; do not add /chat/completions.

  3. Check the connection (replace the key and certificate):

         curl --cacert vm05-ca.crt -H "Authorization: Bearer <key>" \
              https://103.42.50.41:1033/v1/models

Limits of this API, handled by the app: one active request per key (calls are queued
one at a time; do not run two things with the same key at once), 60 requests per
minute, 1024 output tokens per call, 8192 tokens input plus output. On 429 or 503 the
app retries after 2, 4 and 8 seconds, then reports the error.

Azure OpenAI (optional, used only if QWEN_API_KEY is not set): set
AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT and
AZURE_OPENAI_API_VERSION in .env (see .env.example) and install the openai package.

.env holds secrets. It is in .gitignore; do not commit it or paste the key into
screenshots, tickets or chat. The dashboard shows "LLM: Qwen3-8B" (or Azure OpenAI, or
offline mock) in the top right so you can see which is active.

--------------------------------------------------------------------------------
7. RUNNING THE DASHBOARD
--------------------------------------------------------------------------------

      python app.py

Open http://localhost:8080. To use another port:  PORT=9000 python app.py
(Windows PowerShell: $env:PORT=9000; python app.py). The server listens on
127.0.0.1 only (this machine). Set HOST=0.0.0.0 to allow other machines.

To turn on retrieval (RAG) for the Monitoring step (section 11):

      RAG=1 python app.py                 both retrieval parts
      RAG_EXAMPLES=1 python app.py        similar solved examples only
      GROUND_ENTITIES=1 python app.py     company grounding only

(Windows PowerShell: $env:RAG=1; python app.py.) The same variables can go in .env.
The top right then shows, for example, "LLM: Qwen3-8B, RAG: rag_examples+ground_entities".

--------------------------------------------------------------------------------
8. USING THE DASHBOARD
--------------------------------------------------------------------------------

  1. Pick a ready-made scenario (Ukraine conflict, plant fire at a packaging
     supplier, port strike in Vietnam, or the storm-in-Iceland false positive) or
     paste your own article. Choose the grocer.
  2. Click "Run analysis". Each step appears as it finishes, with its time and
     token use: Monitoring, Knowledge graph query, Risk scores, Network diagram,
     CSCO summary. With RAG on, a "Retrieval (RAG)" card comes first, showing which
     solved examples were picked (with their BM25 score) and which companies the
     article names.
  3. Human review: edit the summary or any supplier's action if you like, then
     Approve, "Approve with my edits", or Reject.
  4. After approval, "Alternative sourcing" lists replacement candidates for each
     HIGH-risk supplier. "chain clear = no" means the candidate's own suppliers
     are also exposed; this happens often on this dummy data.
  5. If nothing in the network is hit (or the article is not a disruption), the
     run stops with "no exposure" and skips the rest.

Runs are kept in memory, so restarting the app clears them.

--------------------------------------------------------------------------------
9. RUNNING FROM THE COMMAND LINE
--------------------------------------------------------------------------------

      python cli.py data                      regenerate the dummy network
      python cli.py demo                      tools only (no LLM), Ukraine, Northwind
      python cli.py demo --focal alpenmarkt --industry Packaging
      python cli.py run --text "War in Ukraine halts grain exports." --auto approve
      python cli.py run --article article.txt           asks you to approve
      python cli.py run --rag --text "..." --auto approve   with retrieval (section 11)

Generated files go to outputs/.

--------------------------------------------------------------------------------
10. RUNNING THE TESTS
--------------------------------------------------------------------------------

      pytest

The tests use a 6-node hand-built network where every number can be checked by hand,
and a scripted LLM, so no API key or internet is needed.

EVALUATION (needs the real LLM to be meaningful)

      python cli.py eval                      all 20 articles (about 2.5 minutes with Qwen)
      python cli.py eval --ids co_t1,fp_chips only some
      python cli.py eval --mock               plumbing check with the offline mock
      python cli.py eval-build                regenerate data/eval_articles.json

The set has 15 articles that hit the network (4 single-company events at Tier-1 to
Tier-4, 5 country events, 3 industry events, 3 country-and-industry events) and 5 false
positives that must end with "no exposure". Each has a hand-written truth (what a careful
reader would extract); the expected disrupted count, top Tier-1 supplier and top HIGH
supplier are computed from that truth with the deterministic tools. Because the tools
are deterministic, any mismatch traces back to what the LLM extracted.

It reports extraction precision/recall/F1 (countries, industries, companies; places or
companies named but not in the network count as false positives), whether the network
query and risk results match, false-positive handling, and time and tokens per article.
Cost: pass --price-in and --price-out (USD per 1K tokens); the course API has no price,
so cost shows as 0. Results are saved to outputs/eval_report.txt and eval_results.json.
The LLM can give slightly different answers between runs even at temperature 0.

To evaluate with retrieval turned on, and to compare configurations, see section 11.

--------------------------------------------------------------------------------
11. RETRIEVAL (RAG) FOR THE MONITORING STEP
--------------------------------------------------------------------------------

Optional, OFF by default. With it off, the pipeline and the results are exactly as
before.

WHY

The baseline evaluation shows the Monitoring step (Qwen3-8B) mostly fails by listing
too much: it adds a named company's own industry (one company becomes 484 disrupted
suppliers), adds sectors an article only mentions in passing, or adds the countries that
merely buy the goods. Retrieval gives the model two kinds of help before it answers.

THE TWO PARTS (each has its own switch, so their effect can be measured separately)

  1. Similar solved examples  (--rag-examples)
     data/rag/exemplars.json holds 37 short labeled articles: 8 single-company events,
     8 country events, 6 industry events, 6 country-and-industry events and 9 false
     positives (forecasts, events that are over, minor incidents, routine news). For
     each new article a BM25 keyword search picks the 3 most similar ones (at most 2 of
     the same kind, so the model sees a contrast) and puts them, with their correct
     JSON answers, in the prompt as extra examples.

  2. Company grounding  (--ground-entities)
     Companies from the knowledge graph that the article names (exact name, or a near
     miss such as "Kavee Mining 433") are listed in the prompt with their industry and
     country, plus a reminder: listing the company already covers it; do not also add
     its industry or country unless the whole sector or country is hit. A grocer named
     in the article is marked as "the retailer we monitor, not a supplier".
     Only companies the article names are shown, never the grocer's supplier network,
     so the model is not nudged into "finding" exposure that is not there.

  The search is plain Python (no extra packages, no vector database, no embeddings
  API) and deterministic: the same article always retrieves the same context. It adds
  about 500 prompt tokens per article; the prompt stays well inside Qwen's 8192.

TURNING IT ON

  Command line (run and eval take the same switches):

      --rag-examples       part 1 only
      --ground-entities    part 2 only
      --rag                both
      --k 3                number of examples to retrieve (default 3)

      python cli.py run --rag --text "Workers at Kalun Group 551 walked out..." --auto approve

  The run prints what was retrieved before the Monitoring result:

      [retrieve] examples: ['ex_co_two_companies (8.1)', 'ex_ctry_korea_general_strike (6.11)', ...]
      [retrieve] companies named: ['Kalun Group 551']

  Dashboard: RAG=1 python app.py (or RAG_EXAMPLES=1 / GROUND_ENTITIES=1), see section 7.

  In code:

      from scpoc.pipeline import build_app
      app = build_app(rag_examples=True, ground_entities=True, k_examples=3)

EVALUATING IT (needs the real LLM; about 2.5 minutes per configuration with Qwen)

  Run one configuration at a time (the course API allows one request per key):

      python cli.py eval                     -> eval_report.txt / eval_results.json
      python cli.py eval --rag-examples      -> eval_report_rag-examples.txt / .json
      python cli.py eval --ground-entities   -> eval_report_ground-entities.txt / .json
      python cli.py eval --rag               -> eval_report_rag-examples_ground-entities.txt / .json

  Each configuration writes its own files in outputs/, so runs do not overwrite each
  other (or your saved eval_results_baseline.json). Then compare them side by side;
  the first file is the reference:

      python cli.py eval-compare outputs/eval_results_baseline.json \
          outputs/eval_results_rag-examples.json \
          outputs/eval_results_ground-entities.json \
          outputs/eval_results_rag-examples_ground-entities.json

  This prints extraction F1 (overall, countries, industries, companies), industry
  precision, exact extractions, network/risk matches, false positives handled, prompt
  tokens and time, then lists the articles that became exact (or stopped being exact)
  compared with the reference.

  With retrieval on, the report also has a "Retrieval" block:

      articles with a retrieved example of the same kind   e.g. 19/20
      entity grounding P / R    did grounding find the companies the article names
                                (P = R = 1.000 on the current set)

  "eval --mock --rag" only checks that everything runs: the mock LLM ignores the
  retrieved context, so its scores say nothing about retrieval.

ADDING OR EDITING EXAMPLES

  Each entry in data/rag/exemplars.json looks like this:

      {"id": "ex_co_fire_plastics", "kind": "company",
       "article": "A fire broke out overnight at Ostrel Plastics 870's resin plant ...",
       "answer": {"reasoning": "...", "is_disruption": true,
                  "disruption_type": "industrial_accident", "summary": "...",
                  "countries": [], "industries": [], "companies": ["Ostrel Plastics 870"]}}

  kind is one of: company, country, industry, country_industry, false_positive.
  answer must match the Monitoring output schema (scpoc/schemas.py); industries must
  come from the graph's industry list and countries from the graph's countries.

  Rules, checked by tests/test_rag.py (run pytest after editing):
    - never copy or paraphrase an evaluation article (data/eval_articles.json); each
      example must share less than 40% of its words with every evaluation article;
    - never use an evaluation company, or any real company from the graph: use made-up
      names (the grounding part handles the real ones);
    - ids must be unique.

  Keep examples short (one to three sentences) and make the reasoning say which rule
  applies ("Germany is only a buyer", "one firm is not a whole sector").

TUNING (in scpoc/rag.py)

  ExemplarStore.search(k=3, max_per_kind=2)   how many examples, and how many of one kind
  EntityIndex(min_score=0.85)                 how close a name must be to count as a match
  EntityIndex(max_df=25)                      name words shared by more names than this
                                              (e.g. "Mining") are not used for lookup

CAVEATS

  - The examples were written after looking at the baseline failures. They do not
    copy the evaluation articles, but they target the same kinds of mistake, so an
    improvement may partly fit this evaluation set. Confirm it on a few new articles
    that nobody looked at while writing the examples.
  - BM25 matches words, not meaning: an article worded very differently from every
    example may get less relevant examples (for example fp_retrospective does not
    retrieve the "recovered after the floods" example).
  - Grounding only finds companies that are in the graph. Companies outside the
    network are still extracted by the LLM and show as "not found in the graph".

--------------------------------------------------------------------------------
12. HOW IT WORKS
--------------------------------------------------------------------------------

  [retrieve]  ->  monitoring (LLM)  ->  kg_query  ->  [nothing disrupted? stop: "no exposure"]
     ->  risk  ->  visualizer  ->  csco (LLM)  ->  human review (pause)
     ->  alternatives

  The run also stops with "no exposure" when the model's own verdict is "not a
  disruption" (is_disruption false, or type "none"), even if some names matched.

  retrieve      only with RAG on (section 11): picks similar solved examples and looks
                up the companies the article names; no LLM
  monitoring    reads the article; returns disruption type, countries, industries,
                companies as strict JSON
  kg_query      resolves names to graph entries, then finds every disrupted supplier
                (Tier-1..4) and its chain down to the grocer, with products
  risk          scores each exposed Tier-1 supplier and picks the action
  visualizer    writes the tiered network diagram
  csco          writes the executive summary (the actions themselves are fixed by
                thresholds, not the LLM)
  human review  the run pauses (LangGraph interrupt) until someone decides
  alternatives  for each HIGH-risk supplier: same industry, outside the disrupted
                area, and its own upstream (Tier-2..4) has no disrupted company

--------------------------------------------------------------------------------
13. RISK SCORE
--------------------------------------------------------------------------------

  score = 0.35 x breadth + 0.25 x dependency + 0.20 x criticality
        + 0.10 x centrality + 0.10 x depth       (each part is 0 to 1)

  HIGH >= 0.60 -> replace      MEDIUM 0.45-0.59 -> increase monitoring
  LOW  <  0.45 -> maintain standard operations

  breadth       disrupted suppliers upstream of the Tier-1; closer ones count more
                (1/(hops+1) each); the total x is squashed to x/(x+3)
  dependency    share of the Tier-1's direct suppliers that lead to a disruption
                (1 if the Tier-1 itself is disrupted)
  criticality   highest PageRank or degree centrality among its disrupted suppliers
  centrality    how connected the Tier-1 is, compared with the grocer's other Tier-1s
  depth         deepest disrupted tier / 4

The paper gives the weights and thresholds but not the formulas for breadth and
dependency; those two are this POC's own choices.

--------------------------------------------------------------------------------
14. KNOWN LIMITATIONS
--------------------------------------------------------------------------------

  - Dummy data: 3 made-up grocers and about 2,900 made-up companies.
  - The offline mock LLM is keyword-based; use Qwen for real extraction.
  - Qwen3-8B is a small model: it may add an industry or a place the article only
    mentions in passing (places not in the network show as "not found in the graph").
    The prompt asks it to be strict, but check the Monitoring card. Retrieval
    (section 11) is aimed at this; see its caveats.
  - Retrieval covers the Monitoring step only; the CSCO summary uses no retrieved
    documents.
  - The Azure code path is written but has not been tested against a live
    deployment.
  - Chains shown are one shortest path per supplier; risk scoring looks at all
    upstream suppliers.
  - Very few HIGH-risk suppliers appear (the 0.60 threshold is hard to reach).
  - Country-wide disruptions leave few or no fully clear replacement candidates.
  - The dashboard is a local demo: no login, runs kept in memory.

--------------------------------------------------------------------------------
15. TROUBLESHOOTING
--------------------------------------------------------------------------------

  ModuleNotFoundError: scipy (or another package)
      The install did not finish. Re-run: pip install -r requirements.txt

  pip says "Operation not permitted", or hangs
      The virtual environment is inside OneDrive. Recreate it in your home folder
      (section 4).

  No matching distribution found for a package
      Your Python may be too new for that package. Try Python 3.12 or 3.13.

  Dashboard says "offline mock" but you set up Qwen or Azure
      Check .env is in the project folder (next to app.py) and the values are set.
      Restart the app after editing it.

  "CA certificate not found"
      QWEN_CA_CERT must be the exact path to your .crt file (a bare file name is
      looked up in the project folder).

  Certificate verify failed
      Use the certificate that matches your group's URL (vm04 to vm07). Do not turn
      verification off.

  Qwen API error 401 / 404 / 429 / 503
      401: wrong key or the URL for another group. 404: the base URL must end in /v1.
      429: someone else is using the same key right now, wait. 503: server busy; the
      app already retried three times, so try again later.

  Port already in use
      Run on another port:  PORT=9000 python app.py

  A run shows an error card
      The message is the exception text. Check the key, URL and certificate (or the
      Azure endpoint, deployment name and API version).

  No "Retrieval (RAG)" card / the top right does not mention RAG
      The variable was not set for the app process. Use RAG=1 python app.py (or put
      RAG=1 in .env) and restart the app.

  FileNotFoundError ... data/rag/exemplars.json
      --rag-examples needs that file. Restore it from the package; --ground-entities
      alone does not need it.

  ValidationError when starting with --rag-examples
      An entry in data/rag/exemplars.json does not match the Monitoring schema (for
      example a disruption_type that is not in the list). The error names the field.

  tests/test_rag.py fails after editing the examples
      The message names the example: it is too close to an evaluation article, names a
      company from the graph, or uses an industry or country that is not in the graph.

--------------------------------------------------------------------------------
16. STOPPING THE APP
--------------------------------------------------------------------------------

Press Ctrl+C in the terminal running python app.py.
