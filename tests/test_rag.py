"""Retrieval tests: BM25, example store, entity grounding, the corpus itself, and the pipeline flags.

    pytest tests/test_rag.py
"""
import json

from scpoc import rag, tools
from scpoc.evalset import load as load_eval
from scpoc.pipeline import build_app, resume_review, run_article
from scpoc.schemas import MonitoringOutput
from helpers import tiny
from test_pipeline import ScriptedLLM, ukraine_event


# ------------------------------------------------------------------ BM25 / tokenizer
def test_tokenize_strips_stopwords_and_plurals():
    assert rag.tokenize("The strikes at the Ports") == ["strike", "port"]
    assert rag.tokenize("industries") == ["industry"]


def test_bm25_ranks_the_relevant_document_first():
    bm = rag.BM25(["dock workers strike at the port", "earthquake damages factories", "sugar export ban"])
    s = bm.scores("port strike by dockworkers")
    assert s.index(max(s)) == 0 and s[2] == 0


# ------------------------------------------------------------------ example store
def _ex(i, kind, article):
    return {"id": i, "kind": kind, "article": article,
            "answer": {"reasoning": "r", "is_disruption": True, "disruption_type": "other", "summary": "s",
                       "countries": [], "industries": [], "companies": []}}


def test_search_limits_examples_per_kind():
    store = rag.ExemplarStore([_ex("a", "country", "port strike"), _ex("b", "country", "port strike again"),
                               _ex("c", "country", "port strike once more"), _ex("d", "industry", "port shortage")])
    got = store.search("port strike", k=3, max_per_kind=2)
    assert [e["id"] for e in got] == ["a", "b", "d"] and all("score" in e for e in got)


def test_format_examples_includes_answer_json():
    text = rag.format_examples([_ex("a", "country", "Floods in Kenya.")])
    assert "Floods in Kenya." in text and '"is_disruption": true' in text
    assert rag.format_examples([]) == ""


# ------------------------------------------------------------------ the real corpus
def test_corpus_is_valid_and_uses_graph_vocabulary():
    g = tools.get_graph()
    store = rag.ExemplarStore.load()                       # validates every answer against MonitoringOutput
    countries = {d["country"] for _, d in g.nodes(data=True)}
    industries = {d["industry"] for _, d in g.nodes(data=True)}
    names = {d["name"].lower() for _, d in g.nodes(data=True)}
    assert len({e["id"] for e in store.items}) == len(store.items)
    assert {e["kind"] for e in store.items} == {"company", "country", "industry", "country_industry", "false_positive"}
    for e in store.items:
        a = e["answer"]
        assert set(a["industries"]) <= industries, e["id"]
        assert set(a["countries"]) <= countries, e["id"]
        assert not any(c.lower() in names for c in a["companies"]), f"{e['id']} names a real graph company"


def test_corpus_does_not_leak_the_evaluation_set():
    store = rag.ExemplarStore.load()
    evals = load_eval()
    eval_companies = {c.lower() for a in evals for c in a["truth"]["companies"]}
    for e in store.items:
        ex = set(rag.tokenize(e["article"]))
        assert not any(c.lower() in e["article"].lower() for c in eval_companies), e["id"]
        for a in evals:
            ev = set(rag.tokenize(a["article"]))
            assert len(ex & ev) / len(ex | ev) < 0.4, f"{e['id']} is too close to eval article {a['id']}"


# ------------------------------------------------------------------ entity grounding
def test_grounding_finds_exact_and_misspelled_names_only():
    g = tools.get_graph()
    idx = rag.EntityIndex(g)
    hits = idx.find("Production at Kave Mining 433 has stopped. Mining firms elsewhere are fine.")
    assert [h["company_id"] for h in hits] == ["C00433"] and hits[0]["industry"] == "Mining"
    typo = idx.find("A fire hit the Kavee Mining 433 site.")
    assert [h["company_id"] for h in typo] == ["C00433"] and typo[0]["score"] < 1
    assert idx.find("An earthquake struck central Turkey overnight.") == []


def test_grounding_marks_the_focal_grocer():
    idx = rag.EntityIndex(tools.get_graph())
    hits = idx.find("Northwind Grocers reported record profit.")
    assert len(hits) == 1 and hits[0]["is_focal"]
    assert "not a supplier" in rag.format_entities(hits)


def test_format_entities_warns_about_industry_and_country():
    text = rag.format_entities([{"name": "Kave Mining 433", "industry": "Mining", "country": "Chile",
                                 "is_focal": False}])
    assert "Do not also list its industry (Mining)" in text and "(Chile)" in text
    assert rag.format_entities([]).endswith("none.")


# ------------------------------------------------------------------ pipeline wiring
class RecordingLLM(ScriptedLLM):
    def complete_json(self, step, system, user, schema, payload=None):
        if schema is MonitoringOutput:
            self.monitoring_user = user
        return super().complete_json(step, system, user, schema, payload)


def test_flags_off_keeps_the_original_graph():
    llm = RecordingLLM(ukraine_event())
    app = build_app(llm=llm, g=tiny())
    s = run_article(app, "Conflict in Ukraine.", "G Co", "off")["state"]
    assert "retrieve" not in app.get_graph().nodes and s.retrieved == {}
    assert llm.monitoring_user.startswith("Article:")


def test_flags_on_add_retrieved_context_to_the_prompt():
    llm = RecordingLLM(ukraine_event())
    store = rag.ExemplarStore([_ex("ex_war", "country", "War in Ukraine stops soy exports"),
                               _ex("ex_sugar", "industry", "Sugar export ban")])
    app = build_app(llm=llm, g=tiny(), rag_examples=True, ground_entities=True, k_examples=1,
                    exemplar_store=store)
    r = run_article(app, "War in Ukraine hits A Co soy deliveries.", "G Co", "on")
    s = r["state"]
    assert [e["id"] for e in s.retrieved["examples"]] == ["ex_war"]
    assert [e["company_id"] for e in s.retrieved["entities"]] == ["A"]
    assert "Solved examples" in llm.monitoring_user and "A Co: industry Dairy" in llm.monitoring_user
    assert llm.monitoring_user.rstrip().endswith('"""')                  # the article itself comes last
    steps = [t["step"] for t in s.trace]
    assert steps[:2] == ["retrieve", "monitoring"]
    assert resume_review(app, "on", {"decision": "approve"})["state"].status == "completed"


def test_state_with_retrieval_is_json_serialisable():
    app = build_app(llm=ScriptedLLM(ukraine_event()), g=tiny(), ground_entities=True)
    s = run_article(app, "A Co hit.", "G Co", "ser")["state"]
    json.dumps(s.retrieved)
