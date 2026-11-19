"""Pipeline tests on the 6-node hand-built graph, with a scripted LLM (no API key needed).

    pytest tests/test_pipeline.py
"""
import networkx as nx

from scpoc import tools
from scpoc.llm import BaseLLM, MockLLM
from scpoc.pipeline import build_app, resume_review, run_article
from scpoc.schemas import CSCOOutput, MonitoringOutput, SelfCritiqueOutput
from helpers import tiny


class ScriptedLLM(BaseLLM):
    """Returns a fixed Monitoring output and a canned CSCO summary."""
    def __init__(self, monitoring: MonitoringOutput):
        super().__init__()
        self.monitoring = monitoring

    def complete_json(self, step, system, user, schema, payload=None):
        self.usage["calls"] += 1
        if schema is MonitoringOutput:
            return self.monitoring
        if schema is SelfCritiqueOutput:
            return SelfCritiqueOutput(suggestions=["Check country specificity."], confidence="medium")
        return CSCOOutput(executive_summary="summary", network_impact="impact",
                          key_recommendations=["do things"])


def ukraine_event(**kw):
    base = dict(reasoning="r", is_disruption=True, disruption_type="geopolitical",
                summary="Conflict in Ukraine", countries=["ukraine"], industries=[], companies=[])
    return MonitoringOutput(**{**base, **kw})


def make_app(monitoring=None, g=None):
    return build_app(llm=ScriptedLLM(monitoring or ukraine_event()), g=g or tiny())


def test_runs_to_review_then_approves():
    app = make_app()
    r = run_article(app, "Conflict in Ukraine halts soy exports.", "G Co", "t1")
    s = r["state"]
    assert r["paused"] and r["review"]["csco"]["executive_summary"] == "summary"
    assert s.focal_id == "G" and s.criteria["countries"] == ["Ukraine"]   # name normalised to graph spelling
    assert s.bfs["disrupted_count"] == 1 and s.risk["top_suppliers"][0]["company_id"] == "A"
    assert s.network_html and s.status == "running"
    r = resume_review(app, "t1", {"decision": "approve"})
    s = r["state"]
    assert not r["paused"] and s.approval == "approved" and s.status == "completed"
    assert s.alternatives == []                                             # nothing is HIGH on this graph
    steps = [t["step"] for t in s.trace]
    assert steps[:5] == ["monitoring", "kg_query", "risk", "visualizer", "csco"]
    assert steps[-1] == "alternatives"


def test_no_exposure_exits_early():
    app = make_app(ukraine_event(countries=["Atlantis"]))
    r = run_article(app, "Something about Atlantis.", "G Co", "t2")
    s = r["state"]
    assert not r["paused"] and s.status == "no_exposure" and "No exposure" in s.message
    assert s.unresolved == ["Atlantis"] and s.risk is None and s.csco is None


def test_not_a_disruption_exits_early():
    app = make_app(ukraine_event(is_disruption=False, disruption_type="none", countries=[]))
    s = run_article(app, "Quarterly results.", "G Co", "t3")["state"]
    assert s.status == "no_exposure" and "not a supply chain disruption" in s.message


def test_reject_stops_before_alternatives():
    app = make_app()
    run_article(app, "x", "G Co", "t4")
    s = resume_review(app, "t4", {"decision": "reject", "reason": "wrong event"})["state"]
    assert s.status == "rejected" and s.message == "wrong event" and s.alternatives == []


def test_edit_changes_summary_and_actions_then_finds_alternatives():
    g = tiny()
    # a clean replacement dairy supplier, and one whose upstream is disrupted
    for n, c in [("A2", "UK"), ("A3", "UK"), ("Q", "Ukraine")]:
        g.add_node(n, name=f"{n} Co", country=c, industry="Dairy" if n != "Q" else "Agriculture",
                   is_focal=False)
    g.add_edge("Q", "A3", product="Milk")
    app = make_app(g=g)
    run_article(app, "x", "G Co", "t5")
    forced = [{"company_id": "A", "name": "A Co", "risk_score": 0.9, "risk_level": "HIGH",
               "action": "replace - source alternative supplier"}]
    r = resume_review(app, "t5", {"decision": "edit", "csco": {"executive_summary": "edited"},
                                  "actions": forced})
    s = r["state"]
    assert s.approval == "edited" and s.csco.executive_summary == "edited" and s.csco.network_impact == "impact"
    alt = s.alternatives[0]
    ids = [a["company_id"] for a in alt["alternatives"]]
    assert alt["supplier"]["company_id"] == "A" and ids == ["A2"]           # A3 excluded: its input is Ukrainian
    assert alt["alternatives"][0]["verified_clear"] and alt["clear_count"] == 1


def test_find_alternatives_ranking_and_fallback():
    g = nx.DiGraph()
    rows = [("F", "UK", "Retail", True), ("S", "Ukraine", "Dairy", False), ("P", "UK", "Dairy", False),
            ("R", "UK", "Dairy", False), ("U", "Ukraine", "Agri", False), ("V", "UK", "Agri", False)]
    for n, c, i, f in rows:
        g.add_node(n, name=n, country=c, industry=i, is_focal=f)
    for s, t in [("S", "F"), ("P", "F"), ("U", "R"), ("V", "P")]:
        g.add_edge(s, t, product="x")
    bfs = tools.supply_chain_bfs("F", countries=["Ukraine"], g=g)
    alt = tools.find_alternatives("S", bfs, g=g)
    assert [a["company_id"] for a in alt["alternatives"]] == ["P"]          # R's input U is disrupted
    g.remove_node("V")                                                       # now P has an empty upstream: still clear
    assert [a["company_id"] for a in tools.find_alternatives("S", bfs, g=g)["alternatives"]] == ["P"]
    g.remove_node("P")                                                       # only R left, not clear -> flagged fallback
    alt = tools.find_alternatives("S", bfs, g=g)
    assert [a["company_id"] for a in alt["alternatives"]] == ["R"] and not alt["alternatives"][0]["verified_clear"]


def test_mock_llm_on_real_data_runs_end_to_end():
    g = tools.get_graph()
    llm = MockLLM({d["country"] for _, d in g.nodes(data=True)},
                  {d["industry"] for _, d in g.nodes(data=True)},
                  {d["name"] for _, d in g.nodes(data=True)})
    app = build_app(llm=llm, g=g)
    r = run_article(app, "War in Ukraine has disrupted sunflower oil and grain exports.",
                    "Northwind Grocers", "real")
    assert r["paused"] and r["state"].bfs["disrupted_count"] > 0
    s = resume_review(app, "real", {"decision": "approve"})["state"]
    assert s.status == "completed"



def test_retry_reruns_monitoring_then_approves():
    """Reviewer rejects with feedback, AI retries monitoring, then reviewer approves."""
    app = make_app()
    r = run_article(app, "Conflict in Ukraine halts soy exports.", "G Co", "t_retry")
    assert r["paused"]
    assert r["review"]["retry_count"] == 0 and r["review"]["max_retries"] == 3
    r = resume_review(app, "t_retry", {"decision": "retry",
                                        "reason": "Ukraine is only mentioned, not actually disrupted."})
    s = r["state"]
    assert r["paused"], "should pause again for the second review"
    assert s.retry_count == 1
    assert s.approval == "retry"  # still in retry state until next decision
    assert r["review"]["retry_count"] == 1
    r = resume_review(app, "t_retry", {"decision": "approve"})
    s = r["state"]
    assert not r["paused"] and s.approval == "approved" and s.status == "completed"
    steps = [t["step"] for t in s.trace]
    assert steps.count("monitoring") == 2, "monitoring should run twice (original + 1 retry)"


def test_retry_max_retries_reached():
    """After max retries the pipeline rejects automatically."""
    app = build_app(llm=ScriptedLLM(ukraine_event()), g=tiny(), max_retries=1)
    r = run_article(app, "x", "G Co", "t_maxretry")
    assert r["paused"]
    r = resume_review(app, "t_maxretry", {"decision": "retry", "reason": "fix it"})
    assert r["paused"], "first retry should work"
    r = resume_review(app, "t_maxretry", {"decision": "retry", "reason": "fix it again"})
    s = r["state"]
    assert not r["paused"] and s.status == "rejected"
    assert "Maximum retries" in s.message


def test_self_critique_runs_before_review():
    """Self-critique node runs between csco and human_review, populating ai_suggestions."""
    app = make_app()
    r = run_article(app, "Conflict in Ukraine halts soy exports.", "G Co", "t_sc")
    assert r["paused"]
    assert r["review"]["ai_suggestions"] == ["Check country specificity."]
    assert r["review"]["ai_confidence"] == "medium"
    steps = [t["step"] for t in r["state"].trace]
    assert "self_critique" in steps
    assert steps.index("csco") < steps.index("self_critique")  # human_review not in trace yet (paused at interrupt)


def test_edit_extraction_skips_llm():
    """edit_extraction updates monitoring in-place and re-runs from kg_query (skipping LLM)."""
    app = make_app()
    run_article(app, "Conflict in Ukraine halts soy exports.", "G Co", "t_ex")
    r = resume_review(app, "t_ex", {"decision": "edit_extraction",
                                     "monitoring": {"countries": ["Ukraine"],
                                                    "industries": ["Agriculture"]}})
    s = r["state"]
    assert r["paused"], "should re-run kg→risk→vis→csco→critique→review and pause again"
    assert "Agriculture" in s.monitoring.industries
    steps = [t["step"] for t in s.trace]
    monitoring_count = steps.count("monitoring")
    assert monitoring_count == 1, "monitoring should NOT re-run; only kg_query onwards"
    kg_count = steps.count("kg_query")
    assert kg_count == 2, "kg_query should run twice (original + after edit)"
    r2 = resume_review(app, "t_ex", {"decision": "approve"})
    assert r2["state"].status == "completed"


def test_self_contradicting_monitoring_exits_early():
    # model says "is_disruption" but also type "none": do not trust it, even if suppliers match
    app = make_app(ukraine_event(disruption_type="none"))
    s = run_article(app, "x", "G Co", "t_contra")["state"]
    assert s.status == "no_exposure" and s.csco is None
