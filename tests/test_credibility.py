"""Fake-news check and alternatives-before-approval, on the 6-node hand-built graph.

    pytest tests/test_credibility.py
"""
from scpoc.credibility import check_claims
from scpoc.llm import BaseLLM, MockLLM
from scpoc.pipeline import build_app, resume_review, run_article
from scpoc.schemas import Claim, ClaimsOutput, CSCOOutput, MonitoringOutput, SelfCritiqueOutput
from helpers import tiny


class ScriptedLLM(BaseLLM):
    def __init__(self, claims: ClaimsOutput | None = None):
        super().__init__()
        self.claims = claims or ClaimsOutput(claims=[Claim(text="Conflict in Ukraine halts soy exports.",
                                                            kind="event", countries=["Ukraine"])])

    def complete_json(self, step, system, user, schema, payload=None):
        self.usage["calls"] += 1
        if schema is ClaimsOutput:
            return self.claims
        if schema is MonitoringOutput:
            return MonitoringOutput(reasoning="r", is_disruption=True, disruption_type="geopolitical",
                                    summary="Conflict in Ukraine", countries=["ukraine"])
        if schema is SelfCritiqueOutput:
            return SelfCritiqueOutput(suggestions=[], confidence="medium")
        return CSCOOutput(executive_summary="summary", network_impact="impact", key_recommendations=["x"])


# ---------------------------------------------------------------- evidence and verdict
def test_consistent_claims_are_genuine():
    out = check_claims(ClaimsOutput(claims=[Claim(text="A Co halted output.", kind="event",
                                                  companies=["A Co"], countries=["UK"])]), tiny())
    assert out.verdict == "genuine" and out.score == 0 and out.checks[0].status == "supported"


def test_company_in_the_wrong_country_is_contradicted():
    out = check_claims(ClaimsOutput(claims=[Claim(text="A Co plant burned in Ukraine.", kind="event",
                                                  companies=["A Co"], countries=["Ukraine"])]), tiny())
    assert out.checks[0].status == "contradicted" and "based in UK" in out.checks[0].reasons[0]
    assert out.verdict == "suspicious" and out.score == 2


def test_unknown_company_is_unverifiable_and_counted_once():
    claims = [Claim(text="Zorblax shut down.", kind="event", companies=["Zorblax 777"]),
              Claim(text="Zorblax will not reopen.", kind="impact", companies=["Zorblax 777"])]
    out = check_claims(ClaimsOutput(claims=claims), tiny())
    assert [c.status for c in out.checks] == ["unverifiable", "unverifiable"]
    assert out.score == 1 and out.verdict == "suspicious"


def test_warning_signs_and_contradictions_make_it_likely_fake():
    out = check_claims(ClaimsOutput(claims=[Claim(text="Everything is closed.", companies=["Zorblax 777"])],
                                    contradictions=["plant destroyed vs operating normally"],
                                    red_flags=["sensational wording", "no named source"]), tiny())
    assert out.verdict == "likely_fake" and out.score >= 3


def test_rag_votes_add_suspicion():
    ex = [{"kind": "false_positive"}, {"kind": "false_positive"}, {"kind": "company"}]
    out = check_claims(ClaimsOutput(claims=[Claim(text="x")]), tiny(), examples=ex)
    assert out.score == 1 and any("false positives" in r for r in out.reasons)


# ---------------------------------------------------------------- pipeline
def test_credibility_off_by_default():
    app = build_app(llm=ScriptedLLM(), g=tiny())
    s = run_article(app, "Conflict in Ukraine halts soy exports.", "G Co", "c0")["state"]
    assert s.credibility is None


def test_credibility_runs_first_and_is_in_the_review_payload():
    app = build_app(llm=ScriptedLLM(), g=tiny(), credibility=True)
    r = run_article(app, "Conflict in Ukraine halts soy exports.", "G Co", "c1")
    assert r["paused"] and r["state"].credibility.verdict == "genuine"
    assert r["review"]["credibility"]["verdict"] == "genuine"
    assert [t["step"] for t in r["state"].trace][:2] == ["credibility", "monitoring"]


def test_fake_article_warns_by_default_but_continues():
    fake = ClaimsOutput(claims=[Claim(text="Zorblax shut.", companies=["Zorblax 777"])],
                        red_flags=["sensational wording", "no named source"])
    app = build_app(llm=ScriptedLLM(fake), g=tiny(), credibility=True)
    r = run_article(app, "SHOCKING news", "G Co", "c2")
    assert r["paused"] and r["state"].credibility.verdict == "likely_fake"


def test_block_fake_stops_the_run():
    fake = ClaimsOutput(claims=[Claim(text="Zorblax shut.", companies=["Zorblax 777"])],
                        red_flags=["sensational wording", "no named source"])
    app = build_app(llm=ScriptedLLM(fake), g=tiny(), credibility=True, block_fake=True)
    r = run_article(app, "SHOCKING news", "G Co", "c3")
    s = r["state"]
    assert not r["paused"] and s.status == "flagged_fake" and s.monitoring is None
    assert "credibility check" in s.message


def test_mock_llm_flags_a_sensational_unsourced_article():
    llm = MockLLM({"UK", "Ukraine"}, {"Dairy"}, {"A Co"})
    out = llm.complete_json("credibility", "", "", ClaimsOutput,
                            payload={"article": "SHOCKING: sources say it is all over. Share this!"})
    assert out.red_flags and out.claims


# ---------------------------------------------------------------- alternatives before approval
def _graph_with_replacements():
    g = tiny()
    for n, c, ind in [("A2", "UK", "Dairy"), ("A3", "UK", "Dairy"), ("Q", "Ukraine", "Agriculture")]:
        g.add_node(n, name=f"{n} Co", country=c, industry=ind, is_focal=False)
    g.add_edge("Q", "A3", product="Milk")
    return g


def test_review_payload_shows_alternatives_before_approval():
    app = build_app(llm=ScriptedLLM(), g=_graph_with_replacements())
    r = run_article(app, "x", "G Co", "a1")
    assert r["paused"] and r["state"].status == "running" and r["state"].alternatives == []
    prev = r["review"]["alternatives"]
    top = prev[0]
    assert top["supplier"]["company_id"] == "A" and top["risk_level"] == r["review"]["actions"][0]["risk_level"]
    assert [c["company_id"] for c in top["alternatives"]] == ["A2"]        # A3 excluded: its input is Ukrainian
    assert [t["step"] for t in r["state"].trace][-2:] == ["self_critique", "preview_alternatives"]


def test_approved_alternatives_match_the_preview():
    app = build_app(llm=ScriptedLLM(), g=_graph_with_replacements())
    r = run_article(app, "x", "G Co", "a2")
    forced = [{**r["review"]["actions"][0], "risk_level": "HIGH", "action": "replace"}]
    s = resume_review(app, "a2", {"decision": "edit", "actions": forced})["state"]
    assert [c["company_id"] for c in s.alternatives[0]["alternatives"]] == ["A2"]
    assert "risk_level" not in s.alternatives[0]


def test_reject_after_preview_skips_the_final_step():
    app = build_app(llm=ScriptedLLM(), g=_graph_with_replacements())
    run_article(app, "x", "G Co", "a3")
    s = resume_review(app, "a3", {"decision": "reject"})["state"]
    assert s.status == "rejected" and s.alternatives == [] and s.alternatives_preview
