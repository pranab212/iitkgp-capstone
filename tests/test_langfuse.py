"""Langfuse tracing: off by default, and correct with a fake client (no network, no keys).

    pytest tests/test_langfuse.py
"""
import contextlib

import pytest

from scpoc import observability as obs
from scpoc.llm import BaseLLM
from scpoc.pipeline import build_app, resume_review, run_article
from scpoc.schemas import CSCOOutput, MonitoringOutput, SelfCritiqueOutput
from helpers import tiny


class Obs:
    def __init__(self, name, as_type, kw):
        self.name, self.as_type, self.kw, self.updates = name, as_type, kw, []

    def update(self, **kw):
        self.updates.append(kw)


class FakeLangfuse:
    """Records what the pipeline sends; implements only the calls scpoc uses."""
    def __init__(self):
        self.observations, self.scores, self.events, self.meta, self.flushed = [], [], [], [], 0

    def create_trace_id(self, seed=None):
        return "t" + str(abs(hash(seed)) % 10**8)

    @contextlib.contextmanager
    def start_as_current_observation(self, *, name, as_type="span", **kw):
        o = Obs(name, as_type, kw)
        self.observations.append(o)
        yield o

    def create_score(self, **kw):
        self.scores.append(kw)

    def create_event(self, **kw):
        self.events.append(kw)

    def update_current_span(self, **kw):
        self.meta.append(kw)

    def get_trace_url(self, trace_id=None):
        return f"https://langfuse.example/trace/{trace_id}"

    def flush(self):
        self.flushed += 1

    def get_prompt(self, name, **kw):
        raise RuntimeError("offline")


class ScriptedLLM(BaseLLM):
    name = "scripted"

    def complete_json(self, step, system, user, schema, payload=None):
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += 10
        self.usage["completion_tokens"] += 5
        if schema is MonitoringOutput:
            return MonitoringOutput(reasoning="r", is_disruption=True, disruption_type="geopolitical",
                                    summary="Conflict in Ukraine", countries=["ukraine"])
        if schema is SelfCritiqueOutput:
            return SelfCritiqueOutput(suggestions=[], confidence="medium")
        return CSCOOutput(executive_summary="summary", network_impact="impact", key_recommendations=["x"])


@pytest.fixture
def lf():
    pytest.importorskip("langfuse")           # propagate_attributes comes from the real package
    fake = FakeLangfuse()
    obs.reset()
    obs._STATE.update(client=fake, tried=True)
    yield fake
    obs.reset()


def test_off_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr("scpoc.llm._load_dotenv", lambda *a, **k: None)
    obs.reset()
    assert not obs.enabled() and obs.new_trace("x") == (None, None) and obs.trace_url("t") is None
    obs.score("t", "n", 1)                      # all helpers are silent no-ops
    obs.event("e")
    obs.flush()
    with obs.node_span("monitoring", type("S", (), {})()) as span:
        assert span is None
    obs.reset()


def test_run_creates_one_span_per_step_and_a_generation_per_llm_call(lf):
    app = build_app(llm=ScriptedLLM(), g=tiny())
    r = run_article(app, "Conflict in Ukraine halts soy exports.", "G Co", "L1")
    steps = [o.name for o in lf.observations if o.as_type == "span"]
    assert steps[:5] == ["monitoring", "kg_query", "risk", "visualizer", "csco"]
    gens = [o for o in lf.observations if o.as_type == "generation"]
    assert [g.name for g in gens] == ["llm:monitoring", "llm:csco", "llm:self_critique"]
    assert gens[0].updates[0]["usage_details"] == {"input": 10, "output": 5, "total": 15}
    assert r["state"].trace_id and r["state"].session_id == "L1"
    assert {o.kw["trace_context"]["trace_id"] for o in lf.observations if o.as_type == "span"} == {r["state"].trace_id}


def test_pause_for_review_is_not_an_error_and_trace_survives_resume(lf):
    app = build_app(llm=ScriptedLLM(), g=tiny())
    r = run_article(app, "x", "G Co", "L2")
    tid = r["state"].trace_id
    review = [o for o in lf.observations if o.name == "human_review"][0]
    assert not any(u.get("level") == "ERROR" for u in review.updates)
    resume_review(app, "L2", {"decision": "approve"})
    spans = [o for o in lf.observations if o.as_type == "span"]
    assert {o.kw["trace_context"]["trace_id"] for o in spans} == {tid}      # same trace after the pause
    assert [o.name for o in spans][-1] == "alternatives"
    assert lf.flushed >= 2


def test_scores_for_decision_risk_and_status(lf):
    app = build_app(llm=ScriptedLLM(), g=tiny())
    run_article(app, "x", "G Co", "L3")
    resume_review(app, "L3", {"decision": "approve"})
    got = {s["name"]: s["value"] for s in lf.scores}
    assert got["human_decision"] == "approve" and got["final_status"] == "completed"
    assert "risk_top_score" in got and got["disrupted_suppliers"] == 1
    assert {s["name"]: s["data_type"] for s in lf.scores}["human_decision"] == "CATEGORICAL"


def test_alternatives_spans_carry_the_selection_rule(lf):
    app = build_app(llm=ScriptedLLM(), g=tiny())
    run_article(app, "x", "G Co", "L4")
    meta = [m["metadata"] for m in lf.meta if "suppliers" in m["metadata"]][0]
    assert "same industry" in meta["rule"] and meta["suppliers"][0]["supplier"] == "A Co"


def test_a_failing_langfuse_never_breaks_the_run(lf):
    def boom(**kw):
        raise RuntimeError("langfuse is down")
    lf.create_score = boom
    lf.update_current_span = boom
    app = build_app(llm=ScriptedLLM(), g=tiny())
    r = run_article(app, "x", "G Co", "L5")
    assert r["paused"]


def test_masking_hides_prompts_and_outputs(lf, monkeypatch):
    monkeypatch.setenv("LANGFUSE_MASK_ARTICLES", "1")
    app = build_app(llm=ScriptedLLM(), g=tiny())
    run_article(app, "secret article", "G Co", "L6")
    gen = [o for o in lf.observations if o.as_type == "generation"][0]
    assert gen.kw["input"] == obs.MASK and gen.updates[0]["output"] == obs.MASK


def test_remote_prompt_falls_back_to_local_text(lf):
    assert obs.prompt_text("monitoring_system", "local text") == "local text"
    assert obs.format_prompt("monitoring_system", "hi {industries}", industries="x") == "hi x"


def test_eval_scores_attach_to_the_trace(lf):
    row = {"error": None, "extraction": {"countries": {"tp": 1, "fp": 0, "fn": 1}}, "extraction_exact": False,
           "disrupted": {"match": True}, "status": {"match": True}, "top_supplier": {"match": False},
           "seconds": 1.5, "prompt_tokens": 10, "completion_tokens": 5}
    obs.eval_scores("T1", row)
    got = {s["name"]: s["value"] for s in lf.scores if s["trace_id"] == "T1"}
    assert got["eval_extraction_f1"] == 0.667 and got["eval_extraction_exact"] == 0
    assert got["eval_top_supplier_match"] == 0 and got["eval_tokens"] == 15
