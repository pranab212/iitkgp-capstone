"""Phase 2 - the LangGraph pipeline.

    [retrieve] -> [credibility] -> monitoring (LLM) -> kg_query -> [no exposure? stop] -> risk -> visualizer
        -> csco (LLM) -> self_critique -> preview_alternatives -> human_review (interrupt)
        -> alternatives -> END
                                         \\-> rejected -> END

[retrieve] runs only with build_app(rag_examples=True and/or ground_entities=True); see rag.py.
LLM steps return strict JSON (Pydantic). Tool steps are plain Python. The human review
pauses the run with interrupt(); a MemorySaver checkpointer lets it resume.

    from pipeline import build_app, run_article, resume_review
    app = build_app()
    result = run_article(app, article_text, "Northwind Grocers", thread_id="t1")
    # result["paused"] -> result["review"] has the CSCO summary and actions
    result = resume_review(app, "t1", {"decision": "approve"})
"""
from __future__ import annotations

import functools
import time

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import credibility as cred
from . import observability as obs
from . import tools
from .llm import BaseLLM, get_llm
from .prompts import (CLAIMS_SYSTEM, CLAIMS_USER, CSCO_RAG_CONTEXT, CSCO_SYSTEM, CSCO_USER,
                      CSCO_USER_WITH_CONTEXT,
                      MONITORING_RETRY_CONTEXT,
                      MONITORING_SYSTEM, MONITORING_USER,
                      MONITORING_USER_WITH_CONTEXT,
                      SELF_CRITIQUE_SYSTEM, SELF_CRITIQUE_USER)
from .schemas import ClaimsOutput, CSCOOutput, MonitoringOutput, SelfCritiqueOutput, State


def _memory_saver():
    """MemorySaver that explicitly allows our state classes (newer LangGraph warns otherwise)."""
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        return MemorySaver(serde=JsonPlusSerializer(
            allowed_msgpack_modules=[("scpoc.schemas", "MonitoringOutput"), ("scpoc.schemas", "CSCOOutput"),
                                     ("scpoc.schemas", "State"),
                                     ("scpoc.schemas", "CredibilityOutput"), ("scpoc.schemas", "ClaimCheck")]))
    except (ImportError, TypeError):      # older LangGraph: no such option
        return MemorySaver()


def _timed(step):
    """Record seconds and LLM tokens used by a node into state.trace."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state: State):
            before, t0 = wrapper.llm.snapshot(), time.perf_counter()
            with obs.node_span(step, state) as span:
                update = fn(state)
                obs.note(span, output={k: v for k, v in update.items()
                                       if k in ("status", "message", "approval", "retry_count")} or
                         {"updated": sorted(update)})
            after = wrapper.llm.snapshot()
            update["trace"] = [{"step": step, "seconds": round(time.perf_counter() - t0, 3),
                                "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                                "completion_tokens": after["completion_tokens"] - before["completion_tokens"]}]
            return update
        return wrapper
    return deco


MAX_RETRIES = 3

# how replacement candidates are chosen (also shown in the dashboard and README)
ALT_RULES = ("same industry as the supplier; outside the disrupted countries/industries/companies; "
             "own upstream (Tier-2..4) has no disrupted company; ranked by exposure score then smaller "
             "upstream network then name; exposed candidates only if no clear one exists; top 5")


def build_app(llm: BaseLLM | None = None, g=None, checkpointer=None, *,
              rag_examples: bool = False, ground_entities: bool = False, k_examples: int = 3,
              exemplar_store=None, max_retries: int = MAX_RETRIES,
              credibility: bool = False, block_fake: bool = False, preview_n: int = 5):
    """rag_examples: add the most similar labeled examples to the Monitoring prompt.
    ground_entities: add the graph records of companies the article names.
    With both off the graph is exactly the original one (no retrieve node).
    max_retries: how many times the reviewer can ask the AI to retry (default 3).
    credibility: add a fake-news check (claim extraction + evidence) before Monitoring.
    block_fake: with credibility on, stop the run when the verdict is "likely_fake" (default: warn only).
    preview_n: how many top suppliers get replacement candidates shown to the reviewer before approval."""
    g = g or tools.get_graph()
    llm = llm or get_llm(g)
    use_rag = rag_examples or ground_entities
    store = entity_index = None
    if use_rag:
        from . import rag
        store = (exemplar_store or rag.ExemplarStore.load()) if rag_examples else None
        entity_index = rag.EntityIndex(g) if ground_entities else None
    industries = sorted({d["industry"] for _, d in g.nodes(data=True)})
    vocab = {k: {d[k].lower(): d[k] for _, d in g.nodes(data=True)} for k in ("country", "industry")}

    def node(step):
        def deco(fn):
            w = _timed(step)(fn)
            w.llm = llm
            return w
        return deco

    # ---- 0. Retrieval (tools, optional): similar solved examples + companies the article names
    @node("retrieve")
    def retrieve(state: State):
        out = {}
        if store:
            out["examples"] = store.search(state.article, k=k_examples)
        if entity_index:
            out["entities"] = entity_index.find(state.article)
        return {"retrieved": out}

    # ---- 0b. Credibility (LLM extracts claims, tools check them): genuine / suspicious / likely fake
    @node("credibility")
    def credibility_check(state: State):
        claims = llm.complete_json("credibility", obs.prompt_text("claims_system", CLAIMS_SYSTEM),
                                   CLAIMS_USER.format(article=state.article),
                                   ClaimsOutput, payload={"article": state.article})
        out = cred.check_claims(claims, g, examples=state.retrieved.get("examples"))
        obs.score(state.trace_id, "credibility_verdict", out.verdict, "CATEGORICAL")
        obs.score(state.trace_id, "credibility_score", out.score)
        return {"credibility": out}

    def after_credibility(state: State):
        return "flagged_fake" if block_fake and state.credibility.verdict == "likely_fake" else "monitoring"

    @node("flagged_fake")
    def flagged_fake(state: State):
        obs.score(state.trace_id, "final_status", "flagged_fake", "CATEGORICAL")
        return {"status": "flagged_fake",
                "message": "Stopped: the article failed the credibility check (" +
                           "; ".join(state.credibility.reasons[:3]) + ")."}

    # ---- 1. Monitoring (LLM)
    @node("monitoring")
    def monitoring(state: State):
        r = state.retrieved
        blocks = []
        if "examples" in r:
            blocks.append(rag.format_examples(r["examples"]))
        if "entities" in r:
            blocks.append(rag.format_entities(r["entities"]))
        if state.retry_feedback and state.previous_monitoring:
            pm = state.previous_monitoring
            blocks.append(MONITORING_RETRY_CONTEXT.format(
                attempt=state.retry_count,
                feedback=state.retry_feedback,
                is_disruption=pm.is_disruption,
                disruption_type=pm.disruption_type,
                summary=pm.summary,
                countries=pm.countries,
                industries=pm.industries,
                companies=pm.companies))
        blocks = [b for b in blocks if b]
        user = (MONITORING_USER_WITH_CONTEXT.format(context="\n\n".join(blocks), article=state.article)
                if blocks else MONITORING_USER.format(article=state.article))
        out = llm.complete_json(
            "monitoring", obs.format_prompt("monitoring_system", MONITORING_SYSTEM, industries=", ".join(industries)),
            user, MonitoringOutput, payload={"article": state.article, "retrieved": r})
        return {"monitoring": out, "retry_feedback": "", "previous_monitoring": None}

    # ---- 2. KG query (tools): resolve names, then BFS
    @node("kg_query")
    def kg_query(state: State):
        m = state.monitoring
        focal = tools.resolve_entity(state.focal_name, g=g)["match"]
        if focal is None:
            raise ValueError(f"Could not resolve focal firm {state.focal_name!r}")
        countries, industries_, companies, unresolved = [], [], [], []
        for c in m.countries:
            (countries if c.lower() in vocab["country"] else unresolved).append(
                vocab["country"].get(c.lower(), c))
        for i in m.industries:
            (industries_ if i.lower() in vocab["industry"] else unresolved).append(
                vocab["industry"].get(i.lower(), i))
        for name in m.companies:
            hit = tools.resolve_entity(name, g=g)["match"]
            if hit and hit["score"] >= 0.8:
                companies.append(hit["company_id"])
            else:
                unresolved.append(name)
        bfs = tools.supply_chain_bfs(focal["company_id"], countries=countries,
                                     industries=industries_, companies=companies, g=g)
        return {"focal_id": focal["company_id"], "unresolved": unresolved, "bfs": bfs,
                "criteria": {"countries": countries, "industries": industries_, "companies": companies}}

    def _is_disruption(m: MonitoringOutput) -> bool:
        return m.is_disruption and m.disruption_type != "none"      # the model must agree with itself

    def after_kg(state: State):
        return "risk" if state.bfs["disrupted_count"] and _is_disruption(state.monitoring) else "no_exposure"

    @node("no_exposure")
    def no_exposure(state: State):
        why = ("the article is not a supply chain disruption" if not _is_disruption(state.monitoring)
               else "no supplier in the network matches the disruption")
        obs.score(state.trace_id, "final_status", "no_exposure", "CATEGORICAL")
        return {"status": "no_exposure",
                "message": f"No exposure for {state.focal_name}: {why}."}

    # ---- 3. Risk (tools)
    @node("risk")
    def risk(state: State):
        r = tools.tier1_risk_scores(state.bfs, g=g)
        if r["top_suppliers"]:
            obs.score(state.trace_id, "risk_top_score", r["top_suppliers"][0]["risk_score"])
        obs.score(state.trace_id, "disrupted_suppliers", state.bfs["disrupted_count"])
        return {"risk": r, "actions": tools.recommend_actions(r)}

    # ---- 4. Visualizer (tool)
    @node("visualizer")
    def visualizer(state: State):
        return {"network_html": tools.plot_network(state.bfs, state.risk)}

    # ---- 5. CSCO (LLM) — with optional RAG context
    @node("csco")
    def csco(state: State):
        import json
        m = state.monitoring
        disruption = {"type": m.disruption_type, "summary": m.summary, **state.criteria}
        exposure = {"disrupted_count": state.bfs["disrupted_count"],
                    "per_tier": {t: len(state.bfs[f"tier_{t}"]) for t in range(1, 5)},
                    "exposed_tier1": state.risk["exposed_tier1_count"]}
        # Build RAG context for CSCO if retrieval ran
        r = state.retrieved
        rag_blocks = []
        if use_rag and r:
            if "examples" in r and r["examples"]:
                from . import rag
                rag_blocks.append(rag.format_examples(r["examples"]))
            if "entities" in r and r["entities"]:
                from . import rag
                rag_blocks.append(rag.format_entities(r["entities"]))
        base_user = CSCO_USER.format(disruption=json.dumps(disruption), exposure=json.dumps(exposure),
                                     actions=json.dumps(state.actions))
        if rag_blocks:
            context = "\n\n".join(rag_blocks)
            user_prompt = CSCO_USER_WITH_CONTEXT.format(
                rag_context=CSCO_RAG_CONTEXT.format(context=context),
                disruption=json.dumps(disruption), exposure=json.dumps(exposure),
                actions=json.dumps(state.actions))
        else:
            user_prompt = base_user
        out = llm.complete_json(
            "csco", obs.prompt_text("csco_system", CSCO_SYSTEM), user_prompt,
            CSCOOutput, payload={"disruption": disruption, "bfs": state.bfs, "actions": state.actions})
        return {"csco": out}

    # ---- 5b. Self-critique (LLM) — suggests corrections for the reviewer
    @node("self_critique")
    def self_critique(state: State):
        m = state.monitoring
        try:
            user = SELF_CRITIQUE_USER.format(
                article=state.article, is_disruption=m.is_disruption,
                disruption_type=m.disruption_type, summary=m.summary,
                countries=m.countries, industries=m.industries, companies=m.companies)
            out = llm.complete_json("self_critique", obs.prompt_text("self_critique_system", SELF_CRITIQUE_SYSTEM),
                                   user, SelfCritiqueOutput)
            return {"ai_suggestions": out.suggestions or [], "ai_confidence": out.confidence or "medium"}
        except Exception:
            return {"ai_suggestions": [], "ai_confidence": ""}

    # ---- 5c. Alternative suppliers shown while the reviewer decides (tool)
    @node("preview_alternatives")
    def preview_alternatives(state: State):
        out = [tools.find_alternatives(a["company_id"], state.bfs, g=g) | {"risk_level": a["risk_level"]}
               for a in state.actions[:preview_n]]
        obs.span_meta(rule=ALT_RULES, suppliers=[
            {"supplier": a["supplier"]["name"], "risk_level": a["risk_level"],
             "candidates_considered": a["candidates_considered"], "clear_count": a["clear_count"],
             "fallback_to_exposed": a["clear_count"] == 0 and bool(a["alternatives"])} for a in out])
        return {"alternatives_preview": out}

    # ---- 6. Human review (interrupt)
    @node("human_review")
    def human_review(state: State):
        reply = interrupt({"csco": state.csco.model_dump(), "actions": state.actions,
                           "alternatives": state.alternatives_preview,
                           "credibility": state.credibility.model_dump() if state.credibility else None,
                           "retry_count": state.retry_count, "max_retries": max_retries,
                           "ai_suggestions": state.ai_suggestions, "ai_confidence": state.ai_confidence})
        decision = reply.get("decision")
        obs.score(state.trace_id, "human_decision", str(decision), "CATEGORICAL")
        obs.score(state.trace_id, "retry_count", state.retry_count)
        if decision == "edit":
            changed = any(a.get("risk_level") != b.get("risk_level")
                          for a, b in zip(reply.get("actions", state.actions), state.actions))
            obs.score(state.trace_id, "edited_actions", changed, "BOOLEAN")
        if decision == "reject":
            obs.score(state.trace_id, "final_status", "rejected", "CATEGORICAL")
            return {"approval": "rejected", "status": "rejected",
                    "message": reply.get("reason", "Rejected by reviewer.")}
        if decision == "retry":
            if state.retry_count >= max_retries:
                return {"approval": "rejected", "status": "rejected",
                        "message": f"Maximum retries ({max_retries}) reached. Rejected."}
            feedback = reply.get("reason", "Please re-analyse the article.")
            return {"approval": "retry",
                    "retry_count": state.retry_count + 1,
                    "retry_feedback": feedback,
                    "previous_monitoring": state.monitoring}
        if decision == "edit":
            return {"approval": "edited",
                    "csco": CSCOOutput(**{**state.csco.model_dump(), **reply.get("csco", {})}),
                    "actions": reply.get("actions", state.actions)}
        if decision == "approve":
            return {"approval": "approved"}
        if decision == "edit_extraction":
            edits = reply.get("monitoring", {})
            m = state.monitoring
            updated = MonitoringOutput(**{**m.model_dump(), **edits})
            return {"approval": "edit_extraction", "monitoring": updated}
        raise ValueError(f"Unknown decision {decision!r}; use approve, edit, reject, retry or edit_extraction")

    def after_review(state: State):
        if state.approval == "rejected":
            return END
        if state.approval == "retry":
            return "monitoring"
        if state.approval == "edit_extraction":
            return "kg_query"
        return "alternatives"

    # ---- 7. Alternative sourcing (tool)
    @node("alternatives")
    def alternatives(state: State):
        ready = {p["supplier"]["company_id"]: p for p in state.alternatives_preview}
        alts = [{k: v for k, v in (ready.get(a["company_id"]) or
                                   tools.find_alternatives(a["company_id"], state.bfs, g=g)).items()
                 if k != "risk_level"}
                for a in state.actions if a["risk_level"] == "HIGH"]
        n_high = len(alts)
        obs.span_meta(rule=ALT_RULES, high_risk_suppliers=n_high,
                      clear_counts={a["supplier"]["name"]: a["clear_count"] for a in alts})
        obs.score(state.trace_id, "final_status", "completed", "CATEGORICAL")
        return {"alternatives": alts, "status": "completed",
                "message": f"Completed: {n_high} HIGH-risk supplier(s) with alternatives searched."}

    wf = StateGraph(State)
    for fn in (monitoring, kg_query, no_exposure, risk, visualizer, csco, self_critique,
               preview_alternatives, human_review, alternatives):
        wf.add_node(fn.__name__, fn)
    first = "monitoring"                                   # where the run goes after the optional steps
    if credibility:
        wf.add_node("credibility", credibility_check)
        wf.add_node("flagged_fake", flagged_fake)
        wf.add_conditional_edges("credibility", after_credibility,
                                 {"monitoring": "monitoring", "flagged_fake": "flagged_fake"})
        wf.add_edge("flagged_fake", END)
        first = "credibility"
    if use_rag:
        wf.add_node("retrieve", retrieve)
        wf.add_edge(START, "retrieve")
        wf.add_edge("retrieve", first)
    else:
        wf.add_edge(START, first)
    wf.add_edge("monitoring", "kg_query")
    wf.add_conditional_edges("kg_query", after_kg, {"risk": "risk", "no_exposure": "no_exposure"})
    wf.add_edge("no_exposure", END)
    wf.add_edge("risk", "visualizer")
    wf.add_edge("visualizer", "csco")
    wf.add_edge("csco", "self_critique")
    wf.add_edge("self_critique", "preview_alternatives")
    wf.add_edge("preview_alternatives", "human_review")
    wf.add_conditional_edges("human_review", after_review,
                             {"alternatives": "alternatives", END: END,
                              "monitoring": "monitoring", "kg_query": "kg_query"})
    wf.add_edge("alternatives", END)
    app = wf.compile(checkpointer=checkpointer or _memory_saver())
    app.llm = llm
    app.rag = {"rag_examples": rag_examples, "ground_entities": ground_entities}
    app.credibility = credibility
    return app


# ------------------------------------------------------------------ helpers for callers
def _result(app, config, out: dict) -> dict:
    state = app.get_state(config).values
    paused = "__interrupt__" in out
    return {"paused": paused, "review": out["__interrupt__"][0].value if paused else None,
            "state": state if isinstance(state, State) else State(**state)}


def run_article(app, article: str, focal: str, thread_id: str = "run") -> dict:
    """Run until the pipeline ends or pauses for human review."""
    config = {"configurable": {"thread_id": thread_id}}
    trace_id, session_id = obs.new_trace(thread_id)
    out = app.invoke(State(article=article, focal_name=focal, trace_id=trace_id, session_id=session_id), config)
    obs.flush()
    return _result(app, config, out)


def resume_review(app, thread_id: str, decision: dict) -> dict:
    """decision: {"decision": "approve"} | {"decision": "reject", "reason": ...} |
    {"decision": "edit", "csco": {...partial...}, "actions": [...]}"""
    config = {"configurable": {"thread_id": thread_id}}
    out = app.invoke(Command(resume=decision), config)
    obs.flush()
    return _result(app, config, out)
