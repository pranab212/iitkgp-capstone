"""Run the evaluation set through the pipeline and report how well it does.

    python cli.py eval [--ids a,b] [--price-in 0 --price-out 0] [--mock]

The tools are deterministic, so for the same extracted criteria they always give the same
answer. Any mismatch therefore traces back to what the LLM extracted. Reported:

  * extraction precision / recall / F1 (countries, industries, companies; micro-averaged)
  * whether the network query and risk steps match the expected answers
  * false-positive handling (articles that must end with "no exposure")
  * run time and token cost per article
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

from . import observability as obs
from . import tools
from .evalset import load
from .graph import OUT_DIR
from .llm import MockLLM, get_llm
from .pipeline import build_app, resume_review, run_article


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def _sets(a: dict, g) -> dict:
    """Truth items in the same canonical form the pipeline produces (company names -> ids)."""
    t = a["truth"]
    return {"countries": set(t["countries"]), "industries": set(t["industries"]),
            "companies": set(a["expected"]["company_ids"])}


def _generate_feedback(truth: dict, pred: dict, unresolved: list) -> str:
    """Simulate human feedback by comparing the extraction with the expected answer."""
    parts = []
    for field in ("countries", "industries", "companies"):
        fp = pred[field] - truth[field]
        fn = truth[field] - pred[field]
        if fp:
            parts.append(f"Remove {field}: {', '.join(sorted(fp))} — not actually disrupted.")
        if fn:
            parts.append(f"You missed {field}: {', '.join(sorted(fn))} — add them.")
    if unresolved:
        parts.append(f"These are not in the supply network, remove them: {', '.join(unresolved)}.")
    return " ".join(parts) if parts else ""


def run_one_with_retry(app, a: dict, g, thread: str, max_retries: int = 1) -> dict:
    """Run an article, and if the extraction is wrong, simulate a human giving
    feedback based on the known truth, then retry up to max_retries times.
    Returns the row with both the initial and final extraction recorded."""
    row = {"id": a["id"], "kind": a["kind"], "focal": a["focal"], "error": None}
    t0 = time.perf_counter()
    try:
        r = run_article(app, a["article"], a["focal"], thread)
    except Exception as e:
        row.update(error=f"{type(e).__name__}: {e}", seconds=round(time.perf_counter() - t0, 2))
        return row

    truth = _sets(a, g)
    retries_used = 0
    initial_extraction = None

    for attempt in range(max_retries + 1):
        if not r["paused"]:
            break
        st = r["state"]
        pred = {"countries": set(st.criteria.get("countries", [])),
                "industries": set(st.criteria.get("industries", [])),
                "companies": set(st.criteria.get("companies", []))}
        is_exact = all(pred[k] == truth[k] for k in truth) and not st.unresolved

        if attempt == 0:
            initial_extraction = {
                "countries": sorted(pred["countries"]),
                "industries": sorted(pred["industries"]),
                "companies": sorted(pred["companies"]),
                "unresolved": list(st.unresolved),
                "exact": is_exact}

        if is_exact or attempt == max_retries:
            r = resume_review(app, thread, {"decision": "approve"})
            break

        feedback = _generate_feedback(truth, pred, st.unresolved)
        if not feedback:
            r = resume_review(app, thread, {"decision": "approve"})
            break
        retries_used += 1
        r = resume_review(app, thread, {"decision": "retry", "reason": feedback})

    st = r["state"]
    row["seconds"] = round(time.perf_counter() - t0, 2)
    row["prompt_tokens"] = sum(t["prompt_tokens"] for t in st.trace)
    row["completion_tokens"] = sum(t["completion_tokens"] for t in st.trace)
    row["retries_used"] = retries_used
    row["initial_extraction"] = initial_extraction

    m = st.monitoring
    pred = {"countries": set(st.criteria.get("countries", [])),
            "industries": set(st.criteria.get("industries", [])),
            "companies": set(st.criteria.get("companies", []))}
    counts = {k: {"tp": len(truth[k] & pred[k]), "fp": len(pred[k] - truth[k]), "fn": len(truth[k] - pred[k])}
              for k in truth}
    counts["unresolved"] = {"tp": 0, "fp": len(st.unresolved), "fn": 0}
    row["extraction"] = counts
    row["extraction_exact"] = all(c["fp"] == 0 and c["fn"] == 0 for c in counts.values())
    row["extracted"] = {"is_disruption": m.is_disruption, "type": m.disruption_type,
                        **{k: sorted(v) for k, v in pred.items()}, "unresolved": st.unresolved}
    row["truth"] = {k: sorted(v) for k, v in truth.items()}

    ret = st.retrieved or {}
    if "examples" in ret:
        kinds = [e["kind"] for e in ret["examples"]]
        row["retrieval_examples"] = {"ids": [e["id"] for e in ret["examples"]], "kinds": kinds,
                                     "same_kind_hit": a["kind"] in kinds}
    if "entities" in ret:
        found = {e["company_id"] for e in ret["entities"] if not e["is_focal"]}
        want = set(a["expected"]["company_ids"])
        row["retrieval_entities"] = {"found": sorted(found), "tp": len(found & want),
                                     "fp": len(found - want), "fn": len(want - found)}

    got_n = st.bfs["disrupted_count"] if st.bfs else 0
    exp = a["expected"]
    row["disrupted"] = {"expected": exp["disrupted_count"], "got": got_n, "match": got_n == exp["disrupted_count"]}
    row["status"] = {"expected": exp["final_status"], "got": st.status, "match": st.status == exp["final_status"]}
    if a["kind"] != "false_positive":
        top = st.risk["top_suppliers"][0] if st.risk and st.risk["top_suppliers"] else None
        high = next((s for s in (st.risk or {}).get("top_suppliers", []) if s["risk_level"] == "HIGH"), None)
        same = lambda x, y: (x is None and y is None) or (x and y and x["company_id"] == y["company_id"])
        row["top_supplier"] = {"expected": (exp["top_supplier"] or {}).get("name"), "got": top and top["name"],
                               "match": bool(same(top, exp["top_supplier"]))}
        row["top_high"] = {"expected": (exp["top_high_supplier"] or {}).get("name"), "got": high and high["name"],
                           "match": bool(same(high, exp["top_high_supplier"]))}
    obs.eval_scores(st.trace_id, row)                    # per-article metrics on that article's Langfuse trace
    return row


def run_one(app, a: dict, g, thread: str) -> dict:
    row = {"id": a["id"], "kind": a["kind"], "focal": a["focal"], "error": None}
    t0 = time.perf_counter()
    try:
        r = run_article(app, a["article"], a["focal"], thread)
        if r["paused"]:                                  # approve so alternative sourcing also runs
            r = resume_review(app, thread, {"decision": "approve"})
    except Exception as e:                               # record the failure and carry on
        row.update(error=f"{type(e).__name__}: {e}", seconds=round(time.perf_counter() - t0, 2))
        return row
    st, exp = r["state"], a["expected"]
    row["seconds"] = round(time.perf_counter() - t0, 2)
    row["prompt_tokens"] = sum(t["prompt_tokens"] for t in st.trace)
    row["completion_tokens"] = sum(t["completion_tokens"] for t in st.trace)
    m = st.monitoring

    # extraction, scored on what the pipeline resolved plus anything it could not find in the network
    truth = _sets(a, g)
    pred = {"countries": set(st.criteria.get("countries", [])), "industries": set(st.criteria.get("industries", [])),
            "companies": set(st.criteria.get("companies", []))}
    counts = {k: {"tp": len(truth[k] & pred[k]), "fp": len(pred[k] - truth[k]), "fn": len(truth[k] - pred[k])}
              for k in truth}
    counts["unresolved"] = {"tp": 0, "fp": len(st.unresolved), "fn": 0}   # named things that are not in the network
    row["extraction"] = counts
    row["extraction_exact"] = all(c["fp"] == 0 and c["fn"] == 0 for c in counts.values())
    row["extracted"] = {"is_disruption": m.is_disruption, "type": m.disruption_type,
                        **{k: sorted(v) for k, v in pred.items()}, "unresolved": st.unresolved}
    row["truth"] = {k: sorted(v) for k, v in truth.items()}

    # retrieval (only when RAG is on): did we fetch an example of the same kind, did grounding find the companies
    ret = st.retrieved or {}
    if "examples" in ret:
        kinds = [e["kind"] for e in ret["examples"]]
        row["retrieval_examples"] = {"ids": [e["id"] for e in ret["examples"]], "kinds": kinds,
                                     "same_kind_hit": a["kind"] in kinds}
    if "entities" in ret:
        found = {e["company_id"] for e in ret["entities"] if not e["is_focal"]}
        want = set(exp["company_ids"])
        row["retrieval_entities"] = {"found": sorted(found), "tp": len(found & want),
                                     "fp": len(found - want), "fn": len(want - found)}

    # network query and risk
    got_n = st.bfs["disrupted_count"] if st.bfs else 0
    row["disrupted"] = {"expected": exp["disrupted_count"], "got": got_n, "match": got_n == exp["disrupted_count"]}
    row["status"] = {"expected": exp["final_status"], "got": st.status, "match": st.status == exp["final_status"]}
    if a["kind"] != "false_positive":
        top = st.risk["top_suppliers"][0] if st.risk and st.risk["top_suppliers"] else None
        high = next((s for s in (st.risk or {}).get("top_suppliers", []) if s["risk_level"] == "HIGH"), None)
        same = lambda x, y: (x is None and y is None) or (x and y and x["company_id"] == y["company_id"])
        row["top_supplier"] = {"expected": (exp["top_supplier"] or {}).get("name"), "got": top and top["name"],
                               "match": bool(same(top, exp["top_supplier"]))}
        row["top_high"] = {"expected": (exp["top_high_supplier"] or {}).get("name"), "got": high and high["name"],
                           "match": bool(same(high, exp["top_high_supplier"]))}
    obs.eval_scores(st.trace_id, row)                    # per-article metrics on that article's Langfuse trace
    return row


def summarise(rows: list[dict], price_in: float, price_out: float) -> dict:
    ok = [r for r in rows if not r["error"]]
    out = {"articles": len(rows), "errors": [{"id": r["id"], "error": r["error"]} for r in rows if r["error"]]}
    tot = defaultdict(lambda: defaultdict(int))
    for r in ok:
        for field, c in r["extraction"].items():
            for k in ("tp", "fp", "fn"):
                tot[field][k] += c[k]
                tot["overall"][k] += c[k]
    out["extraction"] = {f: _prf(**{k: v for k, v in c.items()}) for f, c in tot.items()}
    out["extraction_exact_match"] = f"{sum(r['extraction_exact'] for r in ok)}/{len(ok)}"

    pos = [r for r in ok if r["kind"] != "false_positive"]
    fps = [r for r in ok if r["kind"] == "false_positive"]
    rate = lambda xs: f"{sum(xs)}/{len(xs)}"
    out["network_query"] = {"disrupted_count_match": rate([r["disrupted"]["match"] for r in pos]),
                            "exposure_detected": rate([r["status"]["got"] != "no_exposure" for r in pos])}
    out["risk"] = {"top_supplier_match": rate([r["top_supplier"]["match"] for r in pos]),
                   "top_high_supplier_match": rate([r["top_high"]["match"] for r in pos])}
    out["false_positives"] = {"correctly_ended_no_exposure": rate([r["status"]["got"] == "no_exposure" for r in fps]),
                              "wrongly_flagged": [r["id"] for r in fps if r["status"]["got"] != "no_exposure"]}
    secs = [r["seconds"] for r in ok]
    pt, ct = [r["prompt_tokens"] for r in ok], [r["completion_tokens"] for r in ok]
    n = max(len(ok), 1)
    out["cost"] = {"mean_seconds": round(sum(secs) / n, 2), "max_seconds": max(secs, default=0),
                   "mean_prompt_tokens": round(sum(pt) / n), "mean_completion_tokens": round(sum(ct) / n),
                   "mean_cost_usd": round((sum(pt) * price_in + sum(ct) * price_out) / 1000 / n, 5),
                   "price_per_1k_tokens_usd": {"input": price_in, "output": price_out}}
    ex = [r["retrieval_examples"] for r in ok if "retrieval_examples" in r]
    if ex:
        out.setdefault("retrieval", {})["examples_same_kind_hit"] = rate([e["same_kind_hit"] for e in ex])
    en = [r["retrieval_entities"] for r in ok if "retrieval_entities" in r]
    if en:
        out.setdefault("retrieval", {})["entity_grounding"] = _prf(*(sum(e[k] for e in en) for k in ("tp", "fp", "fn")))
    return out


def report(rows: list[dict], s: dict, llm_name: str, config: str = "baseline") -> str:
    L = [f"EVALUATION - LLM: {llm_name} - {s['articles']} articles - config: {config}", ""]
    L.append(f"{'article':18}{'kind':18}{'extract':8}{'disrupted exp/got':>19}  {'top supplier':8}{'top HIGH':9}{'secs':>6}{'tokens':>8}")
    for r in rows:
        if r["error"]:
            L.append(f"{r['id']:18}{r['kind']:18}ERROR {r['error'][:80]}")
            continue
        d = r["disrupted"]
        ts = "-" if "top_supplier" not in r else ("ok" if r["top_supplier"]["match"] else "WRONG")
        th = "-" if "top_high" not in r else ("ok" if r["top_high"]["match"] else "WRONG")
        L.append(f"{r['id']:18}{r['kind']:18}{'exact' if r['extraction_exact'] else 'DIFF':8}"
                 f"{d['expected']:>9}/{d['got']:<9}  {ts:8}{th:9}{r['seconds']:>6}{r['prompt_tokens'] + r['completion_tokens']:>8}")
    L += ["", "Extraction differences (truth vs extracted):"]
    shown = False
    for r in rows:
        if not r["error"] and not r["extraction_exact"]:
            shown = True
            L.append(f"  {r['id']}: truth={r['truth']}  extracted={r['extracted']}")
    if not shown:
        L.append("  none")
    L += ["", "Extraction (micro-averaged; paper reports F1 0.962-0.991 on its own tasks):"]
    for f, v in s["extraction"].items():
        L.append(f"  {f:11} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}  (tp {v['tp']}, fp {v['fp']}, fn {v['fn']})")
    L.append(f"  articles with exact extraction: {s['extraction_exact_match']}")
    L += ["", "Network query:", f"  disrupted count matches expected: {s['network_query']['disrupted_count_match']}",
          f"  exposure detected on positive articles: {s['network_query']['exposure_detected']}",
          "Risk:", f"  top Tier-1 supplier matches: {s['risk']['top_supplier_match']}",
          f"  top HIGH supplier matches (or none expected): {s['risk']['top_high_supplier_match']}",
          "False positives:", f"  correctly ended with 'no exposure': {s['false_positives']['correctly_ended_no_exposure']}"]
    if s["false_positives"]["wrongly_flagged"]:
        L.append(f"  wrongly flagged: {', '.join(s['false_positives']['wrongly_flagged'])}")
    c = s["cost"]
    L += ["Time and cost per article (includes alternative sourcing):",
          f"  mean {c['mean_seconds']}s (max {c['max_seconds']}s), mean tokens {c['mean_prompt_tokens']} in / "
          f"{c['mean_completion_tokens']} out, mean cost ${c['mean_cost_usd']}",
          f"  (price per 1K tokens: ${c['price_per_1k_tokens_usd']['input']} in, ${c['price_per_1k_tokens_usd']['output']} out)"]
    if "retrieval" in s:
        L.append("Retrieval:")
        rv = s["retrieval"]
        if "examples_same_kind_hit" in rv:
            L.append(f"  articles with a retrieved example of the same kind: {rv['examples_same_kind_hit']}")
        if "entity_grounding" in rv:
            v = rv["entity_grounding"]
            L.append(f"  entity grounding (supplier companies named in the article): P={v['precision']:.3f} "
                     f"R={v['recall']:.3f} (tp {v['tp']}, fp {v['fp']}, fn {v['fn']})")
    if "retry" in s:
        rv = s["retry"]
        L += ["Human-in-the-loop retry:",
              f"  articles retried: {rv['articles_retried']}",
              f"  fixed by retry: {rv['fixed_by_retry']}",
              f"  fixed ids: {', '.join(rv['fixed_ids']) or 'none'}",
              f"  still wrong after retry: {', '.join(rv['still_wrong_after_retry']) or 'none'}",
              f"  mean retries per article: {rv['mean_retries']}"]
    if s["errors"]:
        L += ["", f"Errors: {s['errors']}"]
    return "\n".join(L)


def config_tag(rag_examples: bool, ground_entities: bool) -> str:
    parts = [n for n, on in (("rag-examples", rag_examples), ("ground-entities", ground_entities)) if on]
    return "+".join(parts) or "baseline"


def main(ids: list[str] | None = None, price_in: float = 0.0, price_out: float = 0.0, mock: bool = False,
         rag_examples: bool = False, ground_entities: bool = False, k_examples: int = 3,
         with_retry: bool = False, max_retries: int = 1, langfuse_run: str | None = None) -> str:
    obs.RUN_NAME = langfuse_run          # one Langfuse session (and tag) for the whole run, so runs can be compared
    g = tools.get_graph()
    arts = [a for a in load() if not ids or a["id"] in ids]
    llm = (MockLLM({d["country"] for _, d in g.nodes(data=True)}, {d["industry"] for _, d in g.nodes(data=True)},
                   {d["name"] for _, d in g.nodes(data=True)}) if mock else get_llm(g))
    app = build_app(llm=llm, g=g, rag_examples=rag_examples, ground_entities=ground_entities,
                    k_examples=k_examples)
    config = config_tag(rag_examples, ground_entities)
    if with_retry:
        config += "+retry"
    rows = []
    for i, a in enumerate(arts, 1):
        print(f"[{i}/{len(arts)}] {a['id']} ...", flush=True)
        if with_retry:
            rows.append(run_one_with_retry(app, a, g, f"eval_{a['id']}", max_retries=max_retries))
        else:
            rows.append(run_one(app, a, g, f"eval_{a['id']}"))
    s = summarise(rows, price_in, price_out)
    if with_retry:
        retried = [r for r in rows if r.get("retries_used", 0) > 0]
        fixed = [r for r in retried if r["extraction_exact"] and not r.get("initial_extraction", {}).get("exact")]
        s["retry"] = {
            "articles_retried": f"{len(retried)}/{len(rows)}",
            "fixed_by_retry": f"{len(fixed)}/{len(retried)}" if retried else "0/0",
            "fixed_ids": [r["id"] for r in fixed],
            "still_wrong_after_retry": [r["id"] for r in retried if not r["extraction_exact"]],
            "mean_retries": round(sum(r.get("retries_used", 0) for r in rows) / max(len(rows), 1), 2)}
    text = report(rows, s, llm.name, config)
    OUT_DIR.mkdir(exist_ok=True)
    # no flags -> the original file names; with flags -> one file per configuration, so runs don't overwrite each other
    suffix = "" if config == "baseline" else "_" + config.replace("+", "_")
    (OUT_DIR / f"eval_results{suffix}.json").write_text(
        json.dumps({"llm": llm.name, "config": config, "summary": s, "articles": rows}, indent=2))
    (OUT_DIR / f"eval_report{suffix}.txt").write_text(text + "\n", encoding="utf-8")
    obs.flush()
    main.saved = (OUT_DIR / f"eval_report{suffix}.txt", OUT_DIR / f"eval_results{suffix}.json")
    return text


# ------------------------------------------------------------------ side-by-side comparison of saved runs
def _metrics(s: dict) -> dict:
    ex = s["extraction"]
    out = {f"{f} F1": ex[f]["f1"] for f in ("overall", "countries", "industries", "companies") if f in ex}
    out.update({f"{f} precision": ex[f]["precision"] for f in ("overall", "industries") if f in ex})
    out.update({"exact extraction": s["extraction_exact_match"],
                "disrupted count match": s["network_query"]["disrupted_count_match"],
                "top Tier-1 supplier match": s["risk"]["top_supplier_match"],
                "top HIGH supplier match": s["risk"]["top_high_supplier_match"],
                "false positives handled": s["false_positives"]["correctly_ended_no_exposure"],
                "mean prompt tokens": s["cost"]["mean_prompt_tokens"],
                "mean seconds": s["cost"]["mean_seconds"]})
    return out


def compare(paths: list[str]) -> str:
    """Table of the headline metrics of several eval_results*.json files, plus per-article changes."""
    runs = []
    for p in paths:
        d = json.loads(open(p, encoding="utf-8").read())
        runs.append((d.get("config") or p.rsplit("/", 1)[-1], d))
    names = [n for n, _ in runs]
    w = max(14, *(len(n) + 2 for n in names))
    L = [f"{'metric':28}" + "".join(f"{n:>{w}}" for n in names)]
    mets = [_metrics(d["summary"]) for _, d in runs]
    for k in mets[0]:
        L.append(f"{k:28}" + "".join(f"{str(m.get(k, '-')):>{w}}" for m in mets))
    # which articles changed exact-extraction status relative to the first run
    base = {r["id"]: r for r in runs[0][1]["articles"]}
    for n, d in runs[1:]:
        fixed, broke = [], []
        for r in d["articles"]:
            b = base.get(r["id"])
            if not b or r.get("error") or b.get("error"):
                continue
            if r["extraction_exact"] and not b["extraction_exact"]:
                fixed.append(r["id"])
            if b["extraction_exact"] and not r["extraction_exact"]:
                broke.append(r["id"])
        L.append(f"\n{n} vs {names[0]}: now exact: {', '.join(fixed) or 'none'};  no longer exact: {', '.join(broke) or 'none'}")
    return "\n".join(L)
