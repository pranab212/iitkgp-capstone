"""Command-line entry point.

    python cli.py data                                  # regenerate the dummy network (data/*.csv)
    python cli.py demo [--focal NAME] [--country C] [--industry I]
                                                        # tools only, no LLM
    python cli.py run  (--text "..." | --article FILE) [--focal NAME] [--auto approve|reject]
                                                        # full agent pipeline (Qwen/Azure if configured in .env, else mock)
    python cli.py eval-build                            # (re)write data/eval_articles.json (20 articles + expected answers)
    python cli.py eval [--ids a,b] [--mock] [--price-in X --price-out Y]
                       [--rag-examples] [--ground-entities] [--rag] [--k 3]
                                                        # run the evaluation set and print the report
    python cli.py eval-compare outputs/eval_results_baseline.json outputs/eval_results_rag-examples.json ...
                                                        # headline metrics side by side
    (run also takes --rag-examples / --ground-entities / --rag)

Generated files go to outputs/.
"""
import argparse
import json

from scpoc.graph import OUT_DIR


def cmd_data(_):
    from scpoc import datagen
    comp, sup = datagen.generate()
    datagen.DATA_DIR.mkdir(exist_ok=True)
    comp.to_csv(datagen.DATA_DIR / "companies.csv", index=False)
    sup.to_csv(datagen.DATA_DIR / "supplies.csv", index=False)
    print(f"wrote {len(comp)} companies and {len(sup)} supply links to {datagen.DATA_DIR}")


def cmd_demo(a):
    from scpoc.tools import (get_graph, plot_network, recommend_actions, resolve_entity,
                             supply_chain_bfs, tier1_risk_scores)
    countries = a.country if a.country or a.industry else ["Ukraine"]
    focal = resolve_entity(a.focal)["match"]
    print(f"1. resolve_entity('{a.focal}') -> {focal['company_id']} {focal['name']}")

    bfs = supply_chain_bfs(focal["company_id"], countries=countries, industries=a.industry)
    per_tier = {t: len(bfs[f"tier_{t}"]) for t in range(1, 5)}
    print(f"2. supply_chain_bfs({countries or ''}{a.industry or ''}) -> "
          f"{bfs['disrupted_count']} disrupted suppliers {per_tier}")

    risk = tier1_risk_scores(bfs)
    print(f"3. tier1_risk_scores -> {risk['exposed_tier1_count']} exposed Tier-1 suppliers, top 10:")
    print(f"   {'supplier':28}{'score':>7}  level   breadth dep  crit  cent  depth")
    for r in risk["top_suppliers"]:
        c = r["components"]
        print(f"   {r['name'][:28]:28}{r['risk_score']:>7}  {r['risk_level']:7} "
              f"{c['exposure_breadth']:.2f}   {c['dependency_ratio']:.2f} {c['downstream_criticality']:.2f}  "
              f"{c['tier1_centrality']:.2f}  {c['exposure_depth']:.2f}")

    print("4. recommend_actions:")
    for r in recommend_actions(risk)[:5]:
        print(f"   {r['name'][:28]:28} {r['risk_level']:7} -> {r['action']}")

    print(f"5. plot_network -> {plot_network(bfs, risk)}")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "disruption_result.json"
    out.write_text(json.dumps({"bfs": bfs, "risk": risk, "actions": recommend_actions(risk)}, indent=2))
    print(f"   full JSON -> {out}")


def cmd_run(a):
    from scpoc.pipeline import build_app, resume_review, run_article
    article = open(a.article, encoding="utf-8").read() if a.article else a.text
    if not article:
        raise SystemExit("give --article or --text")

    app = build_app(k_examples=a.k, credibility=a.credibility or a.block_fake, block_fake=a.block_fake,
                    **_rag_flags(a))
    print("LLM:", app.llm.name, "| RAG:", {k: v for k, v in app.rag.items() if v} or "off")
    r = run_article(app, article, a.focal, thread_id="cli")
    s = r["state"]
    m = s.monitoring
    if s.credibility:
        c = s.credibility
        print(f"[credibility] {c.verdict.upper()} (suspicion score {c.score})")
        for why in c.reasons[:5]:
            print("   -", why)
        if s.status == "flagged_fake":
            print(f"\nstatus: {s.status}  {s.message}")
            return
    if s.retrieved.get("examples"):
        print("[retrieve] examples:", [f"{e['id']} ({e['score']})" for e in s.retrieved["examples"]])
    if "entities" in s.retrieved:
        print("[retrieve] companies named:", [e["name"] for e in s.retrieved["entities"]] or "none")
    print(f"\n[monitoring] disruption={m.is_disruption} type={m.disruption_type} "
          f"countries={m.countries} industries={m.industries} companies={m.companies}")
    if s.unresolved:
        print(f"[kg_query] not found in graph: {s.unresolved}")
    if s.bfs:
        print(f"[kg_query] {s.bfs['disrupted_count']} disrupted suppliers "
              f"{ {t: len(s.bfs[f'tier_{t}']) for t in range(1, 5)} }")

    while r["paused"]:
        retry_n = r["review"].get("retry_count", 0)
        max_r = r["review"].get("max_retries", 3)
        if retry_n:
            print(f"\n--- AI retry attempt #{retry_n} ---")
        print("\n[csco]", json.dumps(r["review"]["csco"], indent=2))
        for x in r["review"]["actions"][:10]:
            print(f"  {x['name'][:30]:30} {x['risk_score']:>6} {x['risk_level']:7} {x['action']}")
        for alt in r["review"].get("alternatives", []):
            print(f"\n  possible replacements for {alt['supplier']['name']} ({alt['risk_level']}, "
                  f"{alt['clear_count']} fully clear):")
            for c in alt["alternatives"][:3]:
                print(f"    {c['name'][:30]:30} {c['country']:15} clear={c['verified_clear']} upstream={c['upstream_size']}")
        if a.auto:
            choice = a.auto
        else:
            prompt = f"\n[a]pprove / [e]dit / [r]eject"
            if retry_n < max_r:
                prompt += f" / re[t]ry with AI ({max_r - retry_n} left)"
            prompt += "? "
            ans = input(prompt).strip().lower()
            choice = {"a": "approve", "e": "edit", "r": "reject", "t": "retry",
                       "approve": "approve", "edit": "edit", "reject": "reject", "retry": "retry",
                       "y": "approve", "yes": "approve", "n": "reject", "": "approve"}.get(ans, "approve")
        decision = {"decision": choice}
        if choice == "retry":
            feedback = input("  feedback for the AI: ").strip() if not a.auto else "Please re-analyse more carefully."
            decision["reason"] = feedback or "Please re-analyse more carefully."
        r = resume_review(app, "cli", decision)
        s = r["state"]

    print(f"\nstatus: {s.status}  {s.message}")
    for alt in s.alternatives:
        print(f"\nalternatives for {alt['supplier']['name']} ({alt['supplier']['industry']}):")
        for c in alt["alternatives"]:
            print(f"  {c['name'][:30]:30} {c['country']:15} clear={c['verified_clear']} "
                  f"upstream={c['upstream_size']}")
    tok = {k: sum(t[k] for t in s.trace) for k in ("prompt_tokens", "completion_tokens")}
    print(f"\ntime per step: { {t['step']: t['seconds'] for t in s.trace} }  tokens: {tok}")
    from scpoc import observability as obs
    if obs.enabled():
        print("langfuse trace:", obs.trace_url(s.trace_id))
        obs.flush()


def cmd_eval_build(_):
    from scpoc import evalset
    arts = evalset.build()
    print(f"wrote {len(arts)} articles with expected answers to {evalset.EVAL_FILE}")


def _rag_flags(a):
    return {"rag_examples": a.rag or a.rag_examples, "ground_entities": a.rag or a.ground_entities}


def cmd_eval(a):
    from scpoc import evaluate
    ids = a.ids.split(",") if a.ids else None
    print(evaluate.main(ids=ids, price_in=a.price_in, price_out=a.price_out, mock=a.mock,
                        k_examples=a.k, with_retry=a.with_retry,
                        max_retries=a.max_retries, langfuse_run=a.langfuse_run, **_rag_flags(a)))
    report, results = evaluate.main.saved
    print(f"\nsaved to {report} and {results}")


def cmd_langfuse_push_prompts(_):
    from scpoc import observability as obs
    from scpoc.prompts import PROMPTS
    print("\n".join(obs.push_prompts(PROMPTS)))
    obs.flush()


def cmd_langfuse_sync_dataset(a):
    from scpoc import observability as obs
    from scpoc.evalset import load
    print(obs.sync_dataset(load(), a.name))


def cmd_eval_compare(a):
    from scpoc import evaluate
    print(evaluate.compare(a.files))


def _add_rag_args(p):
    p.add_argument("--rag-examples", action="store_true",
                   help="add the most similar labeled examples (data/rag/exemplars.json) to the Monitoring prompt")
    p.add_argument("--ground-entities", action="store_true",
                   help="add the graph records of companies the article names to the Monitoring prompt")
    p.add_argument("--rag", action="store_true", help="both of the above")
    p.add_argument("--k", type=int, default=3, help="number of retrieved examples (default 3)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="regenerate the dummy network").set_defaults(fn=cmd_data)

    d = sub.add_parser("demo", help="run the tools on one scenario (no LLM)")
    d.add_argument("--focal", default="northwind grocers")
    d.add_argument("--country", action="append")
    d.add_argument("--industry", action="append")
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("run", help="run one article through the agent pipeline")
    r.add_argument("--article", help="path to a text file")
    r.add_argument("--text", help="article text")
    r.add_argument("--focal", default="Northwind Grocers")
    r.add_argument("--auto", choices=["approve", "reject"], help="skip the prompt")
    r.add_argument("--credibility", action="store_true", help="add the fake-news check before Monitoring")
    r.add_argument("--block-fake", action="store_true", help="stop the run when the article looks fake (implies --credibility)")
    _add_rag_args(r)
    r.set_defaults(fn=cmd_run)

    sub.add_parser("eval-build", help="write the evaluation articles and expected answers").set_defaults(fn=cmd_eval_build)
    e = sub.add_parser("eval", help="run the evaluation set")
    e.add_argument("--ids", help="comma-separated article ids (default: all)")
    e.add_argument("--mock", action="store_true", help="use the offline mock LLM")
    e.add_argument("--price-in", type=float, default=0.0, help="USD per 1K input tokens")
    e.add_argument("--price-out", type=float, default=0.0, help="USD per 1K output tokens")
    e.add_argument("--with-retry", action="store_true",
                   help="simulate human feedback retry on wrong extractions and measure improvement")
    e.add_argument("--max-retries", type=int, default=1, help="max retry attempts per article (default 1)")
    e.add_argument("--langfuse-run", metavar="NAME",
                   help="send traces to Langfuse as one session named NAME, with per-article eval scores")
    _add_rag_args(e)
    e.set_defaults(fn=cmd_eval)

    sub.add_parser("langfuse-push-prompts",
                   help="create the prompts in Langfuse prompt management (skips ones that exist)"
                   ).set_defaults(fn=cmd_langfuse_push_prompts)
    ds = sub.add_parser("langfuse-sync-dataset", help="upload the evaluation articles as a Langfuse dataset")
    ds.add_argument("--name", default="supply-chain-eval")
    ds.set_defaults(fn=cmd_langfuse_sync_dataset)

    c = sub.add_parser("eval-compare", help="compare saved eval_results*.json files side by side")
    c.add_argument("files", nargs="+", help="results files; the first one is the reference")
    c.set_defaults(fn=cmd_eval_compare)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
