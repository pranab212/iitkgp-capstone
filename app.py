"""Supply-chain disruption monitor — web server + JSON API.

    python app.py            ->  http://localhost:8080   (set PORT to change)

UI lives in dashboard.html (loaded at import time). The pipeline is in scpoc/.
Uses Qwen3-8B if QWEN_API_KEY is set, else Azure OpenAI, else the offline mock LLM.
"""
import json
import os
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from langgraph.types import Command

from scpoc.graph import OUT_DIR
from scpoc.pipeline import build_app
from scpoc.schemas import State
from scpoc import tools

PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST", "127.0.0.1")

GROCERS = ["Northwind Grocers", "Alpenmarkt", "Maple Fresh Markets"]

# Ready-made articles. They name countries / industries / companies that exist in the
# dummy network so the offline mock extracts them; the real LLM reads them naturally.
SCENARIOS = [
    {"id": "ukraine", "title": "Ukraine conflict", "focal": "Northwind Grocers",
     "article": "Fighting in Ukraine has escalated and Black Sea ports have been blockaded. "
                "Producers across Ukraine have halted shipments, and buyers in Europe expect "
                "shortages of grain and sunflower oil within weeks."},
    {"id": "fire", "title": "Plant fire at a packaging supplier", "focal": "Northwind Grocers",
     "article": "A fire broke out overnight at the main plant of Sakri Packaging 020, a key maker of "
                "film and cartons. The company said production is suspended for at least six weeks "
                "while the site is inspected."},
    {"id": "port", "title": "Port strike in Vietnam", "focal": "Northwind Grocers",
     "article": "Dockworkers in Vietnam began an indefinite strike today, shutting container terminals. "
                "Logistics operators say cold-chain transport and freight forwarding out of the country "
                "are suspended, with long delays expected."},
    {"id": "falsepos", "title": "False positive: storm in Iceland", "focal": "Northwind Grocers",
     "article": "A heavy storm closed roads in Iceland on Saturday and cancelled a music festival. "
                "Organisers said tickets will be refunded."},
]

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
PIPELINE = None            # built once at startup


# ------------------------------------------------------------------ running the pipeline
def _summarise(node: str, u: dict) -> dict:
    """Turn a node's state update into a small JSON-friendly dict for the page."""
    if node == "retrieve":
        r = u["retrieved"]
        examples = [{"id": e["id"], "kind": e["kind"], "score": e["score"]} for e in r.get("examples", [])]
        fake_news = None
        if examples:
            fp_votes = sum(1 for e in examples if e["kind"] == "false_positive")
            fake_news = {"verdict": "fake" if fp_votes * 2 > len(examples) else "genuine",
                        "false_positive_votes": fp_votes, "total_votes": len(examples),
                        "top_match": examples[0]["id"], "top_score": examples[0]["score"]}
        return {"examples": examples, "fake_news": fake_news,
                "entities": [f"{e['name']} ({e['industry']}, {e['country']})" for e in r.get("entities", [])],
                "grounding": "entities" in r}
    if node == "monitoring":
        return u["monitoring"].model_dump()
    if node == "kg_query":
        bfs = u["bfs"]
        chains = [c for t in (4, 3, 2, 1) for c in bfs[f"tier_{t}"]][:5]
        return {"focal": bfs["focal"]["name"], "criteria": u["criteria"], "unresolved": u["unresolved"],
                "disrupted_count": bfs["disrupted_count"],
                "per_tier": {t: len(bfs[f"tier_{t}"]) for t in range(1, 5)},
                "sample_chains": [[s["name"] + (f" [{s['product_supplied']}]" if "product_supplied" in s else "")
                                   for s in reversed(c)] for c in chains]}
    if node == "risk":
        r = u["risk"]
        return {"exposed_tier1_count": r["exposed_tier1_count"], "top_suppliers": r["top_suppliers"],
                "weights": r["weights"], "thresholds": r["thresholds"]}
    if node == "csco":
        d = u["csco"].model_dump()
        d["rag_enhanced"] = bool(PIPELINE and PIPELINE.rag and any(PIPELINE.rag.values()))
        return d
    if node == "no_exposure":
        return {"message": u["message"]}
    if node == "alternatives":
        return {"alternatives": u["alternatives"], "message": u["message"]}
    if node == "self_critique":
        return {"suggestions": u.get("ai_suggestions", []), "confidence": u.get("ai_confidence", "")}
    if node == "human_review":
        return {"approval": u.get("approval"), "message": u.get("message", ""),
                "retry_feedback": u.get("retry_feedback", ""),
                "retry_count": u.get("retry_count", 0)}
    return {}


def _consume(job: dict, stream) -> None:
    """Read a LangGraph update stream, recording each finished step on the job."""
    for chunk in stream:
        if "__interrupt__" in chunk:
            job["review"] = chunk["__interrupt__"][0].value
            job["status"] = "awaiting_review"
            continue
        for node, u in chunk.items():
            if u is None:
                continue
            tr = (u.get("trace") or [{}])[0]
            job["steps"].append({"node": node, "data": _summarise(node, u),
                                 "seconds": tr.get("seconds"),
                                 "tokens": (tr.get("prompt_tokens", 0) or 0) + (tr.get("completion_tokens", 0) or 0)})
            if node == "visualizer":
                with open(u["network_html"], encoding="utf-8") as f:
                    job["network"] = f.read()
            if u.get("status") in ("no_exposure", "rejected", "completed"):
                job["status"] = u["status"]
                job["message"] = u.get("message", "")
    if job["status"] == "running":                 # stream ended without a final status
        job["status"] = "completed"


def _work(job: dict, make_stream) -> None:
    try:
        _consume(job, make_stream())
    except Exception as e:                         # show the error on the page instead of dying silently
        traceback.print_exc()
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"


def start_job(article: str, focal: str) -> str:
    job_id = uuid.uuid4().hex[:10]
    job = {"id": job_id, "status": "running", "steps": [], "review": None, "network": None,
           "message": "", "error": None, "started": time.time()}
    JOBS[job_id] = job
    cfg = {"configurable": {"thread_id": job_id}}
    threading.Thread(target=_work, daemon=True, args=(
        job, lambda: PIPELINE.stream(State(article=article, focal_name=focal), cfg, stream_mode="updates"))).start()
    return job_id


def resume_job(job: dict, decision: dict) -> None:
    job["status"], job["review"] = "running", None
    cfg = {"configurable": {"thread_id": job["id"]}}
    threading.Thread(target=_work, daemon=True, args=(
        job, lambda: PIPELINE.stream(Command(resume=decision), cfg, stream_mode="updates"))).start()


def job_view(job: dict) -> dict:
    return {k: job[k] for k in ("id", "status", "steps", "review", "message", "error")} | {
        "has_network": job["network"] is not None, "elapsed": round(time.time() - job["started"], 1)}


# ------------------------------------------------------------------ analytics
def _analytics() -> dict:
    """Aggregate retry stats across all jobs."""
    total = len(JOBS)
    retried = 0
    retry_counts: list[int] = []
    outcomes = {"approved": 0, "edited": 0, "rejected": 0, "no_exposure": 0, "error": 0, "completed": 0}
    feedback_log: list[dict] = []
    edit_extraction_count = 0
    self_critique_used = 0

    for jid, job in JOBS.items():
        steps = job["steps"]
        review_steps = [s for s in steps if s["node"] == "human_review"]
        retries_in_job = sum(1 for s in review_steps if s["data"].get("approval") == "retry")
        if retries_in_job > 0:
            retried += 1
            retry_counts.append(retries_in_job)
        for s in review_steps:
            if s["data"].get("retry_feedback"):
                feedback_log.append({"job_id": jid, "feedback": s["data"]["retry_feedback"],
                                     "retry_count": s["data"].get("retry_count", 0)})
        if any(s["data"].get("approval") == "edit_extraction" for s in review_steps):
            edit_extraction_count += 1
        if any(s["node"] == "self_critique" for s in steps):
            self_critique_used += 1
        outcomes[job["status"]] = outcomes.get(job["status"], 0) + 1

    return {
        "total_jobs": total,
        "jobs_with_retries": retried,
        "retry_rate": round(retried / total, 2) if total else 0,
        "avg_retries": round(sum(retry_counts) / len(retry_counts), 2) if retry_counts else 0,
        "max_retries_used": max(retry_counts) if retry_counts else 0,
        "edit_extraction_count": edit_extraction_count,
        "self_critique_used": self_critique_used,
        "outcomes": outcomes,
        "recent_feedback": feedback_log[-10:],
    }


# ------------------------------------------------------------------ KG grounding impact (from saved `cli.py eval` runs)
# When the server starts with ground-entities on (RAG=1 or GROUND_ENTITIES=1), it runs this pair of
# eval configs itself, in the background, against whichever LLM is actually configured (Qwen3-8B,
# Azure, or the offline mock) — so the comparison reflects the real model, not stale sample data.
KG_LOCK = threading.Lock()
KG_STATUS: dict = {"status": "idle", "error": None}      # idle | running | done | error


def _load_eval(config: str) -> dict | None:
    suffix = "" if config == "baseline" else "_" + config
    p = OUT_DIR / f"eval_results{suffix}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _kg_cache_matches(llm_name: str) -> bool:
    base, ge = _load_eval("baseline"), _load_eval("ground-entities")
    return bool(base and ge and base.get("llm") == llm_name and ge.get("llm") == llm_name)


def _compute_kg_comparison() -> None:
    """Run the 20-article eval twice (baseline, then --ground-entities) against the same LLM
    the live pipeline uses, and save the results where _kg_impact() reads them from."""
    with KG_LOCK:
        if KG_STATUS["status"] == "running":
            return
        KG_STATUS.update(status="running", error=None)
    try:
        from scpoc import evaluate
        mock = PIPELINE.llm.name == "offline mock"
        evaluate.main(mock=mock, ground_entities=False)
        evaluate.main(mock=mock, ground_entities=True)
        KG_STATUS.update(status="done", error=None)
    except Exception as e:
        traceback.print_exc()
        KG_STATUS.update(status="error", error=f"{type(e).__name__}: {e}")


def _ensure_kg_comparison() -> None:
    """Called once at startup when ground-entities is on. Skips the (possibly expensive, real-API)
    recompute if the cached eval files already match the LLM in use; otherwise runs it in a
    background thread so server startup isn't blocked."""
    if _kg_cache_matches(PIPELINE.llm.name):
        KG_STATUS.update(status="done", error=None)
        return
    threading.Thread(target=_compute_kg_comparison, daemon=True).start()


def _kg_impact() -> dict:
    """Compare the saved baseline eval run against the saved ground-entities run, so the
    dashboard can show, with real numbers, what grounding entities against the knowledge
    graph buys you. Regenerated automatically at startup when ground-entities is on
    (see _ensure_kg_comparison), or by hand with `python cli.py eval [--mock]` and
    `python cli.py eval [--mock] --ground-entities`, or on demand via POST /api/kg-impact/refresh."""
    current_llm = PIPELINE.llm.name if PIPELINE else None
    base, ge = _load_eval("baseline"), _load_eval("ground-entities")
    if not base or not ge:
        return {"available": False, "status": KG_STATUS["status"], "error": KG_STATUS.get("error"),
               "current_llm": current_llm}
    bs, gs = base["summary"], ge["summary"]
    out = {
        "available": True, "status": KG_STATUS["status"], "error": KG_STATUS.get("error"),
        "current_llm": current_llm, "articles": bs["articles"], "llm": ge["llm"],
        "without_rag": {
            "extraction_f1": bs["extraction"]["overall"]["f1"],
            "exact_extraction": bs["extraction_exact_match"],
            "exposure_detected": bs["network_query"]["exposure_detected"],
            "false_positives_handled": bs["false_positives"]["correctly_ended_no_exposure"],
            "disrupted_count_match": bs["network_query"]["disrupted_count_match"],
            "top_supplier_match": bs["risk"]["top_supplier_match"],
            "top_high_supplier_match": bs["risk"]["top_high_supplier_match"],
        },
        "with_kg_grounding": {
            "extraction_f1": gs["extraction"]["overall"]["f1"],
            "exact_extraction": gs["extraction_exact_match"],
            "exposure_detected": gs["network_query"]["exposure_detected"],
            "false_positives_handled": gs["false_positives"]["correctly_ended_no_exposure"],
            "disrupted_count_match": gs["network_query"]["disrupted_count_match"],
            "top_supplier_match": gs["risk"]["top_supplier_match"],
            "top_high_supplier_match": gs["risk"]["top_high_supplier_match"],
        },
    }
    eg = gs.get("retrieval", {}).get("entity_grounding")
    if eg:
        out["with_kg_grounding"]["entity_grounding_precision"] = eg["precision"]
        out["with_kg_grounding"]["entity_grounding_recall"] = eg["recall"]
    return out


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):             # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/status":
            return self._json({"llm": PIPELINE.llm.name, "mock": PIPELINE.llm.name == "offline mock"})
        if path == "/api/scenarios":
            return self._json({"scenarios": SCENARIOS, "grocers": GROCERS})
        if path.startswith("/api/job/"):
            job = JOBS.get(path.rsplit("/", 1)[1])
            return self._json(job_view(job)) if job else self._json({"error": "unknown job"}, 404)
        if path.startswith("/api/network/"):
            job = JOBS.get(path.rsplit("/", 1)[1])
            if job and job["network"]:
                return self._send(200, job["network"].encode("utf-8"), "text/html; charset=utf-8")
            return self._json({"error": "no network yet"}, 404)
        if path == "/api/analytics":
            return self._json(_analytics())
        if path == "/api/kg-impact":
            return self._json(_kg_impact())
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON"}, 400)
        if path == "/api/run":
            article, focal = (body.get("article") or "").strip(), body.get("focal") or GROCERS[0]
            if not article:
                return self._json({"error": "article is empty"}, 400)
            if focal not in GROCERS:
                return self._json({"error": f"unknown grocer {focal!r}"}, 400)
            return self._json({"job_id": start_job(article, focal)})
        if path == "/api/review":
            job = JOBS.get(body.get("job_id", ""))
            if not job:
                return self._json({"error": "unknown job"}, 404)
            if job["status"] != "awaiting_review":
                return self._json({"error": "job is not waiting for review"}, 409)
            decision = body.get("decision")
            if decision not in ("approve", "edit", "reject", "retry", "edit_extraction"):
                return self._json({"error": "decision must be approve, edit, reject, retry or edit_extraction"}, 400)
            resume_job(job, {k: v for k, v in body.items() if k in ("decision", "csco", "actions", "reason", "monitoring")})
            return self._json({"ok": True})
        if path == "/api/kg-impact/refresh":
            threading.Thread(target=_compute_kg_comparison, daemon=True).start()
            return self._json({"ok": True})
        self._json({"error": "not found"}, 404)


from pathlib import Path as _Path
PAGE = (_Path(__file__).resolve().parent / "dashboard.html").read_text(encoding="utf-8")


def main():
    global PIPELINE
    from scpoc.llm import _load_dotenv
    _load_dotenv()                                 # so RAG=1 etc. can also live in .env
    on = lambda k: os.environ.get(k, "").strip().lower() in ("1", "true", "yes")
    PIPELINE = build_app(g=tools.get_graph(), rag_examples=on("RAG_EXAMPLES") or on("RAG"),
                         ground_entities=on("GROUND_ENTITIES") or on("RAG"))
    rag_on = [k for k, v in PIPELINE.rag.items() if v]
    kind = PIPELINE.llm.name + (f", RAG: {'+'.join(rag_on)}" if rag_on else "")
    if PIPELINE.rag.get("ground_entities"):
        _ensure_kg_comparison()      # live baseline-vs-grounding comparison, computed in the background
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Supply chain disruption monitor on http://{HOST}:{PORT}  ({kind})  - Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
