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
        return {"examples": [{"id": e["id"], "kind": e["kind"], "score": e["score"]} for e in r.get("examples", [])],
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
        return u["csco"].model_dump()
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
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Supply chain disruption monitor on http://{HOST}:{PORT}  ({kind})  - Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
