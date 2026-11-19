"""Supply-chain disruption monitor - demo dashboard.

Single file, standard library only (same structure as Hermes): an embedded HTML page
plus a small JSON API, served by http.server.

    python app.py            ->  http://localhost:8080   (set PORT to change)

Uses Qwen3-8B if QWEN_API_KEY is set (env or .env), else Azure OpenAI if its variables are
set, else the offline mock LLM. The pipeline itself lives in scpoc/.
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

_PAGE_UNUSED = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supply Chain Disruption Monitor (unused)</title>
<style>
:root{--blue:#2563eb;--blue-light:#dbeafe;--blue-dark:#1d4ed8;--red:#dc2626;--red-light:#fef2f2;--red-border:#fca5a5;
 --amber:#d97706;--amber-light:#fffbeb;--amber-border:#fcd34d;--green:#059669;--green-light:#ecfdf5;--green-border:#6ee7b7;
 --gray-50:#f9fafb;--gray-100:#f3f4f6;--gray-200:#e5e7eb;--gray-300:#d1d5db;--gray-400:#9ca3af;--gray-500:#6b7280;
 --gray-600:#4b5563;--gray-700:#374151;--gray-800:#1f2937;--gray-900:#111827;
 --radius:10px;--radius-lg:14px;--shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);
 --shadow-md:0 4px 6px -1px rgba(0,0,0,.08),0 2px 4px -2px rgba(0,0,0,.06);
 --shadow-lg:0 10px 15px -3px rgba(0,0,0,.08),0 4px 6px -4px rgba(0,0,0,.06)}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--gray-50);color:var(--gray-800);line-height:1.6}
header{background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);color:#fff;padding:0}
.header-inner{max-width:1200px;margin:0 auto;padding:16px 24px;display:flex;justify-content:space-between;align-items:center}
header h1{font-size:20px;font-weight:700;letter-spacing:-.3px;display:flex;align-items:center;gap:10px}
header h1 .icon{width:32px;height:32px;background:linear-gradient(135deg,var(--blue),#7c3aed);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px}
.badge{font-size:11px;padding:4px 12px;border-radius:20px;background:rgba(255,255,255,.12);backdrop-filter:blur(4px);font-weight:500;letter-spacing:.3px}
main{max-width:1200px;margin:0 auto;padding:24px}
.card{background:#fff;border:1px solid var(--gray-200);border-radius:var(--radius-lg);padding:20px 24px;margin-bottom:16px;box-shadow:var(--shadow);transition:box-shadow .2s}
.card:hover{box-shadow:var(--shadow-md)}
.card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.card-header h3{font-size:14px;font-weight:600;color:var(--gray-800);display:flex;align-items:center;gap:8px}
.card-header .step-num{width:24px;height:24px;border-radius:6px;background:var(--blue);color:#fff;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:700}
.card-header .meta{font-size:11px;color:var(--gray-400);font-weight:500;display:flex;gap:12px}
.card-header .meta span{display:flex;align-items:center;gap:3px}
textarea{width:100%;font:14px/1.6 'Inter',system-ui,sans-serif;padding:10px 14px;border:1px solid var(--gray-300);border-radius:var(--radius);resize:vertical;transition:border-color .15s,box-shadow .15s;background:var(--gray-50)}
textarea:focus{outline:none;border-color:var(--blue);box-shadow:0 0 0 3px rgba(37,99,235,.1)}
select{font:13px/1.4 'Inter',system-ui,sans-serif;padding:6px 10px;border:1px solid var(--gray-300);border-radius:var(--radius);background:#fff;cursor:pointer}
select:focus{outline:none;border-color:var(--blue)}
button{font:13px/1 'Inter',system-ui,sans-serif;font-weight:600;padding:9px 18px;border-radius:var(--radius);border:1px solid var(--gray-300);background:#fff;cursor:pointer;transition:all .15s;display:inline-flex;align-items:center;gap:6px}
button:hover{transform:translateY(-1px);box-shadow:var(--shadow)}
button:active{transform:translateY(0)}
button:disabled{opacity:.45;cursor:default;transform:none;box-shadow:none}
.btn-primary{background:var(--blue);color:#fff;border-color:var(--blue)}
.btn-primary:hover{background:var(--blue-dark)}
.btn-green{background:var(--green);color:#fff;border-color:var(--green)}
.btn-green:hover{background:#047857}
.btn-red{background:var(--red);color:#fff;border-color:var(--red)}
.btn-red:hover{background:#b91c1c}
.btn-amber{background:var(--amber);color:#fff;border-color:var(--amber)}
.btn-amber:hover{background:#b45309}
.btn-outline{background:transparent;color:var(--gray-600);border-color:var(--gray-300)}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.chip{display:inline-flex;align-items:center;background:var(--gray-100);border:1px solid var(--gray-200);border-radius:20px;padding:2px 10px;font-size:11px;font-weight:500;color:var(--gray-700)}
.chip.warn{background:var(--red-light);border-color:var(--red-border);color:var(--red)}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-transform:uppercase;font-size:10px;letter-spacing:.5px;color:var(--gray-500);font-weight:600;padding:8px 10px;border-bottom:2px solid var(--gray-200);text-align:left}
td{padding:8px 10px;border-bottom:1px solid var(--gray-100);color:var(--gray-700)}
tr:hover td{background:var(--gray-50)}
.HIGH{color:var(--red);font-weight:700}.MEDIUM{color:var(--amber);font-weight:700}.LOW{color:var(--green);font-weight:700}
iframe{width:100%;height:520px;border:1px solid var(--gray-200);border-radius:var(--radius)}
.muted{color:var(--gray-400);font-size:12px}
.err{color:var(--red)}
pre{white-space:pre-wrap;margin:4px 0;font:12px/1.5 'SF Mono','Fira Code',monospace;background:var(--gray-50);padding:6px 10px;border-radius:6px;border:1px solid var(--gray-200)}
.alert{border-radius:var(--radius);padding:12px 16px;font-size:13px;margin-bottom:14px;display:flex;align-items:flex-start;gap:10px}
.alert-warn{background:var(--amber-light);border:1px solid var(--amber-border);color:#92400e}
.alert-info{background:var(--blue-light);border:1px solid #93c5fd;color:#1e40af}
.alert-err{background:var(--red-light);border:1px solid var(--red-border);color:#991b1b}
.alert-ok{background:var(--green-light);border:1px solid var(--green-border);color:#065f46}
.alert .icon{font-size:16px;line-height:1;flex-shrink:0;margin-top:1px}
.scenarios-bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.scenarios-bar button{font-size:12px;padding:6px 14px;border-radius:20px;font-weight:500}
.scenarios-bar button:hover{background:var(--blue-light);border-color:var(--blue);color:var(--blue)}
.grocer-row{display:flex;gap:14px;align-items:center;margin-top:14px}
.grocer-row label{font-size:13px;font-weight:500;color:var(--gray-600);display:flex;align-items:center;gap:6px}
.status-pill{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:600;padding:4px 12px;border-radius:20px}
.status-running{background:#eff6ff;color:var(--blue)}
.status-done{background:var(--green-light);color:var(--green)}
.status-stop{background:var(--amber-light);color:var(--amber)}
.status-fail{background:var(--red-light);color:var(--red)}
.outcome-card{border-left:4px solid;padding:16px 20px}
.outcome-card.done{border-left-color:var(--green)}.outcome-card.stop{border-left-color:var(--amber)}.outcome-card.fail{border-left-color:var(--red)}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid var(--gray-300);border-top-color:var(--blue);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.divider{border:none;border-top:2px dashed var(--amber-border);margin:20px 0;position:relative}
.divider-label{position:absolute;top:-11px;left:24px;background:var(--gray-50);padding:0 12px;font-size:12px;font-weight:700;color:var(--amber);letter-spacing:.5px;text-transform:uppercase}
.history-toggle{font-size:12px;color:var(--blue);cursor:pointer;font-weight:600;display:inline-flex;align-items:center;gap:4px;padding:4px 0;border:none;background:none}
.history-toggle:hover{text-decoration:underline}
.history-section{overflow:hidden;transition:max-height .3s ease;border:1px dashed var(--gray-300);border-radius:var(--radius);padding:0 16px;margin-bottom:12px;background:var(--gray-50)}
.history-section.collapsed{max-height:0;padding:0 16px;border-color:transparent}
.history-section.expanded{max-height:5000px;padding:14px 16px}
.history-section .card{background:var(--gray-50);border:1px solid var(--gray-200);box-shadow:none;opacity:.85}
.diff-badge{display:inline-flex;align-items:center;gap:4px;font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;text-transform:uppercase;letter-spacing:.5px}
.diff-added{background:#dcfce7;color:#166534;border:1px solid #86efac}
.diff-removed{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5}
.diff-same{background:var(--gray-100);color:var(--gray-500);border:1px solid var(--gray-200)}
.review-panel{background:#fff;border:2px solid var(--blue);border-radius:var(--radius-lg);padding:24px;margin-bottom:16px;box-shadow:var(--shadow-lg)}
.review-panel h3{font-size:16px;font-weight:700;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.review-actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;padding-top:16px;border-top:1px solid var(--gray-200)}
.feedback-section{margin-top:16px;padding:14px;background:var(--gray-50);border-radius:var(--radius);border:1px solid var(--gray-200)}
.feedback-section label{font-size:12px;font-weight:600;color:var(--gray-600);text-transform:uppercase;letter-spacing:.5px;display:block;margin-bottom:6px}
.feedback-section textarea{height:70px;background:#fff;font-size:13px}
</style></head><body>
<header><div class="header-inner">
 <h1><span class="icon">&#x26a0;&#xfe0f;</span> Supply Chain Disruption Monitor</h1>
 <span class="badge" id="llm"></span>
</div></header>
<main>
 <div id="mockbanner" class="alert alert-warn" style="display:none">
  <span class="icon">&#x26a0;&#xfe0f;</span>
  <div>Running with the <b>offline mock LLM</b> (keyword-based). Good for testing the flow. Set <code>QWEN_API_KEY</code> in <code>.env</code> for real analysis.</div>
 </div>
 <div class="card">
  <div class="card-header"><h3><span class="step-num">1</span> Input Article</h3></div>
  <div class="scenarios-bar"><span class="muted" style="padding:6px 0">Scenarios:</span><span id="scen"></span></div>
  <textarea id="article" rows="5" placeholder="Paste a news article about a potential supply chain disruption..."></textarea>
  <div class="grocer-row">
   <label>&#x1f3ea; Grocer <select id="focal"></select></label>
   <button class="btn-primary" id="run" onclick="run()">&#x25b6; Run Analysis</button>
   <span id="runmsg" class="muted"></span>
  </div>
 </div>
 <div id="steps"></div>
 <div id="review"></div>
 <div id="outcome"></div>
</main>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const chips=(a,cls='')=>(a&&a.length?a.map(x=>`<span class="chip ${cls}">${esc(x)}</span>`).join(' '):'<span class="muted">none</span>');

const STEP_META={
 retrieve:{num:'R',title:'Retrieval (RAG)',icon:'&#x1f50d;'},
 monitoring:{num:'2',title:'Monitoring Agent',icon:'&#x1f916;'},
 kg_query:{num:'3',title:'Knowledge Graph Query',icon:'&#x1f578;&#xfe0f;'},
 risk:{num:'4',title:'Risk Scoring',icon:'&#x1f4ca;'},
 visualizer:{num:'5',title:'Network Diagram',icon:'&#x1f310;'},
 csco:{num:'6',title:'CSCO Summary',icon:'&#x1f4dd;'},
 no_exposure:{num:'-',title:'No Exposure',icon:'&#x2705;'},
 alternatives:{num:'8',title:'Alternative Sourcing',icon:'&#x1f504;'},
 human_review:{num:'7',title:'Human Review Decision',icon:'&#x1f9d1;'}
};

let job=null,timer=null,scen=[],reviewShown=false;

async function init(){
 const st=await(await fetch('/api/status')).json();
 $('llm').textContent=st.llm+(st.mock?' (mock)':'');
 if(st.mock)$('mockbanner').style.display='flex';
 const d=await(await fetch('/api/scenarios')).json();scen=d.scenarios;
 $('focal').innerHTML=d.grocers.map(g=>`<option>${esc(g)}</option>`).join('');
 $('scen').innerHTML=scen.map((s,i)=>`<button class="btn-outline" onclick="pick(${i})">${esc(s.title)}</button>`).join('');
}
function pick(i){$('article').value=scen[i].article;$('focal').value=scen[i].focal;}

async function run(){
 const article=$('article').value.trim();
 if(!article){$('runmsg').textContent='Please paste or pick an article first.';return}
 $('runmsg').textContent='';$('run').disabled=true;
 $('steps').innerHTML='';$('review').innerHTML='';$('outcome').innerHTML='';
 reviewShown=false;
 const r=await fetch('/api/run',{method:'POST',body:JSON.stringify({article,focal:$('focal').value})});
 const j=await r.json();
 if(j.error){$('runmsg').textContent=j.error;$('run').disabled=false;return}
 job=j.job_id;poll();
}

function poll(){
 clearInterval(timer);
 timer=setInterval(async()=>{
  const v=await(await fetch('/api/job/'+job)).json();render(v);
  if(v.status!=='running'){clearInterval(timer);if(v.status!=='awaiting_review')$('run').disabled=false}
 },500);
}

function metaHtml(s){
 const parts=[];
 if(s.seconds!=null)parts.push(`<span>&#x23f1; ${s.seconds}s</span>`);
 if(s.tokens)parts.push(`<span>&#x1f4ac; ${s.tokens} tok</span>`);
 return parts.join('');
}

function stepHeader(node,s){
 const m=STEP_META[node]||{num:'?',title:node,icon:''};
 return `<div class="card-header"><h3><span class="step-num">${m.num}</span>${m.icon} ${m.title}</h3><div class="meta">${metaHtml(s)}</div></div>`;
}

function body(n,d){
 if(n==='retrieve')return `${d.examples.length?`<div style="margin-bottom:6px"><b>Similar solved examples</b> ${chips(d.examples.map(e=>e.id+' ('+e.score+')'))}</div>`:''}
  ${d.grounding?`<div><b>Companies named</b> ${chips(d.entities)}</div>`:''}`;
 if(n==='monitoring')return `<div style="margin-bottom:8px">
  <span class="chip" style="font-size:12px;font-weight:600">${d.is_disruption?'&#x1f6a8; Disruption detected':'&#x2705; No disruption'}</span>
  <span class="chip">${esc(d.disruption_type)}</span></div>
  <div style="margin-bottom:6px;font-size:14px">${esc(d.summary)}</div>
  <div style="margin-bottom:4px"><b style="font-size:11px;color:var(--gray-500)">COUNTRIES</b> ${chips(d.countries)}</div>
  <div style="margin-bottom:4px"><b style="font-size:11px;color:var(--gray-500)">INDUSTRIES</b> ${chips(d.industries)}</div>
  <div style="margin-bottom:4px"><b style="font-size:11px;color:var(--gray-500)">COMPANIES</b> ${chips(d.companies)}</div>
  <details style="margin-top:6px"><summary class="muted" style="cursor:pointer">Reasoning</summary><div class="muted" style="margin-top:4px">${esc(d.reasoning)}</div></details>`;
 if(n==='kg_query')return `<div style="margin-bottom:8px;font-size:14px">Focal: <b>${esc(d.focal)}</b> &mdash; <b style="font-size:18px">${d.disrupted_count}</b> disrupted suppliers</div>
  <div class="row" style="gap:16px;margin-bottom:8px">
   <span class="chip">Tier-1: ${d.per_tier[1]}</span><span class="chip">Tier-2: ${d.per_tier[2]}</span>
   <span class="chip">Tier-3: ${d.per_tier[3]}</span><span class="chip">Tier-4: ${d.per_tier[4]}</span></div>
  <div style="margin-bottom:6px"><b style="font-size:11px;color:var(--gray-500)">MATCHED</b> ${chips([...d.criteria.countries,...d.criteria.industries,...d.criteria.companies])}</div>
  ${d.unresolved.length?`<div style="margin-bottom:6px"><b style="font-size:11px;color:var(--gray-500)">NOT IN GRAPH</b> ${chips(d.unresolved,'warn')}</div>`:''}
  ${d.sample_chains.map(c=>`<pre>${esc(c.join('  \u2190  '))}</pre>`).join('')}`;
 if(n==='risk')return `<div style="margin-bottom:8px">${d.exposed_tier1_count} Tier-1 suppliers exposed (top ${d.top_suppliers.length} shown)</div>
  <div style="overflow-x:auto"><table><tr><th>Supplier</th><th>Industry</th><th>Score</th><th>Level</th><th>Breadth</th><th>Depend.</th><th>Crit.</th>
  <th>Central.</th><th>Depth</th></tr>${d.top_suppliers.map(s=>`<tr><td><b>${esc(s.name)}</b></td><td>${esc(s.industry)}</td>
  <td><b>${s.risk_score}</b></td><td class="${s.risk_level}">${s.risk_level}</td><td>${s.components.exposure_breadth}</td>
  <td>${s.components.dependency_ratio}</td><td>${s.components.downstream_criticality}</td><td>${s.components.tier1_centrality}</td>
  <td>${s.components.exposure_depth}</td></tr>`).join('')}</table></div>
  <div class="muted" style="margin-top:6px">Thresholds: HIGH &ge; ${d.thresholds.HIGH} &middot; MEDIUM &ge; ${d.thresholds.MEDIUM}</div>`;
 if(n==='visualizer')return `<iframe src="/api/network/${job}"></iframe>`;
 if(n==='csco')return `<div style="font-size:14px;margin-bottom:8px">${esc(d.executive_summary)}</div>
  <div style="color:var(--gray-600);margin-bottom:8px">${esc(d.network_impact)}</div>
  <ul style="padding-left:18px;font-size:13px">${d.key_recommendations.map(x=>`<li style="margin-bottom:4px">${esc(x)}</li>`).join('')}</ul>`;
 if(n==='no_exposure')return `<div style="font-size:14px">${esc(d.message)}</div>`;
 if(n==='alternatives')return d.alternatives.length?d.alternatives.map(a=>`<div style="margin-bottom:12px">
  <div style="font-size:14px;margin-bottom:6px"><b>${esc(a.supplier.name)}</b> (${esc(a.supplier.industry)}) &mdash; ${a.candidates_considered} candidates, <b>${a.clear_count}</b> fully clear</div>
  <div style="overflow-x:auto"><table><tr><th>Candidate</th><th>Country</th><th>Chain Clear</th><th>Disrupted Upstream</th><th>Upstream Size</th></tr>
  ${a.alternatives.map(c=>`<tr><td><b>${esc(c.name)}</b></td><td>${esc(c.country)}</td><td>${c.verified_clear?'<span style="color:var(--green)">&#x2705; Yes</span>':'<span class="err">&#x274c; No</span>'}</td>
  <td>${c.disrupted_upstream}</td><td>${c.upstream_size}</td></tr>`).join('')}</table></div></div>`).join('')
  :'<div class="muted">No HIGH-risk suppliers &mdash; no replacements needed.</div>';
 return '';
}

/* ---- Render: group steps into attempts for history ---- */
function splitAttempts(steps){
 const attempts=[]; let cur=[];
 for(const s of steps){
  if(s.node==='human_review'&&s.data.approval==='retry'){
   cur.push(s); attempts.push(cur); cur=[];
  } else { cur.push(s); }
 }
 if(cur.length)attempts.push(cur);
 return attempts;
}

function renderAttempt(steps,attemptNum,isCurrent){
 return steps.map(s=>{
  if(s.node==='human_review'&&s.data.approval==='retry'){
   return `<div class="alert alert-warn" style="margin:12px 0">
    <span class="icon">&#x1f504;</span>
    <div><b>Reviewer requested AI retry</b> (attempt ${s.data.retry_count})<br>
    <span style="font-style:italic">"${esc(s.data.retry_feedback)}"</span></div></div>`;
  }
  if(s.node==='human_review')return '';
  const content=body(s.node,s.data);
  return `<div class="card">${stepHeader(s.node,s)}${content}</div>`;
 }).join('');
}

function diffChip(oldVal,newVal,label){
 if(JSON.stringify(oldVal)===JSON.stringify(newVal))return `<span class="diff-badge diff-same">${label}: unchanged</span>`;
 return `<span class="diff-badge diff-removed">${label} before: ${esc(JSON.stringify(oldVal))}</span>
  <span class="diff-badge diff-added">${label} now: ${esc(JSON.stringify(newVal))}</span>`;
}

function renderHistory(attempts){
 if(attempts.length<=1)return '';
 let html='';
 for(let i=0;i<attempts.length-1;i++){
  const a=attempts[i];
  const mon=a.find(s=>s.node==='monitoring');
  const kg=a.find(s=>s.node==='kg_query');
  const retry=a.find(s=>s.node==='human_review'&&s.data.approval==='retry');
  const nextMon=attempts[i+1].find(s=>s.node==='monitoring');
  const nextKg=attempts[i+1].find(s=>s.node==='kg_query');
  const hid='hist_'+i;
  html+=`<div style="margin-bottom:16px">
   <div class="divider"><span class="divider-label">Attempt ${i+1} of ${attempts.length}</span></div>
   ${retry?`<div class="alert alert-warn"><span class="icon">&#x1f504;</span><div><b>Reviewer feedback:</b> "${esc(retry.data.retry_feedback)}"</div></div>`:''}
   ${mon&&nextMon?`<div style="margin-bottom:8px;display:flex;gap:6px;flex-wrap:wrap">
    <b style="font-size:12px;color:var(--gray-500)">CHANGES:</b>
    ${diffChip(mon.data.countries,nextMon.data.countries,'Countries')}
    ${diffChip(mon.data.industries,nextMon.data.industries,'Industries')}
    ${diffChip(mon.data.companies,nextMon.data.companies,'Companies')}
    ${kg&&nextKg?diffChip(kg.data.disrupted_count,nextKg.data.disrupted_count,'Disrupted'):''}
   </div>`:''}
   <button class="history-toggle" onclick="toggleHist('${hid}')">&#x25b6; Show full details for attempt ${i+1}</button>
   <div id="${hid}" class="history-section collapsed">${renderAttempt(a,i+1,false)}</div>
  </div>`;
 }
 return html;
}

function toggleHist(id){
 const el=$(id); if(!el)return;
 const btn=el.previousElementSibling;
 if(el.classList.contains('collapsed')){
  el.classList.remove('collapsed');el.classList.add('expanded');
  btn.innerHTML='&#x25bc; Hide details for'+btn.innerHTML.slice(btn.innerHTML.indexOf(' attempt'));
 } else {
  el.classList.remove('expanded');el.classList.add('collapsed');
  btn.innerHTML='&#x25b6; Show full details for'+btn.innerHTML.slice(btn.innerHTML.indexOf(' attempt'));
 }
}

function render(v){
 const attempts=splitAttempts(v.steps);
 const hasRetries=attempts.length>1;
 const currentAttempt=attempts[attempts.length-1];
 let stepsHtml='';
 if(hasRetries){
  stepsHtml+=renderHistory(attempts);
  stepsHtml+=`<div class="divider"><span class="divider-label">&#x2705; Current \u2014 Attempt ${attempts.length}</span></div>`;
 }
 stepsHtml+=renderAttempt(currentAttempt,attempts.length,true);
 if(v.status==='running')stepsHtml+=`<div class="card" style="text-align:center;padding:24px"><div class="spinner"></div><div class="muted" style="margin-top:8px">Analysing...</div></div>`;
 $('steps').innerHTML=stepsHtml;
 if(v.status==='awaiting_review'&&!reviewShown){reviewShown=true;showReview(v.review,attempts)}
 if(v.status!=='awaiting_review')$('review').innerHTML='';
 const cls={completed:'done',no_exposure:'stop',rejected:'stop',error:'fail'}[v.status];
 const icons={completed:'&#x2705;',no_exposure:'&#x1f7e1;',rejected:'&#x1f6d1;',error:'&#x274c;'}[v.status]||'';
 $('outcome').innerHTML=cls?`<div class="card outcome-card ${cls}"><span style="font-size:16px">${icons}</span> <b>${v.status.replace('_',' ')}</b> &mdash; ${esc(v.error||v.message)}</div>`:'';
}

const ACTIONS={HIGH:'Replace \u2014 source alternative supplier',MEDIUM:'Increase monitoring',LOW:'Maintain standard operations'};

function showReview(rv,attempts){
 const retries=rv.retry_count||0, maxR=rv.max_retries||3, canRetry=retries<maxR;
 const prevMon=attempts&&attempts.length>1?attempts[attempts.length-2].find(s=>s.node==='monitoring'):null;
 const curMon=attempts?attempts[attempts.length-1].find(s=>s.node==='monitoring'):null;
 let diffHtml='';
 if(prevMon&&curMon){
  diffHtml=`<div class="alert alert-info"><span class="icon">&#x1f4ca;</span><div><b>Changes from previous attempt:</b><br>
   ${diffChip(prevMon.data.countries,curMon.data.countries,'Countries')}
   ${diffChip(prevMon.data.industries,curMon.data.industries,'Industries')}
   ${diffChip(prevMon.data.companies,curMon.data.companies,'Companies')}
  </div></div>`;
 }
 $('review').innerHTML=`<div class="review-panel">
  <h3>&#x1f9d1; Human Review ${retries?`<span class="status-pill status-running">Attempt ${retries+1} of ${maxR+1}</span>`:''}</h3>
  ${retries?`<div class="alert alert-info"><span class="icon">&#x1f504;</span><div>This is <b>AI attempt #${retries+1}</b> after ${retries} retry${retries>1?'ies':''}. Review the updated analysis below.</div></div>`:''}
  ${diffHtml}
  <div style="margin-bottom:14px">
   <label style="font-size:12px;font-weight:600;color:var(--gray-500);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;display:block">Executive Summary (editable)</label>
   <textarea id="rv_sum" rows="4">${esc(rv.csco.executive_summary)}</textarea>
  </div>
  <div style="overflow-x:auto;margin-bottom:4px"><table><tr><th>Supplier</th><th>Score</th><th>Action</th></tr>
  ${rv.actions.map((a,i)=>`<tr><td><b>${esc(a.name)}</b></td><td>${a.risk_score}</td>
  <td><select id="rv_a${i}">${['HIGH','MEDIUM','LOW'].map(l=>`<option value="${l}" ${l===a.risk_level?'selected':''}>${l}: ${ACTIONS[l]}</option>`).join('')}</select></td></tr>`).join('')}</table></div>
  <div class="feedback-section">
   <label>&#x1f4ac; Feedback for AI Retry</label>
   <textarea id="rv_feedback" placeholder="Tell the AI what to fix, e.g.: 'Japan is only a buyer, not a disrupted country. Remove it from the countries list.'"></textarea>
  </div>
  <div class="review-actions">
   <button class="btn-green" onclick='decide("approve")'>&#x2705; Approve</button>
   <button class="btn-primary" onclick='decide("edit")'>&#x270f;&#xfe0f; Approve with Edits</button>
   <button class="btn-red" onclick='decide("reject")'>&#x1f6d1; Reject</button>
   ${canRetry?`<button class="btn-amber" onclick='decide("retry")'>&#x1f504; Reject &amp; Ask AI to Retry (${maxR-retries} left)</button>`
   :`<button disabled style="opacity:.4">&#x1f504; No retries remaining</button>`}
  </div>
 </div>`;
 window._rv=rv;
}

async function decide(decision){
 const rv=window._rv,msg={job_id:job,decision};
 if(decision==='edit'){
  msg.csco={executive_summary:$('rv_sum').value};
  msg.actions=rv.actions.map((a,i)=>{const l=$('rv_a'+i).value;return{...a,risk_level:l,action:ACTIONS[l]}});
 }
 if(decision==='reject')msg.reason='Rejected by reviewer.';
 if(decision==='retry'){
  const fb=($('rv_feedback')?$('rv_feedback').value:'').trim();
  if(!fb){
   $('rv_feedback').style.borderColor='var(--red)';
   $('rv_feedback').setAttribute('placeholder','Please enter feedback telling the AI what to fix before retrying.');
   $('rv_feedback').focus();return;
  }
  msg.reason=fb;
 }
 $('review').innerHTML=decision==='retry'
  ?`<div class="card" style="text-align:center;padding:24px"><div class="spinner"></div><div style="margin-top:8px;font-weight:600;color:var(--amber)">&#x1f504; AI is re-analysing with your feedback...</div></div>`
  :`<div class="card" style="text-align:center;padding:16px"><div class="spinner"></div><div class="muted" style="margin-top:6px">Processing...</div></div>`;
 const r=await fetch('/api/review',{method:'POST',body:JSON.stringify(msg)});
 if(!(await r.json()).ok){$('review').innerHTML=`<div class="alert alert-err"><span class="icon">&#x274c;</span><div>Could not submit the decision. Please try again.</div></div>`;return}
 reviewShown=false;poll();
}
init();
</script></body></html>"""


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
