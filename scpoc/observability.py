"""Optional Langfuse tracing. Off (and free) unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

    pip install langfuse
    LANGFUSE_PUBLIC_KEY=pk-lf-...   LANGFUSE_SECRET_KEY=sk-lf-...   LANGFUSE_HOST=https://cloud.langfuse.com

What you get in Langfuse:
  * one trace per run (it spans the pause for human review), grouped by session
  * a span per pipeline step, and a generation (prompt, output, tokens) per LLM call inside it
  * scores: human decision, retry count, fake-news verdict, top risk score, final status,
    and per-article evaluation metrics for `cli.py eval --langfuse-run NAME`
  * prompts fetched from Langfuse prompt management, with the text in prompts.py as the fallback
  * the evaluation set as a Langfuse dataset (`cli.py langfuse-sync-dataset`)

Design rules: nothing here may break a run. Every Langfuse call is wrapped, failures are logged
once, and with no keys every helper is a no-op. Spans use an explicit trace id carried in the
graph State, so a run keeps one trace across the human-review pause and across threads.
Set LANGFUSE_MASK_ARTICLES=1 to keep article text and prompts out of Langfuse.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
import uuid

_STATE = {"client": None, "tried": False, "warned": False}
RUN_NAME: str | None = None          # set by `cli.py eval --langfuse-run NAME`: groups a whole eval as one session
MASK = "[hidden: LANGFUSE_MASK_ARTICLES=1]"


def _warn(msg: str) -> None:
    if not _STATE["warned"]:
        _STATE["warned"] = True
        print(f"[langfuse] {msg} (tracing continues to be skipped for failing calls)", file=sys.stderr)


def client():
    """The Langfuse client, or None when keys are missing or the package is not installed."""
    if _STATE["tried"]:
        return _STATE["client"]
    _STATE["tried"] = True
    try:
        from .llm import _load_dotenv
        _load_dotenv()                                    # keys may live in .env
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            from langfuse import Langfuse
            _STATE["client"] = Langfuse()
    except Exception as e:                                # package missing, bad config, ...
        _warn(f"disabled: {type(e).__name__}: {e}")
    return _STATE["client"]


def enabled() -> bool:
    return client() is not None


def reset() -> None:
    """Forget the client (used by tests)."""
    _STATE.update(client=None, tried=False, warned=False, prompts_down_until=0)


def _safe(fn, *a, default=None, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        _warn(f"call failed: {type(e).__name__}: {e}")
        return default


def _hide(x):
    return MASK if os.environ.get("LANGFUSE_MASK_ARTICLES", "").strip().lower() in ("1", "true", "yes") else x


# ------------------------------------------------------------------ traces
def new_trace(session_id: str) -> tuple[str | None, str | None]:
    """(trace_id, session_id) for a new run; (None, None) when tracing is off."""
    lf = client()
    if not lf:
        return None, None
    tid = _safe(lf.create_trace_id, seed=f"{session_id}:{uuid.uuid4().hex}")
    return tid, RUN_NAME or session_id


def trace_url(trace_id: str | None) -> str | None:
    lf = client()
    return _safe(lf.get_trace_url, trace_id=trace_id) if lf and trace_id else None


@contextlib.contextmanager
def node_span(step: str, state):
    """A span for one pipeline step, inside the run's trace. Yields the span (or None)."""
    lf, tid = client(), getattr(state, "trace_id", None)
    if not lf or not tid:
        yield None
        return
    paused = None
    try:
        from langfuse import propagate_attributes
        meta = {"grocer": state.focal_name}
        tags = [RUN_NAME] if RUN_NAME else None
        cm = contextlib.ExitStack()
        cm.enter_context(propagate_attributes(session_id=state.session_id, trace_name="supply-chain-run",
                                              metadata=meta, tags=tags))
        span = cm.enter_context(lf.start_as_current_observation(
            trace_context={"trace_id": tid}, name=step, as_type="span"))
    except Exception as e:
        _warn(f"span failed: {type(e).__name__}: {e}")
        yield None
        return
    try:
        try:
            yield span
        except BaseException as e:
            if type(e).__name__ == "GraphInterrupt":      # the pause for human review is not an error
                _safe(span.update, output="paused for human review")
                paused = e
            else:
                _safe(span.update, level="ERROR", status_message=f"{type(e).__name__}: {e}"[:300])
                raise
    finally:
        _safe(cm.close)
    if paused is not None:
        raise paused


def note(span, **fields) -> None:
    """Add input / output / metadata to a span (no-op without one)."""
    if span is not None:
        _safe(span.update, **fields)


@contextlib.contextmanager
def generation(step: str, model: str, system: str, user: str, prompt=None):
    """A generation for one LLM call, nested under the current step span. Yields an object with .done()."""
    lf = client()
    if not lf:
        yield None
        return
    try:
        cm = lf.start_as_current_observation(
            name=f"llm:{step}", as_type="generation", model=model,
            input=_hide([{"role": "system", "content": system}, {"role": "user", "content": user}]),
            model_parameters={"temperature": 0}, prompt=prompt)
        gen = cm.__enter__()
    except Exception as e:
        _warn(f"generation failed: {type(e).__name__}: {e}")
        yield None
        return
    try:
        yield gen
    except BaseException as e:
        _safe(gen.update, level="ERROR", status_message=f"{type(e).__name__}: {e}"[:300])
        _safe(cm.__exit__, type(e), e, e.__traceback__)
        raise
    else:
        _safe(cm.__exit__, None, None, None)


def span_meta(**meta) -> None:
    """Add metadata to the span of the step that is currently running."""
    lf = client()
    if lf:
        _safe(lf.update_current_span, metadata=meta)


def event(name: str, **meta) -> None:
    """A point-in-time event on the current span (e.g. an API retry)."""
    lf = client()
    if lf:
        _safe(lf.create_event, name=name, metadata=meta)


# ------------------------------------------------------------------ scores
def score(trace_id: str | None, name: str, value, kind: str = "NUMERIC", comment: str | None = None) -> None:
    """kind: NUMERIC | CATEGORICAL | BOOLEAN. Booleans are sent as 1 / 0."""
    lf = client()
    if not lf or not trace_id:
        return
    if kind == "BOOLEAN":
        value = 1 if value else 0
    _safe(lf.create_score, trace_id=trace_id, name=name, value=value, data_type=kind, comment=comment)


def eval_scores(trace_id: str | None, row: dict) -> None:
    """Per-article evaluation metrics, attached to that article's trace."""
    if not trace_id or row.get("error"):
        return
    ex = row.get("extraction", {})
    tp = sum(c["tp"] for c in ex.values())
    fp = sum(c["fp"] for c in ex.values())
    fn = sum(c["fn"] for c in ex.values())
    p, r = (tp / (tp + fp) if tp + fp else 1.0), (tp / (tp + fn) if tp + fn else 1.0)
    score(trace_id, "eval_extraction_f1", round(2 * p * r / (p + r), 3) if p + r else 0.0)
    score(trace_id, "eval_extraction_exact", row.get("extraction_exact", False), "BOOLEAN")
    score(trace_id, "eval_disrupted_count_match", row["disrupted"]["match"], "BOOLEAN")
    score(trace_id, "eval_status_match", row["status"]["match"], "BOOLEAN")
    for key in ("top_supplier", "top_high"):
        if key in row:
            score(trace_id, f"eval_{key}_match", row[key]["match"], "BOOLEAN")
    score(trace_id, "eval_seconds", row.get("seconds", 0))
    score(trace_id, "eval_tokens", row.get("prompt_tokens", 0) + row.get("completion_tokens", 0))


def flush() -> None:
    lf = _STATE["client"]
    if lf:
        _safe(lf.flush)


# ------------------------------------------------------------------ prompts
# local text is the fallback, so tests, the mock and offline runs never need Langfuse
_PROMPTS: dict[str, object] = {}       # name -> prompt client of the version last fetched (for linking generations)
STEP_PROMPT = {"monitoring": "monitoring_system", "csco": "csco_system",
               "self_critique": "self_critique_system", "credibility": "claims_system"}


def prompt_text(name: str, local: str) -> str:
    """The prompt body from Langfuse (label "production") when available, else the local text."""
    lf = client()
    if not lf or time.monotonic() < _STATE.get("prompts_down_until", 0):
        return local
    p = _safe(lf.get_prompt, name, type="text", fallback=local, max_retries=1, fetch_timeout_seconds=3)
    if p is None or getattr(p, "is_fallback", False):
        # Langfuse unreachable or the prompt does not exist yet: use the local text, and do not
        # wait on the network again for a minute
        _STATE["prompts_down_until"] = time.monotonic() + 60
        return local
    _PROMPTS[name] = p
    return getattr(p, "prompt", None) or local


def format_prompt(name: str, local: str, **kw) -> str:
    """prompt_text(...).format(**kw), falling back to the local text if the remote one will not format."""
    try:
        return prompt_text(name, local).format(**kw)
    except (KeyError, IndexError, ValueError):
        _warn(f"prompt '{name}' from Langfuse could not be formatted; using the local text")
        return local.format(**kw)


def prompt_for_step(step: str):
    p = _PROMPTS.get(STEP_PROMPT.get(step, ""))
    return p if p is not None and not getattr(p, "is_fallback", False) else None


def push_prompts(prompts: dict[str, str]) -> list[str]:
    """Create the first version of each prompt in Langfuse (label 'production') if it is missing."""
    lf = client()
    if not lf:
        raise SystemExit("Langfuse is not configured: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    done = []
    for name, text in prompts.items():
        try:
            lf.get_prompt(name, type="text", max_retries=0)
            done.append(f"{name}: already exists (left unchanged)")
        except Exception:
            lf.create_prompt(name=name, prompt=text, labels=["production"], type="text",
                             commit_message="initial version from scpoc/prompts.py")
            done.append(f"{name}: created")
    return done


# ------------------------------------------------------------------ dataset
def sync_dataset(articles: list[dict], name: str = "supply-chain-eval") -> str:
    lf = client()
    if not lf:
        raise SystemExit("Langfuse is not configured: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    try:
        lf.create_dataset(name=name, description="Golden evaluation articles with expected answers")
    except Exception:
        pass                                              # already exists
    for a in articles:
        lf.create_dataset_item(
            id=f"{name}-{a['id']}", dataset_name=name,
            input={"article": a["article"], "focal": a["focal"]},
            expected_output={"truth": a["truth"], "expected": a["expected"]},
            metadata={"id": a["id"], "kind": a["kind"]})
    lf.flush()
    return f"{len(articles)} items in dataset '{name}'"
