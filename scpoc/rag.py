"""Retrieval for the Monitoring step. Two independent parts, each behind its own flag:

  1. Example retrieval (--rag-examples)
     A small corpus of labeled news articles (data/rag/exemplars.json) is indexed with BM25.
     For each new article the most similar solved examples are added to the prompt as
     dynamic few-shot examples. The corpus is written separately from the evaluation set
     (scpoc/evalset.py) and must never contain its articles; tests/test_rag.py checks this.

  2. Entity grounding (--ground-entities)
     Before the LLM reads the article, companies from the knowledge graph that the article
     names are looked up (exact or near-exact name match) and their records (industry,
     country, focal grocer or supplier) are added to the prompt. Only the companies the
     article names are shown: never the grocer's supplier network, so the model is not
     nudged toward "finding" exposure.

Pure standard library, deterministic, no vector database.

    from scpoc.rag import ExemplarStore, EntityIndex
    store = ExemplarStore.load();   store.search(article, k=3)
    index = EntityIndex(g);         index.find(article)
"""
from __future__ import annotations

import difflib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .graph import DATA_DIR
from .schemas import MonitoringOutput

EXEMPLAR_FILE = DATA_DIR / "rag" / "exemplars.json"

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an and are as at be been but by for from has have in into is it its of on or
that the their this to was were will with which who after over says said than they there
these those while also out up all so not no our we can could would should may""".split())


def _stem(t: str) -> str:
    """Very light plural stripping so 'strikes'/'strike' and 'ports'/'port' match."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1]


# ------------------------------------------------------------------ BM25
class BM25:
    """Okapi BM25 over a list of documents (strings)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lens) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for d, dl in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


# ------------------------------------------------------------------ 1. example retrieval
class ExemplarStore:
    """Labeled example articles: {"id", "kind", "article", "answer": <MonitoringOutput fields>}."""

    def __init__(self, exemplars: list[dict]):
        for e in exemplars:                                  # fail early on a malformed corpus
            MonitoringOutput(**e["answer"])
        self.items = exemplars
        self.bm25 = BM25([e["article"] for e in exemplars])

    @classmethod
    def load(cls, path: Path = EXEMPLAR_FILE) -> "ExemplarStore":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def search(self, article: str, k: int = 3, max_per_kind: int = 2) -> list[dict]:
        """Top-k examples by BM25, at most `max_per_kind` of one kind so the model sees a contrast
        (e.g. a company incident next to a sector-wide event)."""
        ranked = sorted(zip(self.bm25.scores(article), range(len(self.items))), key=lambda x: (-x[0], x[1]))
        out, per_kind = [], Counter()
        for score, i in ranked:
            e = self.items[i]
            if per_kind[e["kind"]] >= max_per_kind:
                continue
            per_kind[e["kind"]] += 1
            out.append({**e, "score": round(score, 3)})
            if len(out) == k:
                break
        return out


def format_examples(examples: list[dict]) -> str:
    if not examples:
        return ""
    parts = ["Solved examples similar to this article. Apply the same rules:"]
    for e in examples:
        parts.append(f'Article: "{e["article"]}"\n{json.dumps(e["answer"], ensure_ascii=False)}')
    return "\n\n".join(parts)


# ------------------------------------------------------------------ 2. entity grounding
class EntityIndex:
    """Find graph companies named in an article.

    Candidates come from an inverted index on each name's distinctive tokens (tokens shared by
    at most `max_df` names, e.g. 'kave' or '433' but not 'mining'). Each candidate's name is then
    compared with every same-length window of article words; a ratio >= `min_score` counts.
    """

    def __init__(self, g, max_df: int = 25, min_score: float = 0.85):
        self.g, self.min_score = g, min_score
        self.words = {n: _WORD.findall(d["name"].lower()) for n, d in g.nodes(data=True)}
        df = Counter(t for ws in self.words.values() for t in set(ws))
        self.index: dict[str, set[str]] = defaultdict(set)
        for n, ws in self.words.items():
            for t in ws:
                if df[t] <= max_df:
                    self.index[t].add(n)

    def find(self, article: str) -> list[dict]:
        words = _WORD.findall(article.lower())
        # match on exact distinctive tokens and on near-misses (one typo in an invented name)
        cands = set()
        for w in set(words):
            cands |= self.index.get(w, set())
            if len(w) >= 5 and not w.isdigit():
                for t in difflib.get_close_matches(w, self.index.keys(), n=3, cutoff=0.85):
                    cands |= self.index[t]
        hits = []
        for n in cands:
            name_ws = self.words[n]
            name, size = " ".join(name_ws), len(name_ws)
            best, text = 0.0, ""
            for i in range(max(len(words) - size + 1, 0)):
                window = " ".join(words[i:i + size])
                r = 1.0 if window == name else difflib.SequenceMatcher(None, name, window).ratio()
                if r > best:
                    best, text = r, window
            if best >= self.min_score:
                d = self.g.nodes[n]
                hits.append({"company_id": n, "name": d["name"], "country": d["country"],
                             "industry": d["industry"], "is_focal": bool(d["is_focal"]),
                             "score": round(best, 3), "matched_text": text})
        # one record per matched span: keep the best-scoring company for overlapping text
        hits.sort(key=lambda h: (-h["score"], h["name"]))
        seen, out = set(), []
        for h in hits:
            if h["matched_text"] not in seen:
                seen.add(h["matched_text"])
                out.append(h)
        return out


def format_entities(entities: list[dict]) -> str:
    if not entities:
        return "Companies from the supplier database named in this article: none."
    lines = ["Companies from the supplier database named in this article (use the database spelling):"]
    for e in entities:
        if e["is_focal"]:
            lines.append(f"- {e['name']}: the grocery retailer we monitor (the customer, not a supplier). "
                         "Do not list it under companies.")
        else:
            lines.append(f"- {e['name']}: industry {e['industry']}, country {e['country']}. If it is affected, "
                         f"list it under companies; that already covers it. Do not also list its industry "
                         f"({e['industry']}) or its country ({e['country']}) unless the article says that whole "
                         "sector or that whole country is disrupted.")
    return "\n".join(lines)
