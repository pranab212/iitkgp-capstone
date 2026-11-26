"""Phase 1 - deterministic tools (no LLM).

These are the paper's Table 3 tools. The Phase 2 agents call them; the LLM only
decides which tool to call and with what arguments. Every function returns
plain JSON-serialisable dicts/lists.

    resolve_entity      free-text company name  -> canonical company_id
    company_list        companies at a given tier of a grocer
    calculate_tier      tier of each company relative to a grocer
    supply_chain_bfs    disrupted supplier chains, grouped Tier-1..4
    tier1_risk_scores   risk score per Tier-1 supplier (35/25/20/10/10 weights)
    recommend_actions   HIGH -> replace, MEDIUM -> monitor, LOW -> standard ops
    plot_network        tiered HTML diagram of the disrupted network
"""
from __future__ import annotations

import difflib
import html
import weakref
from functools import lru_cache
from pathlib import Path

import networkx as nx

from .graph import MAX_TIER, OUT_DIR, load_graph, tiers

# risk weights and thresholds from the paper (Section 3.2.5 / 3.2.6)
WEIGHTS = {"exposure_breadth": 0.35, "dependency_ratio": 0.25,
           "downstream_criticality": 0.20, "tier1_centrality": 0.10, "exposure_depth": 0.10}
HIGH, MEDIUM = 0.60, 0.45
BREADTH_K = 3.0   # breadth saturates: weighted count k gives 0.5 (our choice; paper doesn't define)


@lru_cache(maxsize=1)
def get_graph() -> nx.DiGraph:
    return load_graph()


def _name(g, n):
    return g.nodes[n]["name"]


# ------------------------------------------------------------------ entity resolution
def resolve_entity(name: str, g: nx.DiGraph | None = None, cutoff: float = 0.6) -> dict:
    """Map a free-text company name (e.g. from a news article) to a company_id.

    Exact case-insensitive match first, then fuzzy match. Returns the best match
    plus up to 3 alternatives so the agent can ask a human when it's ambiguous.
    """
    g = g or get_graph()
    by_name = {d["name"].lower(): n for n, d in g.nodes(data=True)}
    q = name.strip().lower()
    if q in by_name:
        n = by_name[q]
        return {"query": name, "match": {"company_id": n, "name": _name(g, n), "score": 1.0},
                "alternatives": []}
    hits = difflib.get_close_matches(q, by_name, n=4, cutoff=cutoff)
    scored = [{"company_id": by_name[h], "name": _name(g, by_name[h]),
               "score": round(difflib.SequenceMatcher(None, q, h).ratio(), 3)} for h in hits]
    return {"query": name, "match": scored[0] if scored else None, "alternatives": scored[1:]}


# ------------------------------------------------------------------ tiers / lists
def calculate_tier(focal: str, g: nx.DiGraph | None = None) -> dict[str, int]:
    """{company_id: tier} for all suppliers of `focal` up to Tier-4."""
    return tiers(g or get_graph(), focal)


def company_list(focal: str, tier: int | None = None, country: str | None = None,
                 industry: str | None = None, g: nx.DiGraph | None = None) -> list[dict]:
    """Suppliers of `focal`, optionally filtered by tier, country and industry."""
    g = g or get_graph()
    out = []
    for n, t in calculate_tier(focal, g).items():
        d = g.nodes[n]
        if (tier and t != tier) or (country and d["country"] != country) \
                or (industry and d["industry"] != industry):
            continue
        out.append({"company_id": n, "name": d["name"], "country": d["country"],
                    "industry": d["industry"], "tier": t})
    return sorted(out, key=lambda r: (r["tier"], r["name"]))


# ------------------------------------------------------------------ BFS
def supply_chain_bfs(focal: str, countries: list[str] | None = None,
                     industries: list[str] | None = None, companies: list[str] | None = None,
                     g: nx.DiGraph | None = None) -> dict:
    """Find every supplier of `focal` (Tier-1..4) hit by the disruption criteria and
    return its chain down to the grocer, grouped by tier (the paper's KG output).

    A supplier is "disrupted" if it matches ANY of: country in `countries`,
    industry in `industries`, company_id in `companies`.
    Each chain runs grocer -> Tier-1 -> ... -> disrupted supplier, with the
    product supplied on each link.
    """
    g = g or get_graph()
    countries, industries, companies = set(countries or []), set(industries or []), set(companies or [])
    rev = g.reverse(copy=False)
    paths = nx.single_source_shortest_path(rev, focal, cutoff=MAX_TIER)
    tier_of = calculate_tier(focal, g)

    disrupted = [n for n in tier_of
                 if g.nodes[n]["country"] in countries or g.nodes[n]["industry"] in industries
                 or n in companies]
    result = {"focal": {"company_id": focal, "name": _name(g, focal)},
              "criteria": {"countries": sorted(countries), "industries": sorted(industries),
                           "companies": sorted(companies)},
              "disrupted_count": len(disrupted),
              **{f"tier_{t}": [] for t in range(1, MAX_TIER + 1)}}
    for n in sorted(disrupted, key=lambda x: (tier_of[x], x)):
        path = paths[n]                          # focal, tier1, ..., n
        chain = []
        for i, node in enumerate(path):
            d = g.nodes[node]
            step = {"company_id": node, "name": d["name"], "country": d["country"],
                    "industry": d["industry"], "tier": i}
            if i > 0:
                step["product_supplied"] = g.edges[node, path[i - 1]]["product"]
            chain.append(step)
        result[f"tier_{tier_of[n]}"].append(chain)
    return result


def _disrupted_ids(bfs: dict) -> dict[str, int]:
    """{disrupted company_id: tier} from a supply_chain_bfs result."""
    out = {}
    for t in range(1, MAX_TIER + 1):
        for chain in bfs[f"tier_{t}"]:
            out[chain[-1]["company_id"]] = t
    return out


# ------------------------------------------------------------------ risk scoring
_CENTRALITY: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()   # per graph object, freed with it


def _centrality(g: nx.DiGraph):
    """PageRank and degree centrality, each scaled so the graph's max = 1. Cached per graph."""
    if g not in _CENTRALITY:
        pr, deg = nx.pagerank(g), nx.degree_centrality(g)
        mpr, mdeg = max(pr.values()), max(deg.values())
        _CENTRALITY[g] = ({n: pr[n] / mpr for n in g}, {n: deg[n] / mdeg for n in g})
    return _CENTRALITY[g]


def tier1_risk_scores(bfs: dict, top_n: int = 10, g: nx.DiGraph | None = None) -> dict:
    """Score each exposed Tier-1 supplier. Five components, each 0-1:

    exposure_breadth       disrupted suppliers upstream of this Tier-1, weighted by
                           proximity (1/hops from the Tier-1), saturating x/(x+3)
    dependency_ratio       share of the Tier-1's direct suppliers that are disrupted or
                           lead to a disruption (1.0 if the Tier-1 itself is disrupted)
    downstream_criticality max over its disrupted suppliers of max(PageRank, degree), normalised
    tier1_centrality       degree centrality of the Tier-1 itself, vs. the grocer's other Tier-1s
    exposure_depth         deepest disrupted tier (relative to the grocer) / 4

    risk = 0.35*breadth + 0.25*dependency + 0.20*criticality + 0.10*centrality + 0.10*depth
    """
    g = g or get_graph()
    pr_n, deg_n = _centrality(g)
    focal = bfs["focal"]["company_id"]
    disrupted = _disrupted_ids(bfs)
    tier1 = [n for n, t in calculate_tier(focal, g).items() if t == 1]
    max_deg_t1 = max((deg_n[t] for t in tier1), default=1) or 1
    rev = g.reverse(copy=False)

    scored = []
    for t1 in tier1:
        hops = nx.single_source_shortest_path_length(rev, t1, cutoff=MAX_TIER - 1)  # t1 itself = 0
        upstream = set(hops)
        hit = [n for n in upstream if n in disrupted]
        if not hit:
            continue
        weighted = sum(1 / (hops[n] + 1) for n in hit)
        if t1 in disrupted:
            dependency = 1.0
        else:
            hit_set = set(hit)
            inputs = list(g.predecessors(t1))
            reach = [u for u in inputs if u in hit_set or hit_set & set(
                nx.single_source_shortest_path_length(rev, u, cutoff=MAX_TIER - 2))]
            dependency = len(reach) / len(inputs) if inputs else 0.0
        comp = {
            "exposure_breadth": weighted / (weighted + BREADTH_K),
            "dependency_ratio": dependency,
            "downstream_criticality": max(max(pr_n[n], deg_n[n]) for n in hit),
            "tier1_centrality": deg_n[t1] / max_deg_t1,
            "exposure_depth": max(hops[n] + 1 for n in hit) / MAX_TIER,
        }
        score = sum(WEIGHTS[k] * v for k, v in comp.items())
        d = g.nodes[t1]
        # human-readable reasoning for why this Tier-1 is scored this way
        reasons = []
        if comp["exposure_breadth"] >= 0.4:
            reasons.append(f"{len(hit)} disrupted supplier(s) upstream, weighted by proximity")
        if comp["dependency_ratio"] >= 0.5:
            if t1 in disrupted:
                reasons.append("directly disrupted (is itself in the affected region/industry)")
            else:
                reasons.append(f"{round(dependency*100):.0f}% of its direct inputs are disrupted or lead to disruption")
        if comp["downstream_criticality"] >= 0.5:
            reasons.append("high criticality — a critical node in the network (PageRank/degree)")
        if comp["tier1_centrality"] >= 0.5:
            reasons.append("high centrality among Tier-1 suppliers — many connections")
        if comp["exposure_depth"] >= 0.5:
            reasons.append(f"disruption reaches deep (tier {max(hops[n]+1 for n in hit)})")
        if not reasons:
            reasons.append("moderate exposure across all factors")
        scored.append({
            "company_id": t1, "name": d["name"], "country": d["country"], "industry": d["industry"],
            "product_supplied": g.edges[t1, focal]["product"],
            "risk_score": round(score, 3),
            "risk_level": "HIGH" if score >= HIGH else "MEDIUM" if score >= MEDIUM else "LOW",
            "disrupted_upstream": len(hit), "upstream_total": len(upstream),
            "components": {k: round(v, 3) for k, v in comp.items()},
            "reasoning": "; ".join(reasons),
        })
    scored.sort(key=lambda r: r["risk_score"], reverse=True)
    return {"focal": bfs["focal"], "weights": WEIGHTS,
            "thresholds": {"HIGH": HIGH, "MEDIUM": MEDIUM},
            "exposed_tier1_count": len(scored), "top_suppliers": scored[:top_n]}


def recommend_actions(risk: dict) -> list[dict]:
    """Deterministic action per Tier-1 (the paper's CSCO thresholds)."""
    action = {"HIGH": "replace - source alternative supplier",
              "MEDIUM": "increase monitoring",
              "LOW": "maintain standard operations"}
    return [{"company_id": r["company_id"], "name": r["name"], "risk_score": r["risk_score"],
             "risk_level": r["risk_level"], "action": action[r["risk_level"]]}
            for r in risk["top_suppliers"]]


# ------------------------------------------------------------------ alternative sourcing
def find_alternatives(supplier_id: str, bfs: dict, top_n: int = 5,
                      g: nx.DiGraph | None = None) -> dict:
    """Replacement candidates for one (HIGH-risk) Tier-1 supplier.

    A candidate must be in the same industry as the supplier, outside the disrupted
    area (not matching the disruption's country / industry / company criteria), and
    its own upstream chain (it would become a Tier-1, so Tier-2..4) must contain no
    disrupted company. Candidates are ranked by exposure (0 when clear) and then by
    the size of their upstream network (smaller = less to go wrong), then by name.
    Candidates with exposure are returned only if no fully clear one exists, flagged
    verified_clear=False.
    """
    g = g or get_graph()
    crit = bfs["criteria"]
    countries, industries, companies = set(crit["countries"]), set(crit["industries"]), set(crit["companies"])
    focal = bfs["focal"]["company_id"]

    def is_disrupted(n):
        d = g.nodes[n]
        return d["country"] in countries or d["industry"] in industries or n in companies

    sup = g.nodes[supplier_id]
    rev = g.reverse(copy=False)
    current = set(g.predecessors(focal))
    cands = []
    for n, d in g.nodes(data=True):
        if n in (supplier_id, focal) or d["is_focal"] or d["industry"] != sup["industry"]:
            continue
        if is_disrupted(n):
            continue
        hops = nx.single_source_shortest_path_length(rev, n, cutoff=MAX_TIER - 1)
        hit = [m for m in hops if m != n and is_disrupted(m)]
        exposure = sum(1 / (hops[m] + 1) for m in hit)
        cands.append({"company_id": n, "name": d["name"], "country": d["country"],
                      "industry": d["industry"], "already_supplier": n in current,
                      "verified_clear": not hit, "disrupted_upstream": len(hit),
                      "exposure_score": round(exposure, 3), "upstream_size": len(hops) - 1})
    cands.sort(key=lambda c: (c["exposure_score"], c["upstream_size"], c["name"]))
    clear = [c for c in cands if c["verified_clear"]]
    return {"supplier": {"company_id": supplier_id, "name": sup["name"], "industry": sup["industry"]},
            "candidates_considered": len(cands), "clear_count": len(clear),
            "alternatives": (clear or cands)[:top_n]}


# ------------------------------------------------------------------ visualisation
def plot_network(bfs: dict, risk: dict | None = None, out: str | None = None,
                 max_chains_per_tier: int = 25) -> str:
    """Write a self-contained HTML diagram: grocer on the left, Tier-1..4 in columns.
    Disrupted suppliers are red; Tier-1 nodes are coloured by risk level.
    Hover a node or link for details. No JS libraries needed."""
    out = out or str(OUT_DIR / "disruption_network.html")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    level = {r["company_id"]: r["risk_level"] for r in (risk or {}).get("top_suppliers", [])}
    nodes, edges, disrupted = {}, {}, set()
    for t in range(1, MAX_TIER + 1):
        for chain in bfs[f"tier_{t}"][:max_chains_per_tier]:
            disrupted.add(chain[-1]["company_id"])
            for i, s in enumerate(chain):
                nodes.setdefault(s["company_id"], s)
                if i:
                    edges[(s["company_id"], chain[i - 1]["company_id"])] = s.get("product_supplied", "")

    cols: dict[int, list[str]] = {}
    for n, s in nodes.items():
        cols.setdefault(s["tier"], []).append(n)
    colw, rowh = 260, 26
    W = 60 + max(cols) * colw + 230
    H = max(len(v) for v in cols.values()) * rowh + 80
    pos = {}
    for t, ns in cols.items():
        ns.sort(key=lambda n: nodes[n]["name"])
        top = (H - len(ns) * rowh) / 2
        for i, n in enumerate(ns):
            pos[n] = (60 + t * colw, top + i * rowh + rowh / 2)

    colour = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
    node_default = "#64748b"
    focal_colour = "#818cf8"
    edge_colour = "rgba(148,163,184,0.25)"
    text_colour = "#cbd5e1"
    heading_colour = "#e2e8f0"
    bg_colour = "#0c0e14"
    surface = "#131620"

    # Build lookup for risk reasoning
    reasoning_map = {r["company_id"]: r.get("reasoning", "") for r in (risk or {}).get("top_suppliers", [])}

    svg = []
    for (s, c), prod in edges.items():
        (x1, y1), (x2, y2) = pos[s], pos[c]
        edge_tip = (f'{html.escape(nodes[s]["name"])} supplies {html.escape(prod or "goods")} '
                    f'to {html.escape(nodes[c]["name"])}')
        if s in disrupted:
            edge_tip += ' ⚠ (supplier is disrupted — this link is at risk)'
        svg.append(f'<line class="e" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{edge_colour}" stroke-width="1">'
                   f'<title>{edge_tip}</title></line>')
    for n, (x, y) in pos.items():
        s = nodes[n]
        fill = (focal_colour if s["tier"] == 0
                else colour.get(level.get(n), node_default) if s["tier"] == 1 and n not in disrupted
                else "#ef4444" if n in disrupted
                else node_default)
        glow = f' filter="url(#glow)"' if n in disrupted or s["tier"] == 0 else ""
        tip = f'{s["name"]} | Tier-{s["tier"]} | {s["country"]} | {s["industry"]}'
        if n in level:
            tip += f' | Risk: {level[n]}'
        if n in disrupted:
            tip += f' | DISRUPTED — matches disruption criteria (country/industry/company)'
        if n in reasoning_map and reasoning_map[n]:
            tip += f' | Why: {reasoning_map[n]}'
        svg.append(f'<g><circle cx="{x}" cy="{y}" r="7" fill="{fill}" opacity="0.9"{glow}/>'
                   f'<text class="t" x="{x + 11}" y="{y + 4}" font-size="11" fill="{text_colour}">'
                   f'{html.escape(s["name"][:26])}</text>'
                   f'<title>{html.escape(tip)}</title></g>')
    heads = "".join(f'<text class="hh" x="{60 + t * colw}" y="20" font-size="13" font-weight="bold" fill="{heading_colour}">'
                    f'{"Grocer" if t == 0 else f"Tier-{t}"}</text>' for t in sorted(cols))
    defs = ('<defs><filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/>'
            '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
    crit = bfs["criteria"]
    title_text = (f'Disrupted network for {html.escape(bfs["focal"]["name"])} — '
                  f'{html.escape(", ".join(crit["countries"] + crit["industries"] + crit["companies"]))} '
                  f'({bfs["disrupted_count"]} disrupted suppliers)')
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title_text}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{--bg:{bg_colour};--surf:{surface};--txt:{text_colour};--head:{heading_colour};--edge:{edge_colour};
 --mut:#64748b;--leg:#94a3b8;--line:rgba(255,255,255,0.06)}}
[data-theme=light]{{--bg:#f4f6fb;--surf:#ffffff;--txt:#334155;--head:#0f172a;--edge:rgba(71,85,105,0.3);
 --mut:#64748b;--leg:#475569;--line:rgba(15,23,42,0.1)}}
body{{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--txt);padding:16px}}
h3{{font-size:14px;font-weight:700;color:var(--head);margin-bottom:6px}}
p{{font-size:12px;color:var(--mut);margin-bottom:12px;line-height:1.5}}
svg{{border:1px solid var(--line);border-radius:10px;background:var(--surf)}}
.t{{fill:var(--txt)}}.hh{{fill:var(--head)}}.e{{stroke:var(--edge)}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:11px;color:var(--leg)}}
.legend span{{display:inline-flex;align-items:center;gap:5px}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:50%}}
</style></head>
<body>
<h3>{title_text}</h3>
<p>Hover nodes or edges for details. Up to {max_chains_per_tier} chains per tier.</p>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{defs}{heads}{''.join(svg)}</svg>
<div class="legend">
 <span><i style="background:{focal_colour}"></i> Grocer (focal)</span>
 <span><i style="background:#ef4444"></i> Disrupted</span>
 <span><i style="background:#ef4444"></i> HIGH risk</span>
 <span><i style="background:#f59e0b"></i> MEDIUM risk</span>
 <span><i style="background:#10b981"></i> LOW risk</span>
 <span><i style="background:#64748b"></i> Not exposed</span>
</div>
<script>
function ap(t){{document.documentElement.setAttribute('data-theme',t==='light'?'light':'dark')}}
try{{ap(parent.document.documentElement.getAttribute('data-theme'))}}catch(e){{}}
addEventListener('message',function(e){{if(e.data&&e.data.theme)ap(e.data.theme)}});
</script>
</body></html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    return out
