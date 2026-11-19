"""Load the dummy network into an in-memory graph and query it.

Edges point supplier -> customer. A supplier's tier is its shortest distance
upstream from the grocer you're looking from (Tier-1 = direct supplier).

    python -m scpoc.graph      -> prints size and the per-tier supplier counts
"""
from pathlib import Path

import networkx as nx
import pandas as pd

MAX_TIER = 4


ROOT = Path(__file__).resolve().parent.parent   # project folder
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def load_graph(data_dir=DATA_DIR):
    comp = pd.read_csv(Path(data_dir) / "companies.csv", keep_default_na=False)
    sup = pd.read_csv(Path(data_dir) / "supplies.csv", keep_default_na=False)
    g = nx.DiGraph()
    for r in comp.to_dict("records"):
        g.add_node(r.pop("company_id"), **r)
    for r in sup.to_dict("records"):
        assert r["supplier_id"] in g and r["customer_id"] in g, f"unknown company in {r}"
        g.add_edge(r["supplier_id"], r["customer_id"], product=r["product"])
    return g


def focal_firms(g):
    return [n for n, d in g.nodes(data=True) if d["is_focal"]]


def tiers(g, focal):
    """{supplier_id: tier} for every supplier up to Tier-4 of one grocer."""
    dist = nx.single_source_shortest_path_length(g.reverse(copy=False), focal, cutoff=MAX_TIER)
    return {n: d for n, d in dist.items() if d > 0}


def find_suppliers(g, focal, country=None, industry=None):
    """Suppliers of `focal` (Tier-1..4) matching a country and/or industry,
    each with its tier and one path down to the grocer."""
    rev = g.reverse(copy=False)
    paths = nx.single_source_shortest_path(rev, focal, cutoff=MAX_TIER)
    out = []
    for sid, tier in tiers(g, focal).items():
        d = g.nodes[sid]
        if (country and d["country"] != country) or (industry and d["industry"] != industry):
            continue
        chain = list(reversed(paths[sid]))          # supplier -> ... -> grocer
        out.append(dict(company_id=sid, name=d["name"], country=d["country"],
                        industry=d["industry"], tier=tier,
                        chain=[g.nodes[n]["name"] for n in chain]))
    return sorted(out, key=lambda r: r["tier"])


if __name__ == "__main__":
    g = load_graph()
    countries = {d["country"] for _, d in g.nodes(data=True)}
    industries = {d["industry"] for _, d in g.nodes(data=True)}
    print(f"{g.number_of_nodes()} companies, {g.number_of_edges()} supply links, "
          f"{len(countries)} countries, {len(industries)} industries\n")
    print(f"{'grocer':22}" + "".join(f"Tier-{t:<4}" for t in range(1, MAX_TIER + 1)))
    for f in focal_firms(g):
        counts = pd.Series(tiers(g, f)).value_counts()
        print(f"{g.nodes[f]['name']:22}" + "".join(f"{counts.get(t, 0):<9}" for t in range(1, MAX_TIER + 1)))

    hits = find_suppliers(g, "F1", country="Ukraine")
    print(f"\nExample - Northwind Grocers suppliers in Ukraine: {len(hits)}")
    for h in hits[:3]:
        print(f"  Tier-{h['tier']} {h['name']} ({h['industry']}): {' -> '.join(h['chain'])}")
