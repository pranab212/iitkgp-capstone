"""Tests on a tiny hand-built network where every number can be checked by hand.

    G <- A <- X <- Y          (G = grocer, A and B are Tier-1)
    G <- B <- Z
    Y is in "Ukraine", everything else in "UK".

    pytest tests/test_tools.py
"""
import json
import os
import tempfile

import networkx as nx

from scpoc import tools
from helpers import tiny




def test_resolve_entity():
    g = tiny()
    assert tools.resolve_entity("a co", g=g)["match"]["company_id"] == "A"
    assert tools.resolve_entity("A Coo", g=g)["match"]["company_id"] == "A"   # fuzzy
    assert tools.resolve_entity("Nothing Like It Ltd", g=g)["match"] is None


def test_tiers_and_list():
    g = tiny()
    assert tools.calculate_tier("G", g) == {"A": 1, "B": 1, "X": 2, "Z": 2, "Y": 3}
    assert [r["company_id"] for r in tools.company_list("G", tier=2, g=g)] == ["X", "Z"]


def test_bfs_chain():
    g = tiny()
    bfs = tools.supply_chain_bfs("G", countries=["Ukraine"], g=g)
    assert bfs["disrupted_count"] == 1 and len(bfs["tier_3"]) == 1
    chain = bfs["tier_3"][0]
    assert [s["company_id"] for s in chain] == ["G", "A", "X", "Y"]
    assert [s.get("product_supplied") for s in chain] == [None, "Milk", "Feed", "Soy"]
    json.dumps(bfs)                                                    # JSON-serialisable


def test_risk_components():
    g = tiny()
    bfs = tools.supply_chain_bfs("G", countries=["Ukraine"], g=g)
    risk = tools.tier1_risk_scores(bfs, g=g)
    assert risk["exposed_tier1_count"] == 1                           # only A leads to Y
    r = risk["top_suppliers"][0]; c = r["components"]
    assert r["company_id"] == "A"
    w = 1 / 3                                                         # Y is 2 hops above A
    assert abs(c["exposure_breadth"] - round(w / (w + 3), 3)) < 1e-9
    assert c["dependency_ratio"] == 1.0                               # A's only input X leads to Y
    assert c["exposure_depth"] == 0.75                                # Tier-3 / 4
    expected = sum(tools.WEIGHTS[k] * v for k, v in c.items())
    assert abs(r["risk_score"] - expected) < 0.002
    level = "HIGH" if r["risk_score"] >= .6 else "MEDIUM" if r["risk_score"] >= .45 else "LOW"
    assert r["risk_level"] == level


def test_tier1_itself_disrupted():
    g = tiny()
    bfs = tools.supply_chain_bfs("G", companies=["B"], g=g)
    r = tools.tier1_risk_scores(bfs, g=g)["top_suppliers"][0]
    assert r["company_id"] == "B" and r["components"]["dependency_ratio"] == 1.0
    assert r["components"]["exposure_depth"] == 0.25


def test_no_disruption():
    g = tiny()
    bfs = tools.supply_chain_bfs("G", countries=["Atlantis"], g=g)
    assert bfs["disrupted_count"] == 0
    assert tools.tier1_risk_scores(bfs, g=g)["top_suppliers"] == []


def test_actions_and_plot():
    g = tiny()
    bfs = tools.supply_chain_bfs("G", countries=["Ukraine"], g=g)
    risk = tools.tier1_risk_scores(bfs, g=g)
    acts = tools.recommend_actions(risk)
    assert acts[0]["action"].startswith({"HIGH": "replace", "MEDIUM": "increase",
                                         "LOW": "maintain"}[acts[0]["risk_level"]])
    out = os.path.join(tempfile.mkdtemp(), "net.html")
    tools.plot_network(bfs, risk, out=out)
    page = open(out).read()
    assert "Y Co" in page and "Soy" in page

