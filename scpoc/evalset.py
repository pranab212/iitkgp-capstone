"""The evaluation set: 20 made-up news articles with their expected answers.

    python cli.py eval-build        -> data/eval_articles.json

Each article carries a hand-written "truth" (what a careful reader would extract: the
countries, industries and company names really affected). The expected network results
(disrupted count per tier, top-scoring Tier-1 supplier, top HIGH supplier) are then computed
from that truth with the deterministic tools, so they are exact and repeatable.

15 articles hit the network: 4 single-company events (Tier-1 to Tier-4), 5 country events,
3 industry events, 3 country-and-industry events. 5 are false positives that must end with
"no exposure". Some articles contain distractor places (buyers, markets) that are NOT affected.
"""
from __future__ import annotations

import json

from . import tools
from .graph import DATA_DIR

EVAL_FILE = DATA_DIR / "eval_articles.json"
N, A, M = "Northwind Grocers", "Alpenmarkt", "Maple Fresh Markets"

# id, kind, focal, article, truth (countries, industries, companies)
SPECS = [
    # ---- single-company events, one per tier of the grocer's network
    dict(id="co_t1", kind="company", tier=1, focal=N,
         article="Taka Beverages 003 said on Monday that a ransomware attack has forced it to shut its "
                 "bottling lines. The soft-drinks and juice maker expects to be offline for at least "
                 "three weeks while systems are rebuilt, and has warned customers of delivery gaps.",
         truth=dict(countries=[], industries=[], companies=["Taka Beverages 003"])),
    dict(id="co_t2", kind="company", tier=2, focal=N,
         article="A fire at the Nornor Electronics 031 factory destroyed its refrigeration-controller "
                 "line late on Tuesday. No one was hurt. The company told customers, including buyers in "
                 "Germany, to expect shipment delays of two months.",
         truth=dict(countries=[], industries=[], companies=["Nornor Electronics 031"])),
    dict(id="co_t3", kind="company", tier=3, focal=M,
         article="Production at Kave Mining 433 has stopped after a section of its main shaft collapsed. "
                 "Engineers will inspect the site before work can restart, and the firm has declared "
                 "force majeure on all open contracts.",
         truth=dict(countries=[], industries=[], companies=["Kave Mining 433"])),
    dict(id="co_t4", kind="company", tier=4, focal=M,
         article="Workers at Kalun Group 551 walked out on Thursday in a dispute over pay. The mining "
                 "company's output of bauxite has halted, and the union says the strike will continue "
                 "until a new offer is made.",
         truth=dict(countries=[], industries=[], companies=["Kalun Group 551"])),
    # ---- country events
    dict(id="ctry_turkey", kind="country", focal=N,
         article="A magnitude 7.4 earthquake struck central Turkey overnight, damaging factories, roads "
                 "and ports. Authorities have suspended industrial output across the affected provinces "
                 "and say the recovery will take months.",
         truth=dict(countries=["Turkey"], industries=[], companies=[])),
    dict(id="ctry_egypt", kind="country", focal=A,
         article="Unrest in Egypt escalated this week, with protesters blocking roads to the main ports and "
                 "the government imposing a nationwide curfew. Shipping lines have suspended calls at "
                 "Egyptian terminals. Importers in Italy and Spain are looking for other routes.",
         truth=dict(countries=["Egypt"], industries=[], companies=[])),
    dict(id="ctry_chile", kind="country", focal=M,
         article="A strong earthquake and tsunami warning hit the coast of Chile on Friday. Ports and "
                 "factories along the coast have been evacuated, and the government declared a state of "
                 "emergency that will last at least a month.",
         truth=dict(countries=["Chile"], industries=[], companies=[])),
    dict(id="ctry_india", kind="country", focal=M,
         article="Severe flooding across India has cut road and rail links and forced factories to close in "
                 "several states. Officials said it is the worst monsoon flooding in a decade and that "
                 "exports from the country will be disrupted for weeks.",
         truth=dict(countries=["India"], industries=[], companies=[])),
    dict(id="ctry_brazil", kind="country", focal=A,
         article="Truckers in Brazil began a nationwide strike on Wednesday, blocking highways to protest "
                 "fuel prices. Deliveries between farms, factories and ports have largely stopped, and "
                 "analysts expect shortages in export markets such as the United Kingdom.",
         truth=dict(countries=["Brazil"], industries=[], companies=[])),
    # ---- industry events (no country)
    dict(id="ind_fert", kind="industry", focal=M,
         article="Fertilizer producers worldwide have cut output after a sharp rise in natural gas costs. "
                 "Supplies of potash, urea and ammonia are tight, and the industry body warns that "
                 "shortages will hit farmers for the rest of the season.",
         truth=dict(countries=[], industries=["Fertilizers"], companies=[])),
    dict(id="ind_glass", kind="industry", focal=N,
         article="Glass manufacturers say they have been forced to shut furnaces because of a shortage of "
                 "soda ash. Makers of bottles and jars across the sector expect output to fall by a "
                 "third, and food and drink companies are being warned to expect supply problems.",
         truth=dict(countries=[], industries=["Glass"], companies=[])),
    dict(id="ind_metals", kind="industry", focal=A,
         article="A new alliance of governments has banned exports of aluminium coil and tinplate steel to "
                 "protect its own supplies, hitting the global metals industry. Can and packaging "
                 "makers say they will struggle to find replacement metal for months.",
         truth=dict(countries=[], industries=["Metals"], companies=[])),
    # ---- country + industry events
    dict(id="mix_pack_india", kind="country_industry", focal=N,
         article="Packaging manufacturers across India have halted production after a gas supply failure "
                 "at industrial parks. Makers of plastic film, cartons and PET bottles in the country say "
                 "customers in Europe should expect delays.",
         truth=dict(countries=["India"], industries=["Packaging"], companies=[])),
    dict(id="mix_log_vietnam", kind="country_industry", focal=M,
         article="Dockworkers in Vietnam began an indefinite strike today, shutting container terminals. "
                 "Logistics operators say cold-chain transport, warehousing and freight forwarding out of "
                 "the country are suspended.",
         truth=dict(countries=["Vietnam"], industries=["Logistics"], companies=[])),
    dict(id="mix_oil_blacksea", kind="country_industry", focal=A,
         article="Sunflower, palm and rapeseed oil exports from Russia and Ukraine have stopped as ports on "
                 "the Black Sea were closed because of the fighting. Edible oil buyers in Germany and "
                 "France are warning of shortages and price spikes.",
         truth=dict(countries=["Russia", "Ukraine"], industries=["Edible Oils"], companies=[])),
    # ---- false positives: must end with "no exposure"
    dict(id="fp_earnings", kind="false_positive", focal=N,
         article="Northwind Grocers reported record quarterly profit on Thursday, helped by strong demand "
                 "for ready meals. The retailer said it will open twelve new stores next year and raise "
                 "its dividend.",
         truth=dict(countries=[], industries=[], companies=[])),
    dict(id="fp_iceland", kind="false_positive", focal=A,
         article="A heavy storm closed roads in Iceland on Saturday and forced the cancellation of a music "
                 "festival. Organisers said tickets will be refunded and the event will return next summer.",
         truth=dict(countries=[], industries=[], companies=[])),
    dict(id="fp_unknown_co", kind="false_positive", focal=M,
         article="A fire damaged the Helios Photonics plant near Luxembourg City on Sunday. The company, which "
                 "makes optical sensors for medical scanners, said the damage is limited and that "
                 "production at its other sites is unaffected.",
         truth=dict(countries=[], industries=[], companies=[])),
    dict(id="fp_chips", kind="false_positive", focal=N,
         article="The global semiconductor shortage is lifting prices for graphics cards and games consoles. "
                 "Chip makers say lead times for advanced processors have reached a year, but carmakers "
                 "and phone makers are the hardest hit.",
         truth=dict(countries=[], industries=[], companies=[])),
    dict(id="fp_retrospective", kind="false_positive", focal=A,
         article="A year on from the shock to Ukraine's grain exports, analysts say food supply chains have "
                 "fully recovered. Prices have returned to normal and shipments from the region are "
                 "flowing without interruption, according to a new industry review.",
         truth=dict(countries=[], industries=[], companies=[])),
]


def _expected(spec: dict, g) -> dict:
    """Expected pipeline results, computed from the truth with the deterministic tools."""
    t = spec["truth"]
    focal = tools.resolve_entity(spec["focal"], g=g)["match"]["company_id"]
    ids = []
    for name in t["companies"]:
        hit = tools.resolve_entity(name, g=g)["match"]
        assert hit and hit["score"] == 1.0, f"{spec['id']}: company {name!r} not in the network"
        ids.append(hit["company_id"])
    for c in t["countries"]:
        assert any(d["country"] == c for _, d in g.nodes(data=True)), f"{spec['id']}: unknown country {c}"
    for i in t["industries"]:
        assert any(d["industry"] == i for _, d in g.nodes(data=True)), f"{spec['id']}: unknown industry {i}"
    bfs = tools.supply_chain_bfs(focal, countries=t["countries"], industries=t["industries"],
                                 companies=ids, g=g)
    risk = tools.tier1_risk_scores(bfs, g=g)
    top = risk["top_suppliers"][0] if risk["top_suppliers"] else None
    high = next((s for s in risk["top_suppliers"] if s["risk_level"] == "HIGH"), None)
    pick = lambda s: {"company_id": s["company_id"], "name": s["name"], "risk_score": s["risk_score"],
                      "risk_level": s["risk_level"]} if s else None
    return {"focal_id": focal, "company_ids": ids, "disrupted_count": bfs["disrupted_count"],
            "per_tier": {str(k): len(bfs[f"tier_{k}"]) for k in range(1, 5)},
            "exposed_tier1": risk["exposed_tier1_count"], "top_supplier": pick(top), "top_high_supplier": pick(high),
            "final_status": "no_exposure" if bfs["disrupted_count"] == 0 else "completed"}


def build(path=EVAL_FILE) -> list[dict]:
    g = tools.get_graph()
    out = []
    for spec in SPECS:
        e = _expected(spec, g)
        if spec["kind"] == "false_positive":
            assert e["disrupted_count"] == 0, spec["id"]
        else:
            assert e["disrupted_count"] > 0, f"{spec['id']} does not touch the network"
            if spec["kind"] == "company":                        # the company really sits at the intended tier
                assert tools.calculate_tier(e["focal_id"], g)[e["company_ids"][0]] == spec["tier"], spec["id"]
        out.append({**spec, "expected": e})
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def load(path=EVAL_FILE) -> list[dict]:
    if not path.exists():
        return build(path)
    return json.loads(path.read_text(encoding="utf-8"))
