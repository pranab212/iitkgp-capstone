"""Make a dummy multi-tier supply network for 3 made-up grocers.

    python cli.py data               -> data/companies.csv, data/supplies.csv
"""
import random
from pathlib import Path

import pandas as pd

from .graph import DATA_DIR

COUNTRIES = [
    "United Kingdom", "Germany", "Canada", "Norway", "United States", "France",
    "Netherlands", "Spain", "Italy", "Poland", "Ireland", "Denmark", "Sweden",
    "China", "India", "Vietnam", "Thailand", "Indonesia", "Malaysia", "Bangladesh",
    "Turkey", "Brazil", "Argentina", "Chile", "Mexico", "Peru", "Kenya",
    "South Africa", "Ukraine", "Russia", "Taiwan", "South Korea", "Japan",
    "Australia", "New Zealand", "Egypt", "Morocco", "Ecuador", "Colombia", "Ghana",
]

# industry -> (products it sells, industries it buys from)
INDUSTRIES = {
    "Grocery Retail": ([], ["Food Processing", "Beverages", "Dairy", "Fresh Produce",
                            "Household Goods", "Logistics", "Packaging", "Meat Processing",
                            "Bakery", "Personal Care"]),
    "Food Processing": (["Ready meals", "Canned vegetables", "Frozen foods", "Sauces"],
                        ["Agriculture", "Packaging", "Food Ingredients", "Edible Oils"]),
    "Beverages": (["Soft drinks", "Bottled water", "Juices"],
                  ["Packaging", "Sugar", "Glass", "Agriculture"]),
    "Dairy": (["Milk", "Cheese", "Yoghurt"], ["Animal Feed", "Packaging", "Agriculture"]),
    "Fresh Produce": (["Fruit", "Vegetables", "Salad leaves"],
                      ["Agriculture", "Fertilizers", "Logistics"]),
    "Meat Processing": (["Fresh poultry", "Beef cuts", "Pork products"],
                        ["Animal Feed", "Packaging", "Logistics"]),
    "Bakery": (["Bread", "Pastries"], ["Agriculture", "Sugar", "Edible Oils", "Packaging"]),
    "Household Goods": (["Detergents", "Paper towels", "Cleaning sprays"],
                        ["Chemicals", "Paper & Pulp", "Plastics"]),
    "Personal Care": (["Shampoo", "Toothpaste", "Soap"], ["Chemicals", "Plastics", "Edible Oils"]),
    "Logistics": (["Cold-chain transport", "Warehousing", "Freight forwarding"],
                  ["Fuel & Energy", "Electronics"]),
    "Packaging": (["Plastic film", "Corrugated boxes", "PET bottles", "Aluminium cans"],
                  ["Plastics", "Paper & Pulp", "Metals"]),
    "Food Ingredients": (["Starch", "Flavourings", "Emulsifiers"], ["Agriculture", "Chemicals"]),
    "Edible Oils": (["Sunflower oil", "Palm oil", "Rapeseed oil"], ["Agriculture", "Fertilizers"]),
    "Sugar": (["Refined sugar", "Glucose syrup"], ["Agriculture", "Fertilizers"]),
    "Glass": (["Glass bottles", "Glass jars"], ["Chemicals", "Fuel & Energy"]),
    "Animal Feed": (["Compound feed", "Soy meal"], ["Agriculture", "Chemicals"]),
    "Agriculture": (["Wheat", "Corn", "Soybeans", "Sunflower seed", "Cocoa", "Coffee beans"],
                    ["Fertilizers", "Chemicals", "Fuel & Energy"]),
    "Chemicals": (["Surfactants", "Solvents", "Caustic soda", "Pesticides"],
                  ["Fuel & Energy", "Metals"]),
    "Plastics": (["PET resin", "Polyethylene", "Polypropylene"], ["Chemicals", "Fuel & Energy"]),
    "Paper & Pulp": (["Kraft paper", "Wood pulp"], ["Forestry", "Chemicals"]),
    "Metals": (["Aluminium coil", "Tinplate steel"], ["Mining", "Fuel & Energy"]),
    "Fertilizers": (["Potash", "Urea", "Ammonia"], ["Mining", "Fuel & Energy"]),
    "Fuel & Energy": (["Diesel", "Natural gas"], ["Mining"]),
    "Electronics": (["Telematics units", "Refrigeration controllers"], ["Metals", "Plastics"]),
    "Forestry": (["Pulpwood logs"], []),
    "Mining": (["Bauxite", "Phosphate rock", "Potash ore"], []),
}

SYL = ["nor", "ve", "ta", "lo", "kri", "sa", "mon", "del", "ar", "vi", "zen", "pra",
       "co", "ri", "an", "tor", "bel", "ka", "mi", "ost", "lun", "fa", "gra", "hel"]
WORDS = ["Industries", "Trading", "Group", "Supply", "Works", "Holdings"]

FOCAL = [("F1", "Northwind Grocers", "United Kingdom"),
         ("F2", "Alpenmarkt", "Germany"),
         ("F3", "Maple Fresh Markets", "Canada")]
TIER_SIZES = (35, 140, 420, 900)   # supplier links to create at Tier-1..4 per grocer
REUSE = 0.35                       # chance of reusing an existing supplier (shared nodes)


def generate(seed=7):
    rng = random.Random(seed)
    companies, links = [], []
    industry_of, pool = {}, {k: [] for k in INDUSTRIES}

    def add_company(cid, name, country, industry, focal=False):
        companies.append(dict(company_id=cid, name=name, country=country,
                              industry=industry, is_focal=focal))
        industry_of[cid] = industry

    def supplier_for(buyer):
        ind = rng.choice(INDUSTRIES[industry_of[buyer]][1])
        if pool[ind] and rng.random() < REUSE:
            sid = rng.choice(pool[ind])
        else:
            sid = f"C{len(companies):05d}"
            word = ind.split()[0].replace("&", "").strip() if rng.random() < 0.6 else rng.choice(WORDS)
            name = f"{rng.choice(SYL).capitalize()}{rng.choice(SYL)} {word} {sid[-3:]}"
            add_company(sid, name, rng.choice(COUNTRIES), ind)
            pool[ind].append(sid)
        product = rng.choice(INDUSTRIES[ind][0]) if INDUSTRIES[ind][0] else ""
        return sid, product

    for fid, fname, fcountry in FOCAL:
        add_company(fid, fname, fcountry, "Grocery Retail", focal=True)
        buyers = [fid]
        for size in TIER_SIZES:
            buyers = [b for b in buyers if INDUSTRIES[industry_of[b]][1]] or buyers
            new = []
            for i in range(size):
                buyer = buyers[i % len(buyers)]
                sid, product = supplier_for(buyer)
                if sid != buyer:
                    links.append(dict(supplier_id=sid, customer_id=buyer, product=product))
                    new.append(sid)
            buyers = list(dict.fromkeys(new))

    sup = pd.DataFrame(links).drop_duplicates(["supplier_id", "customer_id"])
    return pd.DataFrame(companies), sup


if __name__ == "__main__":
    comp, sup = generate()
    DATA_DIR.mkdir(exist_ok=True)
    comp.to_csv(DATA_DIR / "companies.csv", index=False)
    sup.to_csv(DATA_DIR / "supplies.csv", index=False)
    print(f"wrote {len(comp)} companies and {len(sup)} supply links to data/")
