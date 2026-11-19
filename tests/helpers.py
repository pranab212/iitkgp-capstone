"""Shared test fixtures."""
import networkx as nx


def tiny():
    g = nx.DiGraph()
    for n, c, ind in [("G", "UK", "Grocery Retail"), ("A", "UK", "Dairy"), ("B", "UK", "Bakery"),
                      ("X", "UK", "Animal Feed"), ("Y", "Ukraine", "Agriculture"),
                      ("Z", "UK", "Agriculture")]:
        g.add_node(n, name=f"{n} Co", country=c, industry=ind, is_focal=n == "G")
    for s, t, p in [("A", "G", "Milk"), ("B", "G", "Bread"), ("X", "A", "Feed"),
                    ("Y", "X", "Soy"), ("Z", "B", "Wheat")]:
        g.add_edge(s, t, product=p)
    return g
