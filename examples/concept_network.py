"""Prepare article/concept records for the general NetworkX thresholding API."""
from itertools import combinations, product
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import hons


def to_network(records):
  """Count articles per concept and date edges by first co-occurrence."""
  if records.groupby("article_id").year.nunique().ne(1).any():
    raise ValueError("Each article must have one publication year.")
  records = records.drop_duplicates(["article_id", "concept"])
  counts = records.groupby("concept").article_id.nunique()
  graph = nx.Graph()
  graph.add_nodes_from((concept, {"frequency": int(count)}) for concept, count in counts.items())
  for _, group in records.groupby("article_id", sort=False):
    year = float(group.year.iloc[0])
    for a, b in combinations(group.concept, 2):
      previous = graph.get_edge_data(a, b, {}).get("first_seen", year)
      graph.add_edge(a, b, first_seen=min(year, previous))
  return graph


def example_data():
  """Create an octahedral network, a square, and a frequent connecting concept."""
  articles = []
  groups = [("x0", "x1"), ("y0", "y1"), ("z0", "z1")]
  for group_a, group_b in combinations(groups, 2):
    for a, b in product(group_a, group_b):
      articles.append((2000 + len(articles) % 4, [a, b]))
  articles.append((2008, ["x0", "x1"]))
  for concept in ("x0", "x1", "y0", "y1", "z0", "z1"):
    articles.append((2010, ["hub", concept]))
  for year, pair in [(2000, ["s0", "s1"]), (2001, ["s1", "s2"]),
                     (2002, ["s2", "s3"]), (2003, ["s3", "s0"]),
                     (2008, ["s0", "s2"])]:
    articles.append((year, pair))

  # singleton articles vary document frequencies without introducing edges
  frequencies = {"x0": 8, "x1": 8, "y0": 6, "y1": 6, "z0": 5,
                 "z1": 5, "hub": 11, "s0": 4, "s1": 3, "s2": 4, "s3": 3}
  for concept, frequency in frequencies.items():
    current = sum(concept in concepts for _, concepts in articles)
    for _ in range(frequency - current):
      articles.append((2000, [concept]))
  return pd.DataFrame(
    [(f"a{i:03d}", year, concept)
     for i, (year, concepts) in enumerate(articles) for concept in concepts],
    columns=["article_id", "year", "concept"]
  )


def main():
  records = example_data()
  network = to_network(records)
  selected, details = hons.threshold(
    network, node_attribute="frequency", filtration="first_seen",
    filtration_range=(records.year.min(), records.year.max()),
    lower=[1, 3, 5, 7], upper=[7, 9, 12],
    constraints=(50, 50), return_details=True,
  )
  output = Path(__file__).resolve().parent / "output" / "concept_network"
  output.mkdir(parents=True, exist_ok=True)
  records.to_csv(output / "articles.csv", index=False)
  details["grid"].to_csv(output / "grid.csv", index=False)
  print(f"{records.article_id.nunique()} articles; {len(network)} concepts; "
        f"{len(details['grid'])} candidate networks.")
  print(selected.graph["hons"] if selected is not None else "No feasible selection.")


if __name__ == "__main__":
  main()
