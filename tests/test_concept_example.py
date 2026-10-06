"""Check the optional application recipe independently of the generic core."""
import importlib.util
from pathlib import Path
import unittest

import networkx as nx
import numpy as np
import pandas as pd

import hons

path = Path(__file__).resolve().parents[1] / "examples" / "concept_network.py"
spec = importlib.util.spec_from_file_location("concept_network", path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class ConceptExampleTests(unittest.TestCase):
  def test_distinct_articles_earliest_edges_and_isolates(self):
    records = pd.DataFrame([("a1", 2000, "a"), ("a1", 2000, "a"), ("a1", 2000, "b"),
                             ("a2", 2004, "a"), ("a2", 2004, "b"),
                             ("a3", 2001, "c"), ("a4", 2002, "c")],
                            columns=["article_id", "year", "concept"])
    graph = example.to_network(records)
    self.assertEqual(nx.get_node_attributes(graph, "frequency"), {"a": 2, "b": 2, "c": 2})
    self.assertEqual(nx.get_edge_attributes(graph, "first_seen"), {("a", "b"): 2000})
    self.assertIn("c", graph)

  def test_conflicting_article_years_rejected(self):
    records = pd.DataFrame([("a", 1, "x"), ("a", 2, "y")], columns=["article_id", "year", "concept"])
    with self.assertRaises(ValueError):
      example.to_network(records)

  def test_selection_and_opaque_identifiers(self):
    data = example.example_data()
    masked = data.sample(frac=1, random_state=10).copy()
    masked["article_id"] = masked.article_id.map({v: f"record_{i}" for i, v in enumerate(sorted(data.article_id.unique()))})
    masked["concept"] = masked.concept.map({v: f"node_{99-i}" for i, v in enumerate(sorted(data.concept.unique()))})
    results = []
    for records in (data, masked):
      graph = example.to_network(records)
      selected, detail = hons.threshold(graph, node_attribute="frequency", filtration="first_seen",
                                         filtration_range=(2000, 2010), lower=[1, 3, 5, 7], upper=[7, 9, 12],
                                         constraints=(50, 50), return_details=True)
      self.assertEqual((selected.graph["hons"]["lower"], selected.graph["hons"]["upper"]), (3, 12))
      self.assertAlmostEqual(selected.graph["hons"]["score"], 6.699714, places=5)
      results.append(list(detail["scores"].values()))
    np.testing.assert_allclose(*results, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
  unittest.main()
