"""Check graph input/output against the low-level calculation."""
import unittest
from copy import deepcopy

import networkx as nx
import numpy as np

import hons


class NetworkAPI(unittest.TestCase):
  def setUp(self):
    self.graph = hons.example_network()
    self.options = dict(node_attribute="weight", lower=[0, 1, 2, 3, 4],
                        upper=[4, 5, 6, 7, 9], filtration_range=(0, 1),
                        constraints=(50, 25))

  def test_node_selection_and_input_preservation(self):
    before = deepcopy(self.graph)
    result, detail = hons.threshold(self.graph, return_details=True, **self.options)
    self.assertTrue(nx.utils.graphs_equal(self.graph, before))
    expected = self.graph.subgraph(n for n, d in self.graph.nodes(data=True)
                                   if 4 <= d["weight"] <= 7)
    self.assertEqual(set(result), set(expected))
    self.assertEqual(set(result.edges), set(expected.edges))
    self.assertEqual((len(result), result.number_of_edges()), (18, 41))
    self.assertEqual((result.graph["hons"]["H1"], result.graph["hons"]["H2"]), (5, 3))
    self.assertEqual(len(detail["grid"]), 25)

  def test_each_cell_matches_direct_calculation(self):
    _, detail = hons.threshold(self.graph, return_details=True, **self.options)
    for row in detail["grid"].itertuples():
      graph = self.graph.subgraph(n for n, d in self.graph.nodes(data=True)
                                  if row.lower <= d["weight"] <= row.upper)
      diagram = hons.persistence(list(graph), {(a, b): d["filtration"]
                                 for a, b, d in graph.edges(data=True)}, 0, 1)
      np.testing.assert_array_equal(hons.rho(diagram),
                                    detail["images"][(row.upper_index, row.lower_index)])

  def test_edge_selection_preserves_isolated_nodes_and_attributes(self):
    graph = nx.cycle_graph(4)
    graph.add_node("isolated", label="keep")
    nx.set_edge_attributes(graph, .2, "filtration")
    nx.set_edge_attributes(graph, 2, "weight")
    graph[0][1]["weight"] = 9
    result = hons.threshold(graph, edge_attribute="weight", lower=[0, 1],
                            upper=[3], constraints=(0, 0))
    self.assertEqual(set(result), set(graph))
    self.assertEqual(result.number_of_edges(), 3)
    self.assertEqual(result.nodes["isolated"]["label"], "keep")
    self.assertEqual(graph.number_of_edges(), 4)

  def test_node_labels_do_not_affect_scores(self):
    changed = nx.relabel_nodes(self.graph, {n: f"opaque-{100-n}" for n in self.graph})
    first, a = hons.threshold(self.graph, return_details=True, **self.options)
    second, b = hons.threshold(changed, return_details=True, **self.options)
    self.assertEqual(a["cell"], b["cell"])
    np.testing.assert_allclose(list(a["scores"].values()), list(b["scores"].values()), atol=1e-10)

  def test_missing_filtration_rejected(self):
    a, b = next(iter(self.graph.edges))
    del self.graph[a][b]["filtration"]
    with self.assertRaisesRegex(ValueError, "filtration"):
      hons.threshold(self.graph, **self.options)

  def test_invalid_graph_types_and_attributes_rejected(self):
    for graph in [nx.DiGraph(self.graph), nx.MultiGraph(self.graph)]:
      with self.assertRaises(ValueError):
        hons.threshold(graph, **self.options)
    self.graph.nodes[0]["weight"] = np.nan
    with self.assertRaisesRegex(ValueError, "weight"):
      hons.threshold(self.graph, **self.options)

  def test_invalid_percentiles_rejected_before_calculation(self):
    for constraints in [(-1, 25), (50, 101), (np.nan, 25), (50,)]:
      with self.assertRaises(ValueError):
        hons.threshold(self.graph, **dict(self.options, constraints=constraints))

  def test_demo_generation_is_deterministic(self):
    self.assertTrue(nx.utils.graphs_equal(hons.example_network(), hons.example_network()))


if __name__ == "__main__":
  unittest.main()
