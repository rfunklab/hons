"""Check filtration units, shared normalization and degenerate inputs."""
import unittest

import networkx as nx
import numpy as np

import hons


class FiltrationTests(unittest.TestCase):
  def setUp(self):
    self.graph = hons.example_network()
    self.options = dict(node_attribute="weight", lower=[0, 1, 2, 3, 4],
                        upper=[4, 5, 6, 7, 9], return_details=True)

  def test_explicit_range_preserves_change_of_units(self):
    expected, a = hons.threshold(self.graph, filtration_range=(0, 1), **self.options)
    changed = self.graph.copy()
    for _, _, data in changed.edges(data=True):
      data["filtration"] = 2000 + 40 * data["filtration"]
    actual, b = hons.threshold(changed, filtration_range=(2000, 2040), **self.options)
    self.assertEqual(set(expected), set(actual))
    self.assertEqual(a["cell"], b["cell"])
    for cell in a["images"]:
      np.testing.assert_allclose(a["images"][cell], b["images"][cell], rtol=1e-10, atol=1e-10)
    self.assertEqual(b["filtration_range"], (2000, 2040))
    self.assertGreater(min(nx.get_edge_attributes(actual, "filtration").values()), 2000)

  def test_inferred_range_preserves_shift_and_positive_scale(self):
    _, a = hons.threshold(self.graph, **self.options)
    shifted = self.graph.copy()
    for _, _, data in shifted.edges(data=True):
      data["filtration"] = 7 * data["filtration"] - 12
    _, b = hons.threshold(shifted, **self.options)
    self.assertEqual(a["cell"], b["cell"])
    np.testing.assert_allclose(list(a["scores"].values()), list(b["scores"].values()), atol=1e-10)

  def test_range_is_shared_across_candidate_networks(self):
    graph = nx.Graph()
    graph.add_nodes_from((n, {"value": 2}) for n in range(4))
    graph.add_node(4, value=9)
    for a, b, entry in [(0, 1, -4), (1, 2, -3), (2, 3, -2), (0, 3, -1),
                        (0, 2, 6), (0, 4, 20)]:
      graph.add_edge(a, b, entry=entry)
    selected, detail = hons.threshold(graph, node_attribute="value", filtration="entry",
                                      lower=[0, 1], upper=[5], constraints=(0, 0), return_details=True)
    self.assertNotIn(4, selected)
    self.assertEqual(detail["filtration_range"], (-4, 20))
    bars = detail["diagrams"][(0, 1)]
    np.testing.assert_allclose(bars.loc[bars.dimension == 1, ["birth", "death"]], [[3/24, 10/24]])

  def test_constant_and_empty_filtrations(self):
    graph = nx.cycle_graph(4)
    nx.set_edge_attributes(graph, -8.0, "filtration")
    nx.set_node_attributes(graph, 2, "value")
    _, details = hons.threshold(graph, node_attribute="value", lower=[0, 1], upper=[3],
                                return_details=True)
    self.assertEqual(details["filtration_range"], (-8, -8))
    bars = details["diagrams"][(0, 1)]
    np.testing.assert_array_equal(bars.loc[bars.dimension == 1, ["birth", "death"]], [[0, np.inf]])
    empty, details = hons.threshold(nx.Graph(), node_attribute="value", lower=[0, 1], upper=[3],
                                   return_details=True)
    self.assertEqual(len(empty), 0)
    self.assertEqual(details["filtration_range"], (0, 1))

  def test_invalid_ranges_and_values(self):
    for bounds in [(1, 0), (0, .1), (0, np.inf), (0,), (0, np.nan)]:
      with self.assertRaises(ValueError):
        hons.threshold(self.graph, filtration_range=bounds, **self.options)
    edge = next(iter(self.graph.edges))
    self.graph.edges[edge]["filtration"] = np.nan
    with self.assertRaisesRegex(ValueError, "finite"):
      hons.threshold(self.graph, **self.options)
    with self.assertRaises(ValueError):
      hons.persistence([0, 1], {(0, 1): 2}, start=0)

  def test_extreme_finite_values_preserve_square_lifetime(self):
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    for scale in (1.0, 1e308, np.float32(3e38)):
      edges = dict(zip(pairs, [-scale, 0, 0, 0, scale]))
      bars = hons.persistence(list(range(4)), edges)
      np.testing.assert_array_equal(
          bars.loc[bars.dimension == 1, ["birth", "death"]], [[0.5, 1.0]])

  def test_constant_values_with_explicit_nonzero_range(self):
    edges = {(0, 1): 2005, (1, 2): 2005, (2, 3): 2005, (3, 0): 2005}
    bars = hons.persistence(list(range(4)), edges, 2000, 2010)
    np.testing.assert_array_equal(
        bars.loc[bars.dimension == 1, ["birth", "death"]], [[0.5, np.inf]])


if __name__ == "__main__":
  unittest.main()
