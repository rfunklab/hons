"""Check small known topologies, image conventions and selection boundaries."""
import itertools
import math
import unittest

import numpy as np
import pandas as pd

import hons


class MethodTests(unittest.TestCase):
  def test_square_has_one_essential_cycle(self):
    edges = {pair: 2005 for pair in [(0, 1), (1, 2), (2, 3), (0, 3)]}
    diagram = hons.persistence(list(range(4)), edges, 2000, 2010)
    pairs = diagram.loc[diagram.dimension == 1, ["birth", "death"]].to_numpy()
    np.testing.assert_array_equal(pairs, [[0.5, np.inf]])

  def test_octahedron_has_one_essential_cavity(self):
    opposite = [(0, 1), (2, 3), (4, 5)]
    edges = {e: 2005 for e in itertools.combinations(range(6), 2) if e not in opposite}
    diagram = hons.persistence(list(range(6)), edges, 2000, 2010)
    pairs = diagram.loc[diagram.dimension == 2, ["birth", "death"]].to_numpy()
    np.testing.assert_array_equal(pairs, [[0.5, np.inf]])

  def test_isolated_vertex_is_retained(self):
    diagram = hons.persistence(["alone"], {}, 2000, 2010)
    self.assertEqual(diagram.dimension.to_list(), [0])
    self.assertEqual(diagram.birth.to_list(), [0.0])
    self.assertTrue(np.isinf(diagram.death.iloc[0]))

  def test_empty_image_and_single_lifetime_weight(self):
    np.testing.assert_array_equal(hons.image(np.empty((0, 2))), np.zeros(400))
    image = hons.image(np.array([[0.0, 0.5]]))
    self.assertEqual(image.shape, (400,))
    self.assertAlmostEqual(image[0], 0.5 * math.exp(-12.5) / (2 * math.pi * 0.01), places=14)
    diagram = pd.DataFrame([(1, 0.25, np.inf)], columns=["dimension", "birth", "death"])
    np.testing.assert_array_equal(hons.lifetimes(diagram, 1), [[0.25, 1.00001]])
    self.assertTrue(np.isinf(diagram.death.iloc[0]))

  def test_unequal_spacing_and_boundary_neighbors(self):
    lowers, uppers = np.array([0.0, 2.0, 5.0]), np.array([1.0, 4.0, 10.0])
    vectors = {(i, j): np.array([3 * lower + 4 * upper])
               for i, upper in enumerate(uppers) for j, lower in enumerate(lowers)}
    scores = hons.objective(vectors, lowers, uppers)
    self.assertEqual(len(scores), 6)
    np.testing.assert_allclose(list(scores.values()), 5.0)
    self.assertTrue(all(j > 0 for i, j in scores))

  def test_infeasible_percentile_constraints(self):
    keys = [(0, j) for j in range(1, 5)]
    scores = dict(zip(keys, [1.0, 2.0, 3.0, 4.0]))
    h1 = dict(zip(keys, [0, 10, 20, 30]))
    h2 = dict(zip(keys, [30, 20, 10, 0]))
    self.assertEqual(hons.select(scores, h1, h2, 75, 75), (None, []))

  def test_near_tie_reporting_preserves_actual_minimum(self):
    scores = {(0, 3): 2.0, (0, 2): 1.0, (0, 1): 1.0 + 5e-13}
    counts = {k: 1 for k in scores}
    chosen, ties = hons.select(scores, counts, counts, 50, 50)
    self.assertEqual(chosen, (0, 2))
    self.assertEqual(ties, [(0, 1), (0, 2)])


if __name__ == "__main__":
  unittest.main()
