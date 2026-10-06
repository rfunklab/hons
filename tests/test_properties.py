"""Check analytical examples, independent homology ranks, and invariances."""
import itertools
import math
import unittest

import numpy as np
import pandas as pd

import hons


def rank_mod_11(matrix):
  """Row-reduce a small integer boundary matrix over the field with 11 elements."""
  matrix = np.asarray(matrix, dtype=int).copy() % 11
  rank = 0
  for column in range(matrix.shape[1]):
    pivot = next((i for i in range(rank, matrix.shape[0]) if matrix[i, column]), None)
    if pivot is None:
      continue
    matrix[[rank, pivot]] = matrix[[pivot, rank]]
    matrix[rank] = matrix[rank] * pow(int(matrix[rank, column]), -1, 11) % 11
    for i in range(matrix.shape[0]):
      if i != rank:
        matrix[i] = (matrix[i] - matrix[i, column] * matrix[rank]) % 11
    rank += 1
    if rank == matrix.shape[0]:
      break
  return rank


def clique_betti(nodes, edges):
  """Compute Betti numbers from boundary-matrix ranks, independently of GUDHI."""
  simplices = [[(v,) for v in nodes]]
  for size in (2, 3, 4):
    simplices.append([s for s in itertools.combinations(nodes, size)
                      if all(e in edges for e in itertools.combinations(s, 2))])
  ranks = [0]
  for dim in (1, 2, 3):
    rows = {s: i for i, s in enumerate(simplices[dim - 1])}
    boundary = np.zeros((len(rows), len(simplices[dim])), dtype=int)
    for j, simplex in enumerate(simplices[dim]):
      for k in range(len(simplex)):
        boundary[rows[simplex[:k] + simplex[k + 1:]], j] = (-1) ** k
    ranks.append(rank_mod_11(boundary))
  return [len(simplices[d]) - ranks[d] - ranks[d + 1] for d in (0, 1, 2)]


class TopologyProperties(unittest.TestCase):
  def test_empty_network(self):
    self.assertTrue(hons.persistence([], {}, 2000, 2000).empty)

  def test_complete_clique_has_no_holes(self):
    nodes = list(range(6))
    diagram = hons.persistence(nodes, dict.fromkeys(itertools.combinations(nodes, 2), 2005), 2000, 2010)
    self.assertEqual(int((diagram.dimension > 0).sum()), 0)

  def test_finite_square_cycle(self):
    edges = dict.fromkeys([(0, 1), (1, 2), (2, 3), (0, 3)], 2005)
    edges[(0, 2)] = 2008
    diagram = hons.persistence(list(range(4)), edges, 2000, 2010)
    pairs = diagram.loc[diagram.dimension == 1, ["birth", "death"]]
    np.testing.assert_allclose(pairs, [[0.5, 0.8]])

  def test_finite_octahedral_cavity(self):
    opposite = [(0, 1), (2, 3), (4, 5)]
    edges = {e: 2005 for e in itertools.combinations(range(6), 2) if e not in opposite}
    edges[(0, 1)] = 2008
    diagram = hons.persistence(list(range(6)), edges, 2000, 2010)
    pairs = diagram.loc[diagram.dimension == 2, ["birth", "death"]]
    np.testing.assert_allclose(pairs, [[0.5, 0.8]])

  def test_single_year_filtration(self):
    edges = dict.fromkeys([(0, 1), (1, 2), (2, 3), (0, 3)], 2000)
    diagram = hons.persistence(list(range(4)), edges, 2000, 2000)
    np.testing.assert_array_equal(diagram.loc[diagram.dimension == 1, "birth"], [0.0])

  def test_random_graphs_against_independent_boundary_ranks(self):
    rng = np.random.default_rng(1701)
    for n in range(1, 8):
      for probability in (0.0, 0.25, 0.5, 0.8, 1.0):
        with self.subTest(n=n, probability=probability):
          nodes = list(range(n))
          edges = {e: 2005 for e in itertools.combinations(nodes, 2)
                   if rng.random() < probability}
          expected = clique_betti(nodes, edges)
          diagram = hons.persistence(nodes, edges, 2000, 2010)
          actual = [int(((diagram.dimension == d) & np.isinf(diagram.death)).sum())
                    for d in range(3)]
          self.assertEqual(actual, expected)

  def test_invalid_filtration_years(self):
    for lower, upper, year in [(2010, 2000, 2005), (2000, 2010, 2011)]:
      with self.subTest(lower=lower, upper=upper, year=year), self.assertRaises(ValueError):
        hons.persistence([0, 1], {(0, 1): year}, lower, upper)

  def test_duplicate_nodes_rejected(self):
    with self.assertRaises(ValueError):
      hons.persistence([0, 0], {}, 2000, 2010)


class ImageProperties(unittest.TestCase):
  def test_linear_superposition(self):
    a, b = np.array([[0.1, 0.5]]), np.array([[0.4, 0.9]])
    np.testing.assert_allclose(hons.image(np.vstack([a, b])), hons.image(a) + hons.image(b), rtol=1e-14)

  def test_zero_lifetime_has_zero_weight(self):
    np.testing.assert_array_equal(hons.image([[0.3, 0.3]]), np.zeros(400))

  def test_chunk_boundary(self):
    one = np.array([[0.25, 0.75]])
    np.testing.assert_allclose(hons.image(np.repeat(one, 20001, axis=0)),
                               hons.image(one) * 20001, rtol=1e-12, atol=1e-11)

  def test_dimension_concatenation(self):
    diagram = pd.DataFrame([(0, 0, np.inf), (1, 0.2, 0.7), (2, 0.3, np.inf)],
                           columns=["dimension", "birth", "death"])
    vector = hons.rho(diagram)
    np.testing.assert_array_equal(vector[:400], hons.image([[0.2, 0.7]]))
    np.testing.assert_array_equal(vector[400:], hons.image([[0.3, 1.00001]]))

  def test_invalid_pairs(self):
    for values in [[[0.8, 0.2]], [[0, np.nan]], [[0, np.inf]], [[0, 1, 2]]]:
      with self.subTest(values=values), self.assertRaises(ValueError):
        hons.image(values)


class ScoreProperties(unittest.TestCase):
  def setUp(self):
    self.lowers = np.array([0.0, 2.0, 5.0])
    self.uppers = np.array([1.0, 4.0])
    self.vectors = {(i, j): np.array([lower ** 2, upper ** 2])
                    for i, upper in enumerate(self.uppers)
                    for j, lower in enumerate(self.lowers)}

  def test_nonlinear_uneven_grid(self):
    scores = hons.objective(self.vectors, self.lowers, self.uppers)
    for i in (0, 1):
      self.assertAlmostEqual(scores[(i, 1)], math.hypot(4.5, 5))
      self.assertAlmostEqual(scores[(i, 2)], math.hypot(7, 5))

  def test_constant_images(self):
    vectors = {k: np.ones(3) for k in self.vectors}
    self.assertEqual(set(hons.objective(vectors, self.lowers, self.uppers).values()), {0.0})

  def test_translation_rotation_and_scale(self):
    expected = hons.objective(self.vectors, self.lowers, self.uppers)
    rotated = {k: np.array([-v[1], v[0]]) + 12 for k, v in self.vectors.items()}
    self.assertEqual(hons.objective(rotated, self.lowers, self.uppers), expected)
    scaled = hons.objective({k: -2 * v for k, v in self.vectors.items()}, self.lowers, self.uppers)
    np.testing.assert_allclose(list(scaled.values()), 2 * np.array(list(expected.values())))

  def test_threshold_units(self):
    a = hons.objective(self.vectors, self.lowers, self.uppers)
    b = hons.objective(self.vectors, self.lowers * 5, self.uppers * 5)
    np.testing.assert_allclose(list(a.values()), np.array(list(b.values())) * 5)

  def test_incomplete_grid_rejected(self):
    del self.vectors[(0, 2)]
    with self.assertRaises(ValueError):
      hons.objective(self.vectors, self.lowers, self.uppers)

  def test_invalid_thresholds_and_images(self):
    for lowers in ([0, 0, 5], [5, 2, 0], [0, np.nan, 5]):
      with self.subTest(lowers=lowers), self.assertRaises(ValueError):
        hons.objective(self.vectors, lowers, self.uppers)
    self.vectors[(0, 0)] = np.array([np.nan, 1])
    with self.assertRaises(ValueError):
      hons.objective(self.vectors, self.lowers, self.uppers)


class SelectionProperties(unittest.TestCase):
  def test_exact_ties_use_sorted_cell_order(self):
    scores = {(2, 1): 1.0, (0, 1): 1.0}
    self.assertEqual(hons.select(scores, scores, scores, 0, 100), ((0, 1), [(0, 1), (2, 1)]))

  def test_zero_counts_are_feasible(self):
    scores = {(0, 1): 3.0, (0, 2): 1.0}
    counts = dict.fromkeys(scores, 0)
    self.assertEqual(hons.select(scores, counts, counts, 100, 100), ((0, 2), [(0, 2)]))

  def test_constraints_include_the_percentile_boundary(self):
    scores = {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0}
    counts = {(0, 1): 0, (0, 2): 10, (0, 3): 20}
    self.assertEqual(hons.select(scores, counts, counts, 50, 50), ((0, 2), [(0, 2)]))

  def test_independent_feasible_set_enumeration(self):
    rng = np.random.default_rng(772)
    keys = [(0, j) for j in range(1, 12)]
    scores = dict(zip(keys, rng.uniform(0, 100, len(keys))))
    first = dict(zip(keys, rng.integers(0, 100, len(keys))))
    second = dict(zip(keys, rng.integers(0, 100, len(keys))))
    def percentile(values, p):
      values = sorted(values)
      position = (len(values) - 1) * p / 100
      low, high = math.floor(position), math.ceil(position)
      return values[low] + (values[high] - values[low]) * (position - low)
    for p, q in itertools.product(range(0, 101, 10), repeat=2):
      a, b = percentile(first.values(), p), percentile(second.values(), q)
      feasible = [k for k in keys if first[k] >= a and second[k] >= b]
      expected = min(feasible, key=scores.get) if feasible else None
      self.assertEqual(hons.select(scores, first, second, p, q)[0], expected)

  def test_empty_or_nonfinite_scores_rejected(self):
    with self.assertRaises(ValueError):
      hons.select({}, {}, {}, 50, 50)
    with self.assertRaises(ValueError):
      hons.select({(0, 1): np.nan}, {(0, 1): 1}, {(0, 1): 1}, 50, 50)


if __name__ == "__main__":
  unittest.main()
