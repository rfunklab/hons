"""Select network thresholds using persistent homology."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# shared persistence-image settings
RES = 20
SIGMA = 0.1
INF_REPLACEMENT = 1.00001
IM_RANGE = [0.0, 1.0, 0.0, INF_REPLACEMENT]


def _filtration_bounds(values, limits=None):
    """Resolve one finite range covering all supplied edge entry values."""
    values = np.asarray(list(values), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Filtration values must be finite.")
    if limits is None:
        return (float(values.min()), float(values.max())) if len(values) else (0.0, 1.0)
    limits = np.asarray(limits, dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[1] < limits[0]:
        raise ValueError("filtration_range must contain two finite bounds with start <= stop.")
    start, stop = map(float, limits)
    if np.any(values < start) or np.any(values > stop):
        raise ValueError("Filtration values must fall within filtration_range.")
    return start, stop


def persistence(nodes, edges, start=None, stop=None, max_dim=2) -> pd.DataFrame:
    """Compute positive-lifetime intervals through max_dim over Z/11Z.

    edges maps node pairs to numerical entry values. Normalize the supplied
    start/stop range to [0, 1], or infer bounds from the edge values when both
    are omitted. Use the same bounds when comparing multiple networks.
    All vertices enter at normalized zero. A constant range maps to zero.
    Clique expansion supplies higher simplices; infinite deaths remain infinite.
    """
    import gudhi
    if (start is None) != (stop is None):
        raise ValueError("Supply both start and stop, or omit both.")
    start, stop = _filtration_bounds(edges.values(), None if start is None else (start, stop))
    idx = {c: k for k, c in enumerate(nodes)}
    if len(idx) != len(nodes):
        raise ValueError("Node identifiers must be unique.")
    st = gudhi.SimplexTree()
    for k in range(len(nodes)):
        st.insert([k], 0.0)
    span = stop - start
    for (a, b), value in edges.items():
        value = float(value)
        if not np.isfinite(span):
            entry = (value / 2 - start / 2) / (stop / 2 - start / 2)
        else:
            entry = (value - start) / span if span else 0.0
        if not np.isfinite(entry):
            raise ValueError("Normalized filtration values must be finite.")
        st.insert([idx[a], idx[b]], entry)
    st.expansion(max_dim + 1)
    # include the top dimension when it is one of the requested dimensions
    st.compute_persistence(homology_coeff_field=11, min_persistence=0,
                           persistence_dim_max=st.dimension() <= max_dim)
    rows = []
    for d in range(max_dim + 1):
        for birth, death in st.persistence_intervals_in_dimension(d):
            rows.append((d, birth, death))
    return pd.DataFrame(rows, columns=["dimension", "birth", "death"])


def lifetimes(h: pd.DataFrame, dim: int) -> np.ndarray:
    """Copy birth/death pairs, replacing infinite deaths by 1.00001 for imaging."""
    d = h.loc[h["dimension"] == dim, ["birth", "death"]].to_numpy(dtype=float).copy()
    if len(d) == 0:
        return np.empty((0, 2))
    d[:, 1][np.isinf(d[:, 1])] = INF_REPLACEMENT
    return d


def image(bd: np.ndarray) -> np.ndarray:
    """Sample the lifetime-weighted Gaussian image on the fixed 20 by 20 grid.

    Input rows are birth/death pairs with finite deaths. The output uses
    y-major order and contains 400 values; an empty diagram gives zeros.
    """
    bd = np.asarray(bd, dtype=float)
    if bd.size == 0:
        return np.zeros(RES * RES)
    if (bd.ndim != 2 or bd.shape[1] != 2 or not np.isfinite(bd).all()
            or np.any(bd[:, 1] < bd[:, 0])):
        raise ValueError("Supply finite birth/death pairs with death >= birth.")
    b, d = bd[:, 0], bd[:, 1]
    pts = np.column_stack([b, d - b])
    w = d - b
    xs = np.linspace(IM_RANGE[0], IM_RANGE[1], RES)
    ys = np.linspace(IM_RANGE[2], IM_RANGE[3], RES)
    out = np.zeros((RES, RES))
    step = 20000
    for s in range(0, len(pts), step):                       # chunked: bounded memory
        P, W = pts[s:s + step], w[s:s + step]
        X = P[:, 0][:, None, None] - xs[None, None, :]
        Y = P[:, 1][:, None, None] - ys[None, :, None]
        K = np.exp(-(X ** 2 + Y ** 2) / (2 * SIGMA ** 2)) / (2 * np.pi * SIGMA ** 2)
        out += np.tensordot(W, K, 1)
    return out.flatten()


def rho(h: pd.DataFrame) -> np.ndarray:
    """Concatenate the H1 and H2 persistence images into an 800-value vector."""
    return np.concatenate([image(lifetimes(h, 1)), image(lifetimes(h, 2))])


def objective(rho_by_cell: dict, lowers: np.ndarray, uppers: np.ndarray) -> dict:
    """Score neighboring image differences per unit of threshold change.

    Keys are (upper_index, lower_index). Threshold arrays must be strictly
    increasing, and rho_by_cell must contain every grid cell. The first lower
    column supplies neighbors but cannot be selected. Directional means are
    combined using the Euclidean norm.
    """
    lowers, uppers = np.asarray(lowers, float), np.asarray(uppers, float)
    if any(a.ndim != 1 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0)
           for a in (lowers, uppers)) or len(lowers) < 2 or len(uppers) < 1:
        raise ValueError("Use increasing finite thresholds, with >=2 lower and >=1 upper values.")
    n_i, n_j = len(uppers), len(lowers)
    if set(rho_by_cell) != {(i, j) for i in range(n_i) for j in range(n_j)}:
        raise ValueError("Images must contain the complete threshold grid.")
    vectors = [np.asarray(v, float) for v in rho_by_cell.values()]
    if any(v.ndim != 1 or v.shape != vectors[0].shape or not np.isfinite(v).all()
           for v in vectors):
        raise ValueError("Images must be finite vectors of equal length.")
    def dist(a, b):
        return float(np.linalg.norm(rho_by_cell[a] - rho_by_cell[b], ord=2))
    out = {}
    for i in range(n_i):
        for j in range(1, n_j):
            if (i, j) not in rho_by_cell:
                continue
            terms_l = [dist((i, j), (i, j - 1)) / abs(lowers[j] - lowers[j - 1])]
            if j + 1 < n_j and (i, j + 1) in rho_by_cell:
                terms_l.append(dist((i, j), (i, j + 1)) / abs(lowers[j + 1] - lowers[j]))
            terms_u = []
            if i - 1 >= 0 and (i - 1, j) in rho_by_cell:
                terms_u.append(dist((i, j), (i - 1, j)) / abs(uppers[i] - uppers[i - 1]))
            if i + 1 < n_i and (i + 1, j) in rho_by_cell:
                terms_u.append(dist((i, j), (i + 1, j)) / abs(uppers[i + 1] - uppers[i]))
            gl = sum(terms_l) / len(terms_l)
            gu = sum(terms_u) / len(terms_u) if terms_u else 0.0
            out[(i, j)] = float(np.linalg.norm([gl, gu], ord=2))
    return out


def select(mag: dict, f1: dict, f2: dict, delta1: float, delta2: float):
    """Minimize the score subject to H1/H2 count-percentile constraints.

    Percentiles range from 0 to 100 and use all selectable cells. Return the
    minimum's cell and the cells within absolute score tolerance 1e-12 of it.
    Exact minima use sorted cell order; an infeasible setting returns (None, []).
    """
    keys = sorted(mag)
    if not keys:
        raise ValueError("Supply at least one selectable cell.")
    F1 = np.array([f1[k] for k in keys], float)
    F2 = np.array([f2[k] for k in keys], float)
    W = np.array([mag[k] for k in keys], float)
    if not np.isfinite([F1, F2, W]).all() or np.any(F1 < 0) or np.any(F2 < 0):
        raise ValueError("Scores and counts must be finite; counts must be nonnegative.")
    d1, d2 = np.percentile(F1, delta1), np.percentile(F2, delta2)
    feas = (F1 >= d1) & (F2 >= d2)
    if not feas.any():
        return None, []
    Wm = np.where(feas, W, np.inf)
    best = Wm.min()
    ties = [keys[t] for t in np.flatnonzero(np.isclose(Wm, best, rtol=0, atol=1e-12))]
    return keys[int(np.argmin(Wm))], ties


def threshold(graph, *, lower, upper, node_attribute=None, edge_attribute=None,
              filtration="filtration", filtration_range=None,
              constraints=(50, 25), return_details=False):
  """Select and return a thresholded NetworkX graph.

  Choose one scalar node or edge attribute for the inclusive threshold band.
  Edge entry values may use any finite numerical range. Normalize once using
  filtration_range or the full input graph's observed range. Every candidate
  uses those same bounds; all vertices enter at normalized zero.
  The input graph is unchanged. Infeasible constraints return None.
  With return_details=True, return (network, details), including the grid,
  persistence diagrams and images used in the calculation.
  """
  import networkx as nx
  if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
    raise ValueError("Supply a simple undirected NetworkX graph.")
  if nx.number_of_selfloops(graph):
    raise ValueError("Self-loops are not supported.")
  if (node_attribute is None) == (edge_attribute is None):
    raise ValueError("Choose exactly one node_attribute or edge_attribute.")
  lower, upper = np.asarray(lower, float), np.asarray(upper, float)
  if (lower.ndim != 1 or upper.ndim != 1 or len(lower) < 2 or len(upper) < 1
      or not np.isfinite(lower).all() or not np.isfinite(upper).all()
      or np.any(np.diff(lower) <= 0) or np.any(np.diff(upper) <= 0)):
    raise ValueError("Supply increasing finite lower and upper arrays (at least 2 and 1 values).")
  if len(constraints) != 2 or not np.isfinite(constraints).all() or any(
      p < 0 or p > 100 for p in constraints):
    raise ValueError("Supply two constraint percentiles between 0 and 100.")
  attribute = node_attribute or edge_attribute
  records = graph.nodes(data=True) if node_attribute else graph.edges(data=True)
  for *_, data in records:
    if attribute not in data or not np.isfinite(data[attribute]):
      raise ValueError(f"Every selected element needs a finite {attribute!r} attribute.")
  for _, _, data in graph.edges(data=True):
    if filtration not in data:
      raise ValueError(f"Every edge needs a {filtration!r} filtration value.")
  limits = _filtration_bounds((d[filtration] for _, _, d in graph.edges(data=True)),
                               filtration_range)

  def candidate(lo, hi):
    if node_attribute:
      return graph.subgraph(n for n, d in graph.nodes(data=True)
                            if lo <= d[attribute] <= hi).copy()
    result = graph.copy()
    result.remove_edges_from((a, b) for a, b, d in graph.edges(data=True)
                             if not lo <= d[attribute] <= hi)
    return result

  images, diagrams, h1, h2, rows = {}, {}, {}, {}, []
  for i, hi in enumerate(upper):
    for j, lo in enumerate(lower):
      cell = (i, j)
      network = candidate(lo, hi)
      edges = {(a, b): d[filtration] for a, b, d in network.edges(data=True)}
      diagram = persistence(list(network), edges, *limits)
      diagrams[cell], images[cell] = diagram, rho(diagram)
      h1[cell] = int((diagram.dimension == 1).sum())
      h2[cell] = int((diagram.dimension == 2).sum())
      rows.append({"upper_index": i, "lower_index": j, "lower": lo, "upper": hi,
                   "nodes": len(network), "edges": network.number_of_edges(),
                   "H1": h1[cell], "H2": h2[cell]})
  scores = objective(images, lower, upper)
  cell, ties = select(scores, h1, h2, *constraints)
  for row in rows:
    key = (row["upper_index"], row["lower_index"])
    row.update(selectable=key in scores, score=scores.get(key, np.nan))
  network = None if cell is None else candidate(lower[cell[1]], upper[cell[0]])
  if network is not None:
    network.graph["hons"] = {"lower": float(lower[cell[1]]), "upper": float(upper[cell[0]]),
                             "score": scores[cell], "H1": h1[cell], "H2": h2[cell]}
  if return_details:
    return network, {"cell": cell, "ties": ties, "grid": pd.DataFrame(rows),
                     "images": images, "diagrams": diagrams, "scores": scores,
                     "H1": h1, "H2": h2, "lower": lower, "upper": upper,
                     "filtration_range": limits}
  return network


def example_network(seed=7):
  """Create a small NetworkX example with cycles, cavities and extra connections.

  Node 'weight' is the threshold attribute; edge 'filtration' gives entry
  values in [0, 1]. All observations are generated, with no empirical records.
  """
  import networkx as nx
  rng = np.random.default_rng(seed)
  graph = nx.Graph()
  for block in range(3):
    base = 6 * block
    for i in range(6):
      graph.add_node(base + i, weight=float([4, 5, 6, 4, 5, 6][i]))
    for a, b in itertools.combinations(range(6), 2):
      if a // 2 != b // 2:
        graph.add_edge(base + a, base + b, filtration=float(rng.uniform(.05, .35)))
    graph.add_edge(base, base + 1, filtration=.8)
  for a, b in [(0, 6), (6, 12)]:
    graph.add_edge(a, b, filtration=.4)
  for i in range(18, 26):
    graph.add_node(i, weight=float(rng.choice([.5, 1.5, 2.5])))
    for target in rng.choice(18, size=3, replace=False):
      graph.add_edge(i, int(target), filtration=float(rng.uniform(.4, .7)))
  for i in range(26, 29):
    graph.add_node(i, weight=9.0)
    for target in range(26):
      graph.add_edge(i, target, filtration=float(rng.uniform(.85, 1)))
  return graph
