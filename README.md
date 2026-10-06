# Higher-order network selection

<img src="https://raw.githubusercontent.com/rfunklab/hons/v0.1.2/docs/hons_logo.png" alt="HONS logo" width="300">

Network thresholding helps researchers extract interpretable structure from dense relational data by removing nodes or edges according to their properties. Choosing the cutoffs is harder—a ground-truth network is rarely available, thresholds are often selected by trial and error, and small changes can produce substantially different networks. Criteria based on individual nodes or edges can also overlook connections that contribute to higher-order structure.

HONS (higher-order network selection) is a Python package implementing the topological thresholding method described in [*Higher-Order Network Structure Inference: A Topological Approach to Network Selection*](https://arxiv.org/abs/2510.04884). **The method selects thresholds for numerical node attributes or edge weights using the network's cycles and cavities.** Persistent homology measures how these features appear and disappear as connections enter the network. HONS compares their persistence-image representations across a grid of candidate thresholds and selects the network with the smallest change under neighboring threshold choices. Researchers can require cycle and cavity counts to meet chosen percentiles among selectable candidate networks.

## Install

Use Python 3.11–3.13:

```sh
pip install hons
```

For plotting, run `pip install "hons[plot]"`.

## NetworkX example

`hons.example_network()` generates a small NetworkX graph with cycles, cavities and extra connections. The node attribute `weight` supplies the values to threshold, and the edge attribute `filtration` specifies when each connection enters the persistence calculation. For your own network, supply a measured node attribute or edge weight, such as frequency or contact duration.

```python
import hons

network = hons.example_network()
selected = hons.threshold(
    network,
    node_attribute="weight",
    lower=[0, 1, 2, 3, 4],
    upper=[4, 5, 6, 7, 9],
    filtration_range=(0, 1),
    constraints=(50, 25),
)

print(network.number_of_nodes(), network.number_of_edges())    # 29 143
print(selected.number_of_nodes(), selected.number_of_edges())  # 18 41
```

![Input network and selected network](https://raw.githubusercontent.com/rfunklab/hons/a763c0b43e692ad65491f555a7ab080b6846567b/docs/network_demo.png)

Both objects are ordinary NetworkX graphs. The input remains unchanged, and the selected graph retains its node and edge attributes in their original units. Run `python examples/network_demo.py` from a source checkout to recreate the plot.

The example selects nodes with `weight` between **4 and 7**, inclusive. The selected network has five positive-lifetime H1 features and three H2 features across its filtration. H1 features describe cycles; H2 features describe cavities. `constraints=(50, 25)` requires their counts to reach at least the 50th and 25th percentiles, respectively, among selectable candidate networks.

## Candidate thresholds

The `lower` and `upper` arrays define the search grid in the units of the attribute you want to threshold. The numbers above are choices for this small example. Choose bounds that cover the plausible cutoffs for your network; explicit lists, `numpy.linspace` and `numpy.geomspace` are all suitable ways to construct a grid. Grid range and spacing affect which settings are compared and which setting can be selected.

Supply strictly increasing arrays with at least two lower bounds and one upper bound. HONS evaluates every lower/upper combination. The smallest lower bound supplies a neighboring value for the score and is excluded from selection. Bounds are inclusive. A lower bound above its paired upper bound removes all nodes in node-filtering mode or all edges in edge-filtering mode.

Use `node_attribute="name"` to retain nodes whose named attribute falls within the bounds, along with the edges between those nodes. Use `edge_attribute="name"` to retain edges whose named weight falls within the bounds while preserving all nodes, including isolates. Choose one of these options for a simple, undirected NetworkX graph. HONS returns `None` when no candidate meets both feature-count constraints.

## Filtration values

A filtration describes the order in which connections enter the topology calculation. Each edge needs a finite numerical entry value—a time, a distance or another quantity appropriate to your application. Smaller values enter earlier. All vertices are present at the start, and a clique enters when its last edge enters. The threshold attribute determines which nodes or edges to retain; the filtration values determine when the retained edges enter the persistence calculation.

By default, HONS reads entry values from the edge attribute `filtration`. Use `filtration="name"` to choose another attribute. HONS accepts entry values in their original units and converts them internally to 0–1 for the persistence images. By default, the conversion uses the minimum and maximum edge values in the full input graph. Every candidate uses that same range. Set `filtration_range=(start, stop)` for a known observation window or a fixed range across multiple input graphs. For example, use `filtration="first_seen", filtration_range=(1920, 2021)` for edges dated by first appearance. The demo uses its generating interval, `(0, 1)`.

The fixed image grid and Gaussian width apply to this normalized scale. A change of units, with the range changed accordingly, preserves the calculation. Changing the entry order or the spacing of normalized entry values can change the result. For a strength-based filtration in which stronger edges should enter earlier, use negative strength as the entry value. A zero-width normalization range maps entry values to zero. An input graph without edges defaults to a 0–1 range.

## Scientific concept networks

In the [paper](https://arxiv.org/abs/2510.04884), we apply topological thresholding to concepts that co-occur in scientific articles. Nodes represent concepts, document frequency supplies the attribute to threshold, and an edge's first co-occurrence year supplies its entry value. `examples/concept_network.py` shows how to build a NetworkX graph from generated article–concept records and pass that graph to HONS.

| Example | Purpose | Command from a source checkout |
| --- | --- | --- |
| `network_demo.py` | Select nodes from an existing graph and plot the input and selected networks. | `python examples/network_demo.py` |
| `concept_network.py` | Build a concept network from article–concept records, then select nodes by document frequency. | `python examples/concept_network.py` |

The concept example uses a fixed observation window across candidates. The [paper](https://arxiv.org/abs/2510.04884)'s empirical analysis normalizes time separately for each retained network, using the year of the earliest article containing a retained concept and the corpus's final year. The [replication repository](https://github.com/rfunklab/hons-replication) includes the resulting persistence diagrams and reproduces the selection calculations from them. The repository also demonstrates edge-weight thresholding on an open workplace contact network.

## Scores and persistence diagrams

Set `return_details=True` to return `(selected, details)`. `details["grid"]` contains thresholds, network sizes, feature counts and scores; `details["filtration_range"]` records the range used for normalization. The dictionary also includes normalized persistence diagrams, image vectors and near-ties. The selected graph's `graph["hons"]` attribute records its bounds, score and feature counts.

The calculation uses H1 and H2 persistence images, a 20 × 20 sampling grid per dimension, Gaussian standard deviation 0.1, lifetime weights and coefficients in Z/11Z. Features still present when the filtration ends are included in selection; their infinite death times become 1.00001 for imaging. Neighboring image distances are divided by their threshold separations, averaged in each direction and combined by the Euclidean norm. Dense graphs can be expensive because computing H2 requires clique expansion through tetrahedra.

For an existing grid of persistence diagrams, use `rho`, `objective` and `select` directly. Diagrams are pandas DataFrames with `dimension`, `birth` and `death` columns and must use a common normalized scale. The dictionaries passed to `objective` and `select` use `(upper_index, lower_index)` keys. The `persistence` helper accepts node identifiers and an edge-to-entry-value mapping; supply common `start` and `stop` bounds when comparing multiple networks. When several cells have the exact minimum score, HONS selects the first in sorted cell order. The returned near-ties include cells within an absolute score tolerance of 1e-12.

Run the tests from a source checkout with `python -m unittest discover -s tests`.

## Citation and license

Adam Schroeder, Russell J. Funk, Jingyi Guan, Taylor Okonek and Lori Ziegelmeier. [*Higher-Order Network Structure Inference: A Topological Approach to Network Selection*](https://arxiv.org/abs/2510.04884).

The [MIT license](https://github.com/rfunklab/hons/blob/main/LICENSE) applies to this repository's code, documentation and original assets. Contact Russell J. Funk at rfunk@umn.edu.
