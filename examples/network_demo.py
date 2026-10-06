"""Select a NetworkX graph and draw the input and output with shared positions."""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

import hons

network = hons.example_network()
selected = hons.threshold(network, node_attribute="weight",
                          lower=[0, 1, 2, 3, 4], upper=[4, 5, 6, 7, 9],
                          filtration_range=(0, 1), constraints=(50, 25))
positions = nx.spring_layout(network, seed=7)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
for ax, graph, title in zip(axes, [network, selected], ["Input network", "Selected network"]):
  nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#b5c2c8", width=.7)
  nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=95,
                        node_color="#f07967" if graph is selected else "#24495b",
                        edgecolors="#173645", linewidths=.6)
  ax.set_title(f"{title}\n{len(graph)} nodes, {graph.number_of_edges()} edges")
  ax.set_xlim(-1.2, 1.2)
  ax.set_ylim(-1.2, 1.2)
  ax.axis("off")
output = Path(__file__).resolve().parents[1] / "docs"
output.mkdir(exist_ok=True)
fig.savefig(output / "network_demo.png", dpi=180)
plt.close(fig)
print(selected.graph["hons"])
print(f"Saved {output / 'network_demo.png'}")
