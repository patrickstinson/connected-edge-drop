# connected-edge-drop

A replacement for `torch_geometric`'s `edge_drop` that preserves (strong) graph connectivity. 

Message passing propagates information over nodes in a graph. Edge dropout can be useful to prevent overfitting based on a static or arbitrary set of edges. However, vanilla dropout can lead to a disconnected graph, preventing the flow of information across a graph. connected-edge-drop ensures the graph remains connected. The spanning tree subgraph is sampled via Wilson's algorithm. If the graph is not connected before edge dropout, the method reverts to vanilla edge dropout. Supports undirected, directed, weighted, and unweighted edges.

## Installation

pip install git+https://github.com/patrickstinson/connected-edge-drop.git

## Usage

```python

from connected_edge_drop import ConnectedEdgeDrop

edge_drop = ConnectedEdgeDrop(p=0.5)
edge_index = edge_drop(edge_index)

```

## License

MIT — see [LICENSE](LICENSE).
