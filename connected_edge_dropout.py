import torch
from torch import Tensor

from collections import defaultdict
from torch_geometric.utils import contains_isolated_nodes

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import numpy as np

class ConnectedEdgeDropout(torch.nn.Module):
    """
    p is set to match as closely as possible the effecive p for each node, where effective p is
    based on out degree for a directed graph
    
    """
    def __init__(self, edge_index: Tensor, p: float,
                 force_undirected: bool = False,
                 edge_weights: Tensor = None):
        super().__init__()

        if p < 0. or p > 1.:
            raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

        if edge_weights:
            if edge_weights.shape[1] != edge_weights.shape[0]:
                raise ValueError("Misalignment between edges and edge weights.")

            if torch.any(edge_weights < 0):
                raise ValueError("Edge weights must be non-negative.")

        self._contains_isolated_nodes = contains_isolated_nodes(edge_index)
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.p = p
        self.force_undirected = force_undirected

        if not self._contains_isolated_nodes:
            self._num_nodes = int(edge_index.max()) + 1
            self._degrees = torch.bincount(edge_index[0])

            self._structured_edge_index = self._get_structured_edge_index()
            self._directed = self._is_directed()
            self._connected = self._is_connected()

    def _is_directed(self):

        edges = defaultdict(list)
        for n1, n2 in self.edge_index.T:
            if n1 not in self._structured_edge_index[n2.item()]:
                return False
        return True

    def _num_components(self):
        edge_index = self.edge_index.numpy()
        A = coo_matrix((np.ones(edge_index.shape[1]), 
                        (edge_index[0], edge_index[1])), 
                        shape=(self._num_nodes, self._num_nodes))
        if self._directed:
            return connected_components(A, directed=True, connection='strong')[0]
        else:
            return connected_components(A, directed=True, connection='weak')[0]

    def _is_connected(self):
        return self._num_components() == 1

    def _get_structured_edge_index(self):

        adj_list = defaultdict(lambda: torch.empty(0, dtype=torch.int))
        for node1, node2 in self.edge_index.T:
            adj_list[node1.item()] = torch.cat([adj_list[node1.item()], node2[None]])
        return adj_list


    def _getpath(self, path, endnodes):
        get_rand_element = lambda x: x[torch.randint(x.shape[0], ())]
        while path[-1] not in endnodes:
            if self.edge_weights:
                nextnode = torch.multinomial(self._structured_edge_index[path[-1].item()], 1)
            else:
                nextnode = get_rand_element(self._structured_edge_index[path[-1].item()])
            path = torch.cat([path, nextnode[None]])
        return path

    def _erase_loop(self, path):
        newpath = torch.empty(0, dtype=torch.int)
        i = 0
        while i < path.shape[0]:
            for j in range(path.shape[0] - 1, i - 1, -1):
                if path[j] == path[i]:
                    newpath = torch.cat([newpath, path[j][None]])
                    i = j + 1
                    break
        return newpath

    def _sample_spanning_tree(self):
        tree = torch.randint(self._num_nodes, (1,))
        nodes_to_connect = torch.cat([torch.arange(tree[0]), 
                    torch.arange(tree[0] + 1, self._num_nodes)])
        tree_edges = torch.empty((0, 2), dtype=torch.int)
        tree_degrees = torch.zeros(self._num_nodes, dtype=torch.int)

        while nodes_to_connect.shape[0]:
            newnode_index = torch.randint(nodes_to_connect.shape[0], ())
            newnode = nodes_to_connect[newnode_index][None]
            path = self._getpath(newnode, tree)
            path = self._erase_loop(path)

            tree = torch.cat([tree, path[:-1]])

            nodes_to_connect = nodes_to_connect[~torch.isin(nodes_to_connect, path[:-1])]

            tree_edges = torch.cat([tree_edges,
                            torch.cat([path[:-1][:, None], path[1:][:, None]], 1)])

            tree_degrees += torch.bincount(path[:-1], minlength=self._num_nodes)
        return tree_edges, tree_degrees
        

    def __call__(self):
        if not self.training or self.p == 0.0:
            edge_mask = self.edge_index.new_ones(self.edge_index.size(1), dtype=torch.bool)
            return self.edge_index, edge_mask

        row, col = self.edge_index
        if not self._contains_isolated_nodes and self._connected:
            fixed_edges, tree_degrees = self._sample_spanning_tree()

            edge_mask = torch.rand(row.size(0), device=self.edge_index.device) >= \
                self.p + (tree_degrees/self._degrees)[row].to(self.edge_index.device)

            tree_row = fixed_edges[:, 0]
            tree_col = fixed_edges[:, 1]
            edge_keys = self.edge_index[0].long() * self._num_nodes + self.edge_index[1].long()
            tree_keys = tree_row.long() * self._num_nodes + tree_col.long()
            fixed_edge_mask = torch.isin(edge_keys, tree_keys)
            edge_mask[fixed_edge_mask] = True

        else:
            edge_mask = torch.rand(row.size(0), device=self.edge_index.device) >= self.p

        if self.force_undirected:
            edge_mask[row > col] = False

        edge_index = self.edge_index[:, edge_mask]

        if self.force_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

        return edge_index, edge_mask