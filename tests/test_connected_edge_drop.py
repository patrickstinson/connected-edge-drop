import numpy as np
import torch
from torch import Tensor

from connected_edge_dropout import ConnectedEdgeDropout
from .sbm import sbm

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def num_components(edge_index, directed):
    edge_index = edge_index.numpy()
    num_nodes = edge_index.reshape(-1).max() + 1
    A = coo_matrix((np.ones(edge_index.shape[1]),
                    (edge_index[0], edge_index[1])),
                    shape=(num_nodes, num_nodes))
    if directed:
        return connected_components(A, directed=True, connection='strong')[0]
    else:
        return connected_components(A, directed=True, connection='weak')[0]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for mixing in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3):
        ei, blocks = sbm(3000, 6, mean_degree=8.0, mixing=mixing, rng=rng, connect="repair")
        for p in (0, .1, .25, .5, .75, 1.):
            sed = ConnectedEdgeDropout(torch.tensor(ei), p)
            surviving_edges = sed()[0]
            if not num_components(surviving_edges, directed=False) == 1:
                raise Exception("Failed to produce a connected graph.")
    print('Passed all tests.')


