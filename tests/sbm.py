import numpy as np

__all__ = ["sbm"]


def sbm(n, k, mean_degree, mixing, rng=None, connect=True):
    """Sample an SBM with the given mean degree and inter-block edge fraction. If connect=True,
    # returns a connected graph by adding (num_components - 1) edges to join the components

    Returns a list of edges and block ids
    """
    if not 0.0 <= mixing <= 1.0:
        raise ValueError(f"mixing must be in [0, 1], got {mixing}")
    if mean_degree < 0:
        raise ValueError(f"mean_degree must be non-negative, got {mean_degree}")
    if connect not in ("lcc", "repair", None):
        raise ValueError(f"connect must be 'lcc', 'repair' or None, got {connect!r}")

    sizes = _block_sizes(n, k)
    pairs_in = float(np.sum(sizes * (sizes - 1) // 2))
    pairs_out = n * (n - 1) / 2.0 - pairs_in
    if pairs_in == 0:
        raise ValueError("blocks are too small; need at least 2 nodes in one block")
    if pairs_out == 0:
        raise ValueError("need k >= 2 for mixing to be meaningful")

    n_edges = n * mean_degree / 2.0
    p_in = (1.0 - mixing) * n_edges / pairs_in
    p_out = mixing * n_edges / pairs_out
    if p_in > 1.0 or p_out > 1.0:
        raise ValueError(
            f"mean_degree={mean_degree} is unreachable at mixing={mixing} "
            f"(needs p_in={p_in:.3f}, p_out={p_out:.3f}); "
            f"lower the degree or use more blocks"
        )
    rng = np.random.default_rng(rng)
    edge_index, block_of = sbm_from_probs(n, k, p_in, p_out, rng)
    if connect == "lcc":
        edge_index, block_of = largest_component(edge_index, block_of)
    elif connect == "repair":
        edge_index = connect_components(edge_index, block_of, rng)
    return _symmetrize(edge_index), block_of


def _symmetrize(edge_index):
    """Add the reverse of every edge, so each undirected edge is stored in
    both directions."""
    if edge_index.shape[1] == 0:
        return edge_index
    return np.ascontiguousarray(np.concatenate([edge_index, edge_index[::-1]], axis=1))


def component_labels(edge_index, num_nodes):
    """Weak connected-component label per node, via union-find."""
    parent = np.arange(num_nodes)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(edge_index.shape[1]):
        a, b = find(edge_index[0, i]), find(edge_index[1, i])
        if a != b:
            parent[b] = a
    roots = np.array([find(v) for v in range(num_nodes)])
    return np.unique(roots, return_inverse=True)[1]


def largest_component(edge_index, block_of):
    """Restrict to the largest connected component and relabel nodes 0..n'-1."""
    n = block_of.size
    labels = component_labels(edge_index, n)
    keep = labels == np.bincount(labels).argmax()

    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))

    if edge_index.shape[1]:
        inside = keep[edge_index[0]] & keep[edge_index[1]]
        edge_index = remap[edge_index[:, inside]]
    else:
        edge_index = edge_index.copy()
    return np.ascontiguousarray(edge_index), block_of[keep]


def connect_components(edge_index, block_of, rng=None):
    """Join every component with a single added edge, preserving node count.

    Endpoints are chosen at random within each component. Note that each added
    edge is by construction a bridge.
    """
    rng = np.random.default_rng(rng)
    n = block_of.size
    labels = component_labels(edge_index, n)
    n_comp = labels.max() + 1
    if n_comp <= 1:
        return edge_index

    members = [np.flatnonzero(labels == c) for c in range(n_comp)]
    merged = list(members[0])
    new = []
    for c in range(1, n_comp):
        u = int(rng.choice(members[c]))
        v = int(rng.choice(merged))
        new.append((min(u, v), max(u, v)))
        merged.extend(members[c])

    extra = np.array(new, dtype=np.int64).T
    combined = np.concatenate([edge_index, extra], axis=1)
    combined = np.unique(combined.T, axis=0).T
    return np.ascontiguousarray(combined)


def sbm_from_probs(n, k, p_in, p_out, rng=None):
    """Sample an SBM directly from edge probabilities."""
    if n < 2:
        raise ValueError(f"need n >= 2, got {n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    for name, p in (("p_in", p_in), ("p_out", p_out)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {p}")

    rng = np.random.default_rng(rng)
    sizes = _block_sizes(n, k)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    block_of = np.repeat(np.arange(k), sizes)

    parts = []
    for a in range(k):
        if sizes[a] >= 2:
            parts.append(_within(starts[a], sizes[a], p_in, rng))
        for b in range(a + 1, k):
            if sizes[a] and sizes[b]:
                parts.append(
                    _between(starts[a], sizes[a], starts[b], sizes[b], p_out, rng)
                )

    parts = [p for p in parts if p.size]
    if not parts:
        return np.zeros((2, 0), dtype=np.int64), block_of

    edges = np.unique(np.concatenate(parts, axis=0), axis=0)
    return np.ascontiguousarray(edges.T.astype(np.int64)), block_of


def _block_sizes(n, k):
    """Contiguous blocks, as equal as possible."""
    return np.diff(np.linspace(0, n, k + 1).astype(int))


def _within(offset, size, p, rng):
    """Edges inside one block. Pairs are numbered 0..C(size,2)-1 in row-major
    order over the strict lower triangle, then inverted exactly."""
    n_pairs = size * (size - 1) // 2
    idx = _sample_indices(n_pairs, p, rng)
    if idx.size == 0:
        return _empty()

    # row_starts[u] = number of pairs before row u = u*(u-1)/2.
    # searchsorted inverts this exactly, with no floating-point sqrt.
    rows = np.arange(size, dtype=np.int64)
    row_starts = rows * (rows - 1) // 2
    u = np.searchsorted(row_starts, idx, side="right") - 1
    v = idx - row_starts[u]
    return np.stack([v + offset, u + offset], axis=1)


def _between(off_a, size_a, off_b, size_b, p, rng):
    """Edges between two disjoint blocks; index inversion is plain divmod."""
    idx = _sample_indices(size_a * size_b, p, rng)
    if idx.size == 0:
        return _empty()
    u, v = np.divmod(idx, size_b)
    return np.stack([u + off_a, v + off_b], axis=1)


def _sample_indices(n_pairs, p, rng):
    """Distinct pair-indices from [0, n_pairs), each included with probability p.

    Draws the count from a binomial first, so the O(n^2) candidate pairs are
    never materialised.
    """
    if n_pairs == 0 or p == 0.0:
        return np.zeros(0, dtype=np.int64)
    m = int(rng.binomial(n_pairs, p))
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    if m > n_pairs // 8:
        # Dense enough that choice() without replacement is worth its allocation.
        return rng.choice(n_pairs, size=m, replace=False).astype(np.int64)
    # Sparse: collisions are rare, so draw with replacement and top up.
    out = np.unique(rng.integers(0, n_pairs, size=m))
    while out.size < m:
        extra = rng.integers(0, n_pairs, size=2 * (m - out.size))
        out = np.unique(np.concatenate([out, extra]))
    return out[:m].astype(np.int64)


def _empty():
    return np.zeros((0, 2), dtype=np.int64)
