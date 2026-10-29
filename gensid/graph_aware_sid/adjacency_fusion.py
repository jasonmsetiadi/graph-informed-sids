"""Build and combine adjacency views for graph-aware clustering."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors


def build_semantic_knn_adj(x: torch.Tensor, knn: int) -> sp.csr_matrix:
    """Build a symmetric cosine-similarity kNN graph from item embeddings."""
    if knn < 1:
        raise ValueError("knn must be >= 1")

    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
    num_nodes = x_np.shape[0]
    if num_nodes <= 1:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    num_neighbors = min(knn + 1, num_nodes)
    neighbors = NearestNeighbors(n_neighbors=num_neighbors, metric="cosine", algorithm="brute")
    neighbors.fit(x_np)
    distances, indices = neighbors.kneighbors(x_np, return_distance=True)

    row_grid = np.arange(num_nodes, dtype=np.int64)[:, None]
    non_self = indices != row_grid
    neighbor_rank = np.cumsum(non_self, axis=1)
    keep_matrix = non_self & (neighbor_rank <= knn)

    rows = np.repeat(np.arange(num_nodes, dtype=np.int64), num_neighbors)
    cols = indices.reshape(-1).astype(np.int64, copy=False)
    similarities = (1.0 - distances.reshape(-1)).astype(np.float64, copy=False)
    keep = keep_matrix.reshape(-1) & np.isfinite(similarities) & (similarities > 0.0)

    semantic_adj = sp.coo_matrix(
        (similarities[keep], (rows[keep], cols[keep])),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    semantic_adj = semantic_adj.maximum(semantic_adj.T)
    semantic_adj.setdiag(0.0)
    semantic_adj.eliminate_zeros()
    return semantic_adj


def normalize_affinity_for_fusion(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    """Scale an affinity matrix by its Frobenius norm."""
    norm = float(np.sqrt(adjacency.multiply(adjacency).sum()))
    return adjacency if norm <= 0.0 else adjacency / norm


def fuse_adjacencies(
    interaction_adj: sp.csr_matrix,
    semantic_adj: sp.csr_matrix,
    semantic_weight: float,
) -> sp.csr_matrix:
    """Fuse interaction and semantic affinities using a weight in [0, 1]."""
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be between 0 and 1")
    if interaction_adj.shape != semantic_adj.shape:
        raise ValueError(
            "Interaction and semantic adjacency shapes must match, got "
            f"{interaction_adj.shape} and {semantic_adj.shape}"
        )

    if semantic_weight <= 0.0:
        return interaction_adj
    if semantic_weight >= 1.0:
        return semantic_adj

    fused = (
        (1.0 - semantic_weight) * normalize_affinity_for_fusion(interaction_adj)
        + semantic_weight * normalize_affinity_for_fusion(semantic_adj)
    ).tocsr()
    fused.eliminate_zeros()
    return fused
