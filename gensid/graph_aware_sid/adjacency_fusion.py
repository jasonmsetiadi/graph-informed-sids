"""Build and combine adjacency views for graph-aware clustering."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors


def build_embedding_knn_adj(
    embeddings: torch.Tensor | np.ndarray,
    knn: int,
) -> sp.csr_matrix:
    """Build a symmetric cosine-similarity kNN graph from node embeddings."""
    if knn < 1:
        raise ValueError("knn must be >= 1")

    if isinstance(embeddings, torch.Tensor):
        x_np = embeddings.detach().cpu().numpy()
    else:
        x_np = np.asarray(embeddings)
    if x_np.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {x_np.shape}")
    if not np.all(np.isfinite(x_np)):
        raise ValueError("embeddings contain non-finite values")
    x_np = x_np.astype(np.float32, copy=False)
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

    embedding_adj = sp.coo_matrix(
        (similarities[keep], (rows[keep], cols[keep])),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    embedding_adj = embedding_adj.maximum(embedding_adj.T)
    embedding_adj.setdiag(0.0)
    embedding_adj.eliminate_zeros()
    return embedding_adj


def normalize_affinity_for_fusion(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    """Scale an affinity matrix by its Frobenius norm."""
    norm = float(np.sqrt(adjacency.multiply(adjacency).sum()))
    return adjacency if norm <= 0.0 else adjacency / norm


def fuse_adjacencies(
    interaction_adj: sp.csr_matrix,
    embedding_adj: sp.csr_matrix,
    fusion_alpha: float,
) -> sp.csr_matrix:
    """Fuse interaction and embedding affinities using alpha in [0, 1]."""
    if not 0.0 <= fusion_alpha <= 1.0:
        raise ValueError("fusion_alpha must be between 0 and 1")
    if interaction_adj.shape != embedding_adj.shape:
        raise ValueError(
            "Interaction and embedding adjacency shapes must match, got "
            f"{interaction_adj.shape} and {embedding_adj.shape}"
        )

    if fusion_alpha <= 0.0:
        return interaction_adj
    if fusion_alpha >= 1.0:
        return embedding_adj

    fused = (
        (1.0 - fusion_alpha) * normalize_affinity_for_fusion(interaction_adj)
        + fusion_alpha * normalize_affinity_for_fusion(embedding_adj)
    ).tocsr()
    fused.eliminate_zeros()
    return fused
