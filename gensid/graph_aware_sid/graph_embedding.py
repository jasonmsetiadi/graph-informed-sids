"""Graph-to-feature preprocessing for graph-aware clustering."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch


def build_fastrp_embeddings(
    adjacency: sp.csr_matrix,
    node_features: torch.Tensor | np.ndarray,
    embedding_dim: int,
    iteration_weights: Sequence[float],
    feature_weight: float,
    seed: int,
) -> np.ndarray:
    """Generate feature-aware FastRP embeddings from a weighted graph."""
    try:
        from fastrp import fit_csr
    except ImportError as exc:
        raise ImportError(
            "FastRP graph embeddings require fastrp==0.1.1. "
            "Install the dependencies from requirements.txt."
        ) from exc

    adjacency = adjacency.astype(np.float64, copy=False).tocsr()
    adjacency.sort_indices()
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"FastRP adjacency must be square, got {adjacency.shape}")
    if embedding_dim < 1:
        raise ValueError("FastRP embedding_dim must be >= 1")

    weights = [float(weight) for weight in iteration_weights]
    if not weights:
        raise ValueError("FastRP iteration_weights must contain at least one value")
    if not np.all(np.isfinite(weights)):
        raise ValueError("FastRP iteration_weights must be finite")
    if not np.isfinite(feature_weight) or feature_weight < 0.0:
        raise ValueError("FastRP feature_weight must be finite and >= 0")

    if isinstance(node_features, torch.Tensor):
        features = node_features.detach().cpu().numpy()
    else:
        features = np.asarray(node_features)
    if features.ndim != 2:
        raise ValueError(f"FastRP node_features must be 2D, got shape {features.shape}")
    if features.shape[0] != adjacency.shape[0]:
        raise ValueError(
            "FastRP node feature rows must match adjacency nodes, got "
            f"{features.shape[0]} and {adjacency.shape[0]}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("FastRP node_features contain non-finite values")

    embeddings = fit_csr(
        row_ptrs=adjacency.indptr,
        col_indices=adjacency.indices,
        values=adjacency.data,
        num_nodes=adjacency.shape[0],
        dim=int(embedding_dim),
        weights=weights,
        seed=int(seed),
        # InteractionGraphBuilder output is already made undirected, and
        # edge_index_to_sparse_adj symmetrizes it before this function.
        undirected=False,
        node_features=features,
        feature_weight=float(feature_weight),
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    expected_shape = (adjacency.shape[0], int(embedding_dim))
    if embeddings.shape != expected_shape:
        raise ValueError(f"FastRP returned shape {embeddings.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("FastRP returned non-finite embeddings")
    return embeddings


def build_graph_embeddings(
    method: str,
    adjacency: sp.csr_matrix,
    node_features: torch.Tensor | np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Dispatch a graph embedding method through one shared interface."""
    if method == "fastrp":
        return build_fastrp_embeddings(
            adjacency=adjacency,
            node_features=node_features,
            **kwargs,
        )
    raise ValueError(f"Unknown graph embedding method: {method!r}")
