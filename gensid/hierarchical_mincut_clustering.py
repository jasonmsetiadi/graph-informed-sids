#!/usr/bin/env python3
"""
Hierarchical graph clustering for graph-aware SID / RQ-VAE supervision.

Graph input is built only through graph_utils.InteractionGraphBuilder.

Methods:
  - stacked_dmon: explicit dense stacked DMoN hierarchy with --cluster_sizes fine->coarse.
  - recursive_spectral: top-down recursive spectral splits.
  - recursive_kmeans: graph embedding preprocessing followed by top-down KMeans splits.
  - recursive_leiden: top-down recursive Leiden community detection splits.
  - recursive_dmon: top-down recursive local DMoN-style modularity splits.

For recursive spectral, graph-embedding KMeans, and DMoN methods, hierarchy is controlled by:
  --recursive_factor F
  --recursive_levels L

This creates coarse-to-fine partitions with target cluster counts roughly:
  level_1: F
  level_2: F^2
  ...
  level_L: F^L

Each parent cluster is split into up to F child clusters. Small parents with fewer
than F nodes are split into singleton-like clusters as needed.

For recursive Leiden, each parent is instead split at a stable resolution selected
from a log-spaced scan. The number of children and hierarchy depth are determined
by the graph. A branch stops when Leiden returns one community.

Main output:
  <output_dir>/<output_prefix>_labels.json

with labels indexed by global item_id:
  labels_json["coarse_to_fine_item_labels"]["level_1"] -> coarsest labels
  labels_json["coarse_to_fine_item_labels"]["level_2"] -> next finer labels

For RQ-VAE prefix alignment, use coarse_to_fine_item_labels in natural order:
  prefix-1       -> level_1
  prefix-1+2     -> level_2
  ...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from torch_geometric.nn import GCNConv

from graph_aware_sid.adjacency_fusion import build_embedding_knn_adj, fuse_adjacencies
from graph_aware_sid.fastrp import build_fastrp_embeddings
from graph_aware_sid.graph_utils import InteractionGraphBuilder, load_train_item_mask
from graph_aware_sid.magi import build_magi_embeddings
from graph_aware_sid.ns4gc import build_ns4gc_embeddings
from graph_aware_sid.s3gc import build_s3gc_embeddings


# -----------------------------------------------------------------------------
# Graph loading / filtering
# -----------------------------------------------------------------------------


def build_graph_from_builder(
    embeddings_path: str,
    interactions_path: str | None,
    graph_builder_dump_path: str | None,
    graph_type: str,
    graph_building_kwargs: Dict[str, Any],
    make_undirected: bool,
    skip_n_last: int | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, np.ndarray, np.ndarray, Any]:
    raw_embeddings = np.load(embeddings_path)
    if raw_embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings [num_items, dim], got {raw_embeddings.shape}")

    num_items, _ = raw_embeddings.shape
    x0 = torch.as_tensor(raw_embeddings, dtype=torch.float32)

    if interactions_path is not None:
        num_items, dim = x0.shape
        train_item_ids = load_train_item_mask(
            interactions_path,
            num_items,
            skip_n_last=skip_n_last,
        )
        builder = InteractionGraphBuilder(
            interactions_path=interactions_path,
            graph_type=graph_type,
            graph_building_kwargs=graph_building_kwargs,
            skip_n_last=skip_n_last,
            make_undirected=make_undirected,
            add_identity_features=False,
            all_item_ids=train_item_ids.tolist(),
        )
    else:
        raise ValueError("One of --interactions_path or --graph_builder_dump_path is required")

    pyg_data = builder.pyg_data
    node_map = builder.node_map.sort("node_id")
    item_ids = node_map["item_id"].to_numpy()

    if len(item_ids) == 0:
        raise ValueError("Graph has no nodes after preprocessing")

    x_graph = torch.as_tensor(raw_embeddings[item_ids], dtype=torch.float32)
    edge_index = pyg_data.edge_index.long()
    edge_weight = getattr(pyg_data, "edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.float()

    return x_graph, edge_index, edge_weight, item_ids, raw_embeddings, builder


def drop_zero_degree_nodes(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    item_ids: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, np.ndarray, np.ndarray]:
    num_nodes = x.size(0)
    deg = torch.zeros(num_nodes, dtype=torch.float32)

    if edge_index.numel() > 0:
        w = torch.ones(edge_index.size(1), dtype=torch.float32) if edge_weight is None else edge_weight.cpu().float()
        deg.scatter_add_(0, edge_index[0].cpu(), w)
        deg.scatter_add_(0, edge_index[1].cpu(), w)

    keep = deg > 0
    kept_old = keep.nonzero(as_tuple=False).view(-1)
    removed_item_ids = item_ids[(~keep).numpy()]

    if kept_old.numel() == num_nodes:
        return x, edge_index, edge_weight, item_ids, removed_item_ids

    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
    old_to_new[kept_old] = torch.arange(kept_old.numel(), dtype=torch.long)

    edge_keep = keep[edge_index[0].cpu()] & keep[edge_index[1].cpu()]
    edge_index_f = old_to_new[edge_index[:, edge_keep].cpu()]
    edge_weight_f = None if edge_weight is None else edge_weight.cpu()[edge_keep]

    return x[kept_old], edge_index_f, edge_weight_f, item_ids[kept_old.numpy()], removed_item_ids


def edge_index_to_sparse_adj(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
) -> sp.csr_matrix:
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    val = (
        np.ones(row.shape[0], dtype=np.float64) if edge_weight is None else edge_weight.cpu().numpy().astype(np.float64)
    )

    A = sp.coo_matrix((val, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A = 0.5 * (A + A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


# -----------------------------------------------------------------------------
# Shared output helpers
# -----------------------------------------------------------------------------


def project_labels_back_to_item_space(
    graph_labels: Sequence[np.ndarray],
    item_ids: np.ndarray,
    num_items: int,
) -> List[np.ndarray]:
    full = []
    for labels in graph_labels:
        arr = np.full((num_items,), -1, dtype=np.int64)
        arr[item_ids] = labels.astype(np.int64)
        full.append(arr)
    return full


def relabel_compact(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64)
    out = np.full_like(labels, -1)
    vals = [v for v in np.unique(labels).tolist() if v >= 0]
    for new_id, old_id in enumerate(vals):
        out[labels == old_id] = new_id
    return out


def graph_embedding_metadata(method: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Return method-specific graph embedding configuration for run metadata."""
    if method == "fastrp":
        return {
            "fastrp_dim": int(args.fastrp_dim),
            "fastrp_feature_weight": float(args.fastrp_feature_weight),
            "fastrp_iteration_weights": [float(x) for x in args.fastrp_iteration_weights],
            "fastrp_seed": int(args.fastrp_seed),
        }
    if method == "s3gc":
        return {
            "s3gc_dim": int(args.s3gc_dim),
            "s3gc_epochs": int(args.s3gc_epochs),
            "s3gc_batch_size": int(args.s3gc_batch_size),
            "s3gc_lr": float(args.s3gc_lr),
            "s3gc_weight_decay": float(args.s3gc_weight_decay),
            "s3gc_walk_length": int(args.s3gc_walk_length),
            "s3gc_context_size": int(args.s3gc_context_size),
            "s3gc_walks_per_node": int(args.s3gc_walks_per_node),
            "s3gc_num_negative_samples": int(args.s3gc_num_negative_samples),
            "s3gc_diffusion_hops": int(args.s3gc_diffusion_hops),
            "s3gc_ppr_alpha": float(args.s3gc_ppr_alpha),
            "s3gc_scale": args.s3gc_scale,
            "s3gc_seed": int(args.s3gc_seed),
            "s3gc_uses_edge_weights": False,
        }
    if method == "magi":
        return {
            "magi_dim": int(args.magi_dim),
            "magi_num_layers": int(args.magi_num_layers),
            "magi_epochs": int(args.magi_epochs),
            "magi_batch_size": int(args.magi_batch_size),
            "magi_lr": float(args.magi_lr),
            "magi_weight_decay": float(args.magi_weight_decay),
            "magi_num_neighbors": int(args.magi_num_neighbors),
            "magi_num_walks": int(args.magi_num_walks),
            "magi_walk_length": int(args.magi_walk_length),
            "magi_inference_batch_size": int(args.magi_inference_batch_size),
            "magi_inference_neighbors": int(args.magi_inference_neighbors),
            "magi_temperature": float(args.magi_temperature),
            "magi_seed": int(args.magi_seed),
            "magi_uses_edge_weights": False,
        }
    if method == "ns4gc":
        return {
            "ns4gc_variant": "weighted_local",
            "ns4gc_dim": int(args.ns4gc_dim),
            "ns4gc_num_layers": int(args.ns4gc_num_layers),
            "ns4gc_epochs": int(args.ns4gc_epochs),
            "ns4gc_batch_size": int(args.ns4gc_batch_size),
            "ns4gc_lr": float(args.ns4gc_lr),
            "ns4gc_weight_decay": float(args.ns4gc_weight_decay),
            "ns4gc_num_neighbors": int(args.ns4gc_num_neighbors),
            "ns4gc_inference_batch_size": int(args.ns4gc_inference_batch_size),
            "ns4gc_inference_neighbors": int(args.ns4gc_inference_neighbors),
            "ns4gc_feature_mask": float(args.ns4gc_feature_mask),
            "ns4gc_edge_drop": float(args.ns4gc_edge_drop),
            "ns4gc_similarity_threshold": float(args.ns4gc_similarity_threshold),
            "ns4gc_temperature": float(args.ns4gc_temperature),
            "ns4gc_neighbor_weight": float(args.ns4gc_neighbor_weight),
            "ns4gc_sparsity_weight": float(args.ns4gc_sparsity_weight),
            "ns4gc_training_mode": args.ns4gc_training_mode,
            "ns4gc_max_full_nodes": int(args.ns4gc_max_full_nodes),
            "ns4gc_seed": int(args.ns4gc_seed),
            "ns4gc_uses_edge_weights": True,
            "ns4gc_edge_weight_transform": "raw",
            "ns4gc_unit_weight_fallback": bool(not args.input_has_edge_weight),
        }
    raise ValueError(f"Unknown graph embedding method: {method!r}")


def save_outputs(
    args: argparse.Namespace,
    graph_labels_coarse_to_fine: Sequence[np.ndarray],
    item_ids: np.ndarray,
    raw_embeddings: np.ndarray,
    edge_index: torch.Tensor,
    metadata_extra: Dict[str, Any] | None = None,
    history: List[Dict[str, float]] | None = None,
    artifacts: Dict[str, np.ndarray] | None = None,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    num_items = raw_embeddings.shape[0]
    item_labels_coarse_to_fine = project_labels_back_to_item_space(
        graph_labels=graph_labels_coarse_to_fine,
        item_ids=item_ids,
        num_items=num_items,
    )

    labels_json = {
        "coarse_to_fine_item_labels": {
            f"level_{i}": labels.tolist() for i, labels in enumerate(item_labels_coarse_to_fine, start=1)
        },
        "fine_to_coarse_item_labels": {
            f"level_{i}": labels.tolist() for i, labels in enumerate(reversed(item_labels_coarse_to_fine), start=1)
        },
        "coarse_to_fine_graph_labels": {
            f"level_{i}": labels.astype(np.int64).tolist()
            for i, labels in enumerate(graph_labels_coarse_to_fine, start=1)
        },
    }

    labels_path = output_dir / f"{prefix}_labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_json, f)

    node_map_path = output_dir / f"{prefix}_node_map.csv"
    pd.DataFrame(
        {
            "node_id": np.arange(len(item_ids), dtype=np.int64),
            "item_id": item_ids.astype(np.int64),
        }
    ).to_csv(node_map_path, index=False)

    metadata = {
        "method": args.method,
        "embeddings_path": args.embeddings_path,
        "interactions_path": args.interactions_path,
        "graph_builder_dump_path": args.graph_builder_dump_path,
        "graph_type": args.graph_type,
        "graph_building_kwargs": json.loads(args.graph_building_kwargs),
        "num_items": int(num_items),
        "num_graph_nodes_used": int(len(item_ids)),
        "num_edges_used": int(edge_index.size(1)),
        "make_undirected": bool(not args.directed),
        "skip_n_last": args.skip_n_last,
        "drop_zero_degree_nodes": bool(args.drop_zero_degree_nodes),
        "graph_source": "graph_utils.InteractionGraphBuilder",
    }
    if args.method.startswith("recursive"):
        metadata.update(
            {
                "cluster_counts_coarse_to_fine": [int(len(np.unique(x[x >= 0]))) for x in graph_labels_coarse_to_fine],
            }
        )
        if args.method == "recursive_leiden":
            metadata["natural_depth"] = int(len(graph_labels_coarse_to_fine))
        else:
            metadata["recursive_factor"] = int(args.recursive_factor)
            metadata["recursive_levels"] = int(args.recursive_levels)
        if args.method == "recursive_spectral":
            metadata.update(
                {
                    "spectral_embedding_method": args.spectral_embedding_method,
                    "spectral_fusion_alpha": float(args.spectral_fusion_alpha),
                    "spectral_knn": int(args.spectral_knn),
                }
            )
            if args.spectral_embedding_method != "semantic":
                metadata.update(graph_embedding_metadata(args.spectral_embedding_method, args))
        elif args.method == "recursive_kmeans":
            metadata["graph_embedding_method"] = args.graph_embedding_method
            metadata.update(graph_embedding_metadata(args.graph_embedding_method, args))
        elif args.method == "recursive_leiden":
            metadata.update(
                {
                    "leiden_embedding_method": args.leiden_embedding_method,
                    "leiden_fusion_alpha": float(args.leiden_fusion_alpha),
                    "leiden_knn": int(args.leiden_knn),
                    "leiden_resolution_min": float(args.leiden_resolution_min),
                    "leiden_resolution_max": float(args.leiden_resolution_max),
                    "leiden_resolution_points": int(args.leiden_resolution_points),
                    "leiden_plateau_tolerance": float(args.leiden_plateau_tolerance),
                    "leiden_early_stop_patience": int(args.leiden_early_stop_patience),
                    "leiden_iterations": int(args.leiden_iterations),
                    "leiden_selection": "widest_stable_nontrivial_plateau_else_natural_stop",
                }
            )
            if args.leiden_embedding_method != "semantic":
                metadata.update(graph_embedding_metadata(args.leiden_embedding_method, args))
    else:
        metadata.update(
            {
                "cluster_sizes_fine_to_coarse": args.cluster_sizes,
                "cluster_sizes_coarse_to_fine": list(reversed(args.cluster_sizes or [])),
            }
        )
    if metadata_extra:
        metadata.update(metadata_extra)

    metadata_path = output_dir / f"{prefix}_meta.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if history is not None:
        history_path = output_dir / f"{prefix}_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    else:
        history_path = None

    if artifacts is None:
        artifacts = {}
    artifacts = dict(artifacts)
    artifacts["item_ids_graph_order"] = item_ids.astype(np.int64)
    for i, labels in enumerate(graph_labels_coarse_to_fine, start=1):
        artifacts[f"coarse_to_fine_level{i}_graph_labels"] = labels.astype(np.int64)
    np.savez_compressed(output_dir / f"{prefix}_artifacts.npz", **artifacts)

    print(f"Saved labels to:   {labels_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved node map to: {node_map_path}")
    print(f"Saved artifacts to:{output_dir / f'{prefix}_artifacts.npz'}")
    if history_path is not None:
        print(f"Saved history to:  {history_path}")
    print("\nFor RVQ prefix alignment, consume labels_json['coarse_to_fine_item_labels'].")
    print("  level_1 -> coarsest supervision for prefix-1")
    print("  level_2 -> finer supervision for prefix-2, etc.")


def build_selected_graph_embeddings(
    method: str,
    A: sp.csr_matrix,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    """Run one graph-aware embedding backend through its method-specific module."""
    if method == "fastrp":
        return build_fastrp_embeddings(
            adjacency=A,
            node_features=x,
            embedding_dim=args.fastrp_dim,
            iteration_weights=args.fastrp_iteration_weights,
            feature_weight=args.fastrp_feature_weight,
            seed=args.fastrp_seed,
        )
    if method == "s3gc":
        return build_s3gc_embeddings(
            edge_index=edge_index,
            node_features=x,
            embedding_dim=args.s3gc_dim,
            epochs=args.s3gc_epochs,
            batch_size=args.s3gc_batch_size,
            learning_rate=args.s3gc_lr,
            weight_decay=args.s3gc_weight_decay,
            walk_length=args.s3gc_walk_length,
            context_size=args.s3gc_context_size,
            walks_per_node=args.s3gc_walks_per_node,
            num_negative_samples=args.s3gc_num_negative_samples,
            diffusion_hops=args.s3gc_diffusion_hops,
            ppr_alpha=args.s3gc_ppr_alpha,
            scale=args.s3gc_scale,
            seed=args.s3gc_seed,
            device=device,
            print_every=args.s3gc_print_every,
        )
    if method == "magi":
        return build_magi_embeddings(
            edge_index=edge_index,
            node_features=x,
            embedding_dim=args.magi_dim,
            num_layers=args.magi_num_layers,
            epochs=args.magi_epochs,
            batch_size=args.magi_batch_size,
            learning_rate=args.magi_lr,
            weight_decay=args.magi_weight_decay,
            num_neighbors=args.magi_num_neighbors,
            num_walks=args.magi_num_walks,
            walk_length=args.magi_walk_length,
            inference_batch_size=args.magi_inference_batch_size,
            inference_neighbors=args.magi_inference_neighbors,
            temperature=args.magi_temperature,
            seed=args.magi_seed,
            device=device,
            print_every=args.magi_print_every,
        )
    if method == "ns4gc":
        return build_ns4gc_embeddings(
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_features=x,
            embedding_dim=args.ns4gc_dim,
            num_layers=args.ns4gc_num_layers,
            epochs=args.ns4gc_epochs,
            batch_size=args.ns4gc_batch_size,
            learning_rate=args.ns4gc_lr,
            weight_decay=args.ns4gc_weight_decay,
            num_neighbors=args.ns4gc_num_neighbors,
            inference_batch_size=args.ns4gc_inference_batch_size,
            inference_neighbors=args.ns4gc_inference_neighbors,
            feature_mask=args.ns4gc_feature_mask,
            edge_drop=args.ns4gc_edge_drop,
            similarity_threshold=args.ns4gc_similarity_threshold,
            temperature=args.ns4gc_temperature,
            neighbor_weight=args.ns4gc_neighbor_weight,
            sparsity_weight=args.ns4gc_sparsity_weight,
            training_mode=args.ns4gc_training_mode,
            max_full_nodes=args.ns4gc_max_full_nodes,
            seed=args.ns4gc_seed,
            device=device,
            print_every=args.ns4gc_print_every,
        )
    raise ValueError(f"Unknown graph embedding method: {method!r}")


# -----------------------------------------------------------------------------
# Spectral recursive clustering
# -----------------------------------------------------------------------------


def spectral_cluster_subset(
    A: sp.csr_matrix,
    A_embedding: sp.csr_matrix | None,
    fusion_alpha: float,
    subset: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    n = int(len(subset))
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    if k <= 1:
        return np.zeros((n,), dtype=np.int64)
    if n <= k:
        return np.arange(n, dtype=np.int64)

    A_sub = A[subset][:, subset].astype(np.float64).tocsr()
    if A_embedding is not None and fusion_alpha > 0.0:
        A_embedding_sub = A_embedding[subset][:, subset].astype(np.float64).tocsr()
        A_sub = fuse_adjacencies(A_sub, A_embedding_sub, fusion_alpha)
    if A_sub.nnz == 0:
        # No internal structure: deterministic split by node order inside the subset.
        return np.arange(n, dtype=np.int64) % k

    deg = np.asarray(A_sub.sum(axis=1)).reshape(-1)
    deg = np.maximum(deg, 1e-12)
    inv_sqrt = 1.0 / np.sqrt(deg)
    L = sp.eye(n, format="csr") - sp.diags(inv_sqrt) @ A_sub @ sp.diags(inv_sqrt)

    eig_k = min(k, n - 1)
    try:
        evals, evecs = eigsh(L, k=eig_k, which="SM")
        order = np.argsort(evals)
        Z = evecs[:, order]
        if Z.shape[1] > 1:
            Z = Z[:, 1:]
        if Z.shape[1] == 0:
            Z = evecs[:, order]
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    except Exception as e:
        print(f"[warn] eigsh failed for subset size={n}, k={k}: {e}. Falling back to adjacency rows.")
        Z = A_sub.toarray()

    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)
    return labels.astype(np.int64)


def embedding_cluster_subset(
    graph_embeddings: np.ndarray,
    subset: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Cluster one recursive parent using precomputed graph embeddings."""
    n = int(len(subset))
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    if k <= 1:
        return np.zeros((n,), dtype=np.int64)
    if n <= k:
        return np.arange(n, dtype=np.int64)

    features = graph_embeddings[subset]
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(features)
    return labels.astype(np.int64)


# -----------------------------------------------------------------------------
# Recursive Leiden clustering
# -----------------------------------------------------------------------------


def leiden_cluster_subset(
    A: sp.csr_matrix,
    A_embedding: sp.csr_matrix | None,
    fusion_alpha: float,
    subset: np.ndarray,
    seed: int,
    resolution_min: float,
    resolution_max: float,
    resolution_points: int,
    plateau_tolerance: float,
    early_stop_patience: int,
    n_iterations: int,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Run Leiden over a resolution grid and select a stable nontrivial plateau."""
    try:
        import igraph as ig
        import leidenalg as la
    except ImportError as exc:
        raise ImportError(
            "recursive_leiden requires the 'igraph' and 'leidenalg' packages. "
            "Install the dependencies from requirements.txt."
        ) from exc

    n = int(len(subset))
    if n == 0:
        return np.empty((0,), dtype=np.int64), []
    if n == 1:
        return np.zeros((n,), dtype=np.int64), []
    if resolution_min <= 0.0:
        raise ValueError("--leiden_resolution_min must be > 0")
    if resolution_max <= resolution_min:
        raise ValueError("--leiden_resolution_max must be greater than --leiden_resolution_min")
    if resolution_points < 2:
        raise ValueError("--leiden_resolution_points must be >= 2")
    if plateau_tolerance < 0.0:
        raise ValueError("--leiden_plateau_tolerance must be >= 0")
    if early_stop_patience < 0:
        raise ValueError("--leiden_early_stop_patience must be >= 0")
    if n_iterations == 0:
        raise ValueError("--leiden_iterations must be positive, or negative to run until convergence")

    A_sub = A[subset][:, subset].astype(np.float64).tocsr()
    if A_embedding is not None and fusion_alpha > 0.0:
        A_embedding_sub = A_embedding[subset][:, subset].astype(np.float64).tocsr()
        A_sub = fuse_adjacencies(A_sub, A_embedding_sub, fusion_alpha)
    A_upper = sp.triu(A_sub, k=1, format="coo")
    weights = A_upper.data.astype(np.float64, copy=False)
    if not np.all(np.isfinite(weights)):
        raise ValueError("Leiden received non-finite edge weights")
    if np.any(weights < 0.0):
        raise ValueError("Leiden RB configuration clustering requires non-negative edge weights")

    keep = weights > 0.0
    rows = A_upper.row[keep]
    cols = A_upper.col[keep]
    weights = weights[keep]
    if weights.size == 0:
        return np.arange(n, dtype=np.int64), []

    edges = np.column_stack((rows, cols)).tolist()
    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.es["weight"] = weights.tolist()

    resolutions = np.geomspace(resolution_min, resolution_max, num=resolution_points)
    candidates: List[Tuple[float, np.ndarray, int]] = []
    stable_nontrivial_seen = False
    unstable_streak = 0
    for resolution_idx, resolution in enumerate(resolutions, start=1):
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            n_iterations=n_iterations,
            seed=seed,
        )
        labels = relabel_compact(np.asarray(partition.membership, dtype=np.int64))
        num_communities = int(len(np.unique(labels)))
        candidates.append((float(resolution), labels, num_communities))
        print(
            f"  Leiden resolution {resolution_idx}/{resolution_points}: "
            f"gamma={resolution:.6g}, communities={num_communities}",
            flush=True,
        )
        if len(candidates) >= 2:
            previous_count = candidates[-2][2]
            relative_change = abs(num_communities - previous_count) / max(
                previous_count,
                num_communities,
                1,
            )
            stable_transition = relative_change <= plateau_tolerance
            if stable_transition and 1 < previous_count < n and 1 < num_communities < n:
                stable_nontrivial_seen = True
                unstable_streak = 0
            elif stable_nontrivial_seen and not stable_transition:
                unstable_streak += 1
            elif stable_nontrivial_seen:
                unstable_streak = 0

            if early_stop_patience > 0 and stable_nontrivial_seen and unstable_streak >= early_stop_patience:
                print(
                    "  Stopping resolution scan early after "
                    f"{unstable_streak} consecutive unstable transitions.",
                    flush=True,
                )
                break

    plateaus: List[Tuple[int, int]] = []
    plateau_start = 0
    for idx in range(1, len(candidates)):
        previous_count = candidates[idx - 1][2]
        current_count = candidates[idx][2]
        relative_change = abs(current_count - previous_count) / max(previous_count, current_count, 1)
        if relative_change > plateau_tolerance:
            plateaus.append((plateau_start, idx - 1))
            plateau_start = idx
    plateaus.append((plateau_start, len(candidates) - 1))

    # Boundary plateaus at one community or all singletons are resolution-range
    # saturation effects. Prefer a stable, nontrivial interior plateau.
    stable_nontrivial_plateaus = [
        (start, end)
        for start, end in plateaus
        if end > start
        and any(1 < candidates[idx][2] < n for idx in range(start, end + 1))
    ]
    one_community_plateaus = [
        (start, end)
        for start, end in plateaus
        if all(candidates[idx][2] == 1 for idx in range(start, end + 1))
    ]
    nonsaturated_plateaus = [
        (start, end)
        for start, end in plateaus
        if any(candidates[idx][2] < n for idx in range(start, end + 1))
    ]
    eligible_plateaus = (
        stable_nontrivial_plateaus
        or one_community_plateaus
        or nonsaturated_plateaus
        or plateaus
    )
    selected_start, selected_end = max(
        eligible_plateaus,
        key=lambda bounds: (
            np.log(candidates[bounds[1]][0]) - np.log(candidates[bounds[0]][0]),
            bounds[1] - bounds[0] + 1,
            -bounds[0],
        ),
    )

    plateau_midpoint = (selected_start + selected_end) / 2.0
    eligible_indices = [
        idx
        for idx in range(selected_start, selected_end + 1)
        if 1 < candidates[idx][2] < n
    ]
    if not eligible_indices:
        eligible_indices = list(range(selected_start, selected_end + 1))
    selected_idx = min(eligible_indices, key=lambda idx: (abs(idx - plateau_midpoint), idx))
    selected_resolution, labels, selected_communities = candidates[selected_idx]

    print(
        "  Selected Leiden plateau: "
        f"gamma=[{candidates[selected_start][0]:.6g}, {candidates[selected_end][0]:.6g}], "
        f"points={selected_end - selected_start + 1}, "
        f"chosen_gamma={selected_resolution:.6g}, communities={selected_communities}",
        flush=True,
    )

    history = []
    for idx, (resolution, _, num_communities) in enumerate(candidates):
        history.append(
            {
                "resolution": float(resolution),
                "clusters": float(num_communities),
                "subset_size": float(n),
                "selected": float(idx == selected_idx),
                "selected_plateau_start": float(candidates[selected_start][0]),
                "selected_plateau_end": float(candidates[selected_end][0]),
            }
        )
    return labels, history


# -----------------------------------------------------------------------------
# Recursive neural local splits: DMoN
# -----------------------------------------------------------------------------


class LocalAssignNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, k: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, k),
        )

    def forward(self, x):
        return self.net(x)


def dense_dmon_local_loss(
    adj: torch.Tensor,
    logits: torch.Tensor,
    balance_weight: float = 1.0,
    entropy_weight: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    s = F.softmax(logits, dim=-1)  # [N, K]
    deg = adj.sum(dim=1)
    m2 = deg.sum().clamp_min(1e-12)
    expected = torch.outer(deg, deg) / m2
    modularity_matrix = adj - expected
    modularity = torch.trace(s.t() @ modularity_matrix @ s) / m2

    # Simple collapse/balance regularizer. Keeps clusters from all collapsing.
    p = s.mean(dim=0)
    k = s.size(1)
    balance = ((p - 1.0 / k) ** 2).sum()

    # Positive entropy term encourages sharper assignments when minimized.
    entropy = -(s * torch.log(s.clamp_min(1e-12))).sum(dim=1).mean()

    loss = -modularity + balance_weight * balance + entropy_weight * entropy
    metrics = {
        "modularity": modularity.detach(),
        "balance": balance.detach(),
        "entropy": entropy.detach(),
    }
    return loss, metrics


def neural_cluster_subset(
    method: str,
    x_all: torch.Tensor,
    A: sp.csr_matrix,
    subset: np.ndarray,
    k: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    n = int(len(subset))
    if n == 0:
        return np.empty((0,), dtype=np.int64), []
    if k <= 1:
        return np.zeros((n,), dtype=np.int64), []
    if n <= k:
        return np.arange(n, dtype=np.int64), []
    if n > args.max_dense_nodes:
        raise ValueError(
            f"Recursive {method} local subset has {n} nodes, above --max_dense_nodes={args.max_dense_nodes}."
        )

    A_sub_np = A[subset][:, subset].astype(np.float32).toarray()
    x_sub = x_all[torch.as_tensor(subset, dtype=torch.long)].to(device)
    adj = torch.as_tensor(A_sub_np, dtype=torch.float32, device=device)

    model = LocalAssignNet(x_sub.size(1), args.assign_hidden_dim, k, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []

    for epoch in range(1, args.recursive_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x_sub)
        if method == "recursive_dmon":
            loss, metrics = dense_dmon_local_loss(
                adj=adj,
                logits=logits,
                balance_weight=args.dmon_balance_weight,
                entropy_weight=args.dmon_entropy_weight,
            )
        else:
            raise ValueError(f"Unknown neural recursive method: {method}")

        if torch.isnan(loss):
            raise ValueError(f"NaN loss in {method} local split")
        loss.backward()
        if args.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if epoch == 1 or epoch == args.recursive_epochs:
            row = {"epoch": float(epoch), "loss": float(loss.item()), "subset_size": float(n), "k": float(k)}
            row.update({name: float(value.item()) for name, value in metrics.items()})
            history.append(row)

    model.eval()
    with torch.no_grad():
        labels = model(x_sub).argmax(dim=-1).cpu().numpy().astype(np.int64)
    return labels, history


# -----------------------------------------------------------------------------
# Generic recursive top-down driver
# -----------------------------------------------------------------------------


def recursive_hierarchy(
    method: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    factor: int,
    levels: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    if factor < 2:
        raise ValueError("--recursive_factor must be >= 2")
    if levels < 1:
        raise ValueError("--recursive_levels must be >= 1")

    n = x.size(0)
    A = edge_index_to_sparse_adj(edge_index, edge_weight, n)
    A_embedding = None
    graph_embeddings = None
    if method == "recursive_spectral":
        if not 0.0 <= args.spectral_fusion_alpha <= 1.0:
            raise ValueError("--spectral_fusion_alpha must be between 0 and 1")
        if args.spectral_fusion_alpha > 0.0:
            if args.spectral_embedding_method == "semantic":
                spectral_embeddings = x
            else:
                spectral_embeddings = build_selected_graph_embeddings(
                    method=args.spectral_embedding_method,
                    A=A,
                    x=x,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    args=args,
                    device=device,
                )
            A_embedding = build_embedding_knn_adj(spectral_embeddings, args.spectral_knn)
            print(
                f"Built {args.spectral_embedding_method} embedding kNN graph: "
                f"nodes={n}, edges={A_embedding.nnz}, k={args.spectral_knn}, "
                f"fusion_alpha={args.spectral_fusion_alpha}"
            )
    elif method == "recursive_kmeans":
        graph_embeddings = build_selected_graph_embeddings(
            method=args.graph_embedding_method,
            A=A,
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            args=args,
            device=device,
        )
        print(
            f"Built {args.graph_embedding_method} graph embeddings: "
            f"shape={graph_embeddings.shape}"
        )

    parent_labels = np.zeros(n, dtype=np.int64)
    hierarchy: List[np.ndarray] = []
    history: List[Dict[str, float]] = []

    rng_seed = int(args.seed)

    for level in range(1, levels + 1):
        new_labels = np.full(n, -1, dtype=np.int64)
        next_cluster_id = 0
        parents = [p for p in np.unique(parent_labels).tolist() if p >= 0]

        for parent in parents:
            subset = np.where(parent_labels == parent)[0].astype(np.int64)
            k = min(factor, len(subset))

            if method == "recursive_spectral":
                local = spectral_cluster_subset(
                    A=A,
                    A_embedding=A_embedding,
                    fusion_alpha=args.spectral_fusion_alpha,
                    subset=subset,
                    k=k,
                    seed=rng_seed + level * 1009 + int(parent),
                )
                local_history = []
            elif method == "recursive_kmeans":
                local = embedding_cluster_subset(
                    graph_embeddings=graph_embeddings,
                    subset=subset,
                    k=k,
                    seed=rng_seed + level * 1009 + int(parent),
                )
                local_history = []
            else:
                local, local_history = neural_cluster_subset(
                    method=method,
                    x_all=x,
                    A=A,
                    subset=subset,
                    k=k,
                    args=args,
                    device=device,
                )

            for local_id in sorted(np.unique(local).tolist()):
                mask = local == local_id
                new_labels[subset[mask]] = next_cluster_id
                next_cluster_id += 1

            for row in local_history:
                row = dict(row)
                row["level"] = float(level)
                row["parent"] = float(parent)
                history.append(row)

        new_labels = relabel_compact(new_labels)
        hierarchy.append(new_labels)
        parent_labels = new_labels
        print(f"recursive level {level}: clusters={len(np.unique(new_labels[new_labels >= 0]))}")

    return hierarchy, history


def recursive_leiden_hierarchy(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    """Recursively split active Leiden communities until every branch stops naturally."""
    n = x.size(0)
    A = edge_index_to_sparse_adj(edge_index, edge_weight, n)
    A_embedding = None
    if not 0.0 <= args.leiden_fusion_alpha <= 1.0:
        raise ValueError("--leiden_fusion_alpha must be between 0 and 1")
    if args.leiden_fusion_alpha > 0.0:
        if args.leiden_embedding_method == "semantic":
            leiden_embeddings = x
        else:
            leiden_embeddings = build_selected_graph_embeddings(
                method=args.leiden_embedding_method,
                A=A,
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                args=args,
                device=device,
            )
        A_embedding = build_embedding_knn_adj(leiden_embeddings, args.leiden_knn)
        print(
            f"Built {args.leiden_embedding_method} embedding kNN graph for Leiden: "
            f"nodes={n}, edges={A_embedding.nnz}, k={args.leiden_knn}, "
            f"fusion_alpha={args.leiden_fusion_alpha}"
        )
    parent_labels = np.zeros(n, dtype=np.int64)
    active_parents = {0}
    hierarchy: List[np.ndarray] = []
    history: List[Dict[str, float]] = []
    level = 1

    while active_parents:
        new_labels = np.full(n, -1, dtype=np.int64)
        next_active_parents = set()
        next_cluster_id = 0
        any_split = False

        for parent in [p for p in np.unique(parent_labels).tolist() if p >= 0]:
            subset = np.where(parent_labels == parent)[0].astype(np.int64)
            if parent in active_parents:
                print(
                    f"Leiden level {level}, parent {parent}: scanning {len(subset)} nodes",
                    flush=True,
                )
                local, local_history = leiden_cluster_subset(
                    A=A,
                    A_embedding=A_embedding,
                    fusion_alpha=args.leiden_fusion_alpha,
                    subset=subset,
                    seed=int(args.seed) + level * 1009 + int(parent),
                    resolution_min=args.leiden_resolution_min,
                    resolution_max=args.leiden_resolution_max,
                    resolution_points=args.leiden_resolution_points,
                    plateau_tolerance=args.leiden_plateau_tolerance,
                    early_stop_patience=args.leiden_early_stop_patience,
                    n_iterations=args.leiden_iterations,
                )
            else:
                local = np.zeros(len(subset), dtype=np.int64)
                local_history = []

            local_ids = sorted(np.unique(local).tolist())
            parent_split = len(local_ids) > 1
            any_split = any_split or parent_split

            for local_id in local_ids:
                mask = local == local_id
                child_id = next_cluster_id
                new_labels[subset[mask]] = child_id
                next_cluster_id += 1
                if parent_split and int(mask.sum()) > 1:
                    next_active_parents.add(child_id)

            for row in local_history:
                row = dict(row)
                row["level"] = float(level)
                row["parent"] = float(parent)
                history.append(row)

        new_labels = relabel_compact(new_labels)
        if not any_split:
            if not hierarchy:
                hierarchy.append(new_labels)
                print("recursive Leiden stopped at root: clusters=1")
            break

        hierarchy.append(new_labels)
        print(
            f"recursive Leiden level {level}: clusters={len(np.unique(new_labels[new_labels >= 0]))}",
            flush=True,
        )
        parent_labels = new_labels
        active_parents = next_active_parents
        level += 1

    return hierarchy, history


# -----------------------------------------------------------------------------
# Stacked DMoN baseline
# -----------------------------------------------------------------------------


class SparseEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = float(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.norm2(x)
        return x


class HierarchicalDMoN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embed_dim: int,
        cluster_sizes: Sequence[int],
        assign_hidden_dim: int,
        dropout: float,
        balance_weight: float,
        entropy_weight: float,
    ):
        super().__init__()
        self.cluster_sizes = list(cluster_sizes)
        self.balance_weight = float(balance_weight)
        self.entropy_weight = float(entropy_weight)
        self.encoder = SparseEncoder(in_dim, hidden_dim, embed_dim, dropout)
        self.assigners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, assign_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(assign_hidden_dim, k),
                )
                for k in self.cluster_sizes
            ]
        )
        self.refiners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in self.cluster_sizes
            ]
        )

    @staticmethod
    def to_dense_adj(edge_index, num_nodes, edge_weight, device):
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
        if edge_index.numel() > 0:
            if edge_weight is None:
                adj[edge_index[0], edge_index[1]] = 1.0
            else:
                adj[edge_index[0], edge_index[1]] = edge_weight.float()
        adj = 0.5 * (adj + adj.t())
        adj.fill_diagonal_(0.0)
        return adj

    @staticmethod
    def dmon_pool(
        x: torch.Tensor, adj: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = F.softmax(logits, dim=-1)  # [N, K]
        cluster_mass = s.sum(dim=0).clamp_min(1e-12).unsqueeze(-1)  # [K, 1]
        x_next = (s.t() @ x) / cluster_mass
        adj_next = s.t() @ adj @ s
        adj_next = 0.5 * (adj_next + adj_next.t())
        adj_next.fill_diagonal_(0.0)
        return x_next, adj_next, s

    def forward(self, x, edge_index, edge_weight=None):
        device = x.device
        z0 = self.encoder(x, edge_index, edge_weight)
        x_curr = z0
        adj_curr = self.to_dense_adj(edge_index, x.size(0), edge_weight, device)

        levels = []
        total_loss = x.new_tensor(0.0)

        for i, (assigner, refiner, k) in enumerate(zip(self.assigners, self.refiners, self.cluster_sizes), start=1):
            if k > x_curr.size(0):
                raise ValueError(f"Level {i}: k={k} > nodes={x_curr.size(0)}")

            logits = assigner(x_curr)
            level_loss, metrics = dense_dmon_local_loss(
                adj=adj_curr,
                logits=logits,
                balance_weight=self.balance_weight,
                entropy_weight=self.entropy_weight,
            )
            x_next, adj_next, s_probs = self.dmon_pool(x_curr, adj_curr, logits)
            x_next = refiner(x_next)

            total_loss = total_loss + level_loss
            levels.append(
                {
                    "level": i,
                    "s_logits": logits,
                    "s_probs": s_probs,
                    "loss": level_loss,
                    "modularity": metrics["modularity"],
                    "balance": metrics["balance"],
                    "entropy": metrics["entropy"],
                }
            )

            x_curr = x_next
            adj_curr = adj_next

        return {"encoded_nodes": z0, "levels": levels, "loss": total_loss}


def compose_assignments_to_original(level_probs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    projected = []
    running = None
    for probs in level_probs:
        running = probs if running is None else running @ probs
        projected.append(running)
    return projected


@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    epochs: int
    print_every: int
    grad_clip: float | None


def train_stacked_dmon(model, x, edge_index, edge_weight, cfg: TrainConfig) -> Dict[str, Any]:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index, edge_weight)
        loss = out["loss"]
        if torch.isnan(loss):
            raise ValueError("NaN loss in stacked_dmon")
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        row = {"epoch": float(epoch), "loss": float(loss.item())}
        for lvl in out["levels"]:
            row[f"level{lvl['level']}_loss"] = float(lvl["loss"].item())
            row[f"level{lvl['level']}_modularity"] = float(lvl["modularity"].item())
            row[f"level{lvl['level']}_balance"] = float(lvl["balance"].item())
            row[f"level{lvl['level']}_entropy"] = float(lvl["entropy"].item())
        history.append(row)

        if epoch == 1 or epoch % cfg.print_every == 0 or epoch == cfg.epochs:
            print(f"epoch={epoch:04d} loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_weight)
    out["history"] = history
    return out


def run_stacked_dmon(args, x_graph, edge_index, edge_weight, device):
    if not args.cluster_sizes:
        raise ValueError("--cluster_sizes is required for --method stacked_dmon")
    if x_graph.size(0) > args.max_dense_nodes:
        raise ValueError(
            f"Graph has {x_graph.size(0)} nodes; stacked_dmon uses dense adjacency. "
            f"Increase --max_dense_nodes if intentional."
        )

    model = HierarchicalDMoN(
        in_dim=x_graph.size(1),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        cluster_sizes=args.cluster_sizes,
        assign_hidden_dim=args.assign_hidden_dim,
        dropout=args.dropout,
        balance_weight=args.dmon_balance_weight,
        entropy_weight=args.dmon_entropy_weight,
    ).to(device)
    out = train_stacked_dmon(
        model=model,
        x=x_graph.to(device),
        edge_index=edge_index.to(device),
        edge_weight=None if edge_weight is None else edge_weight.to(device),
        cfg=TrainConfig(args.lr, args.weight_decay, args.epochs, args.print_every, args.grad_clip),
    )

    level_probs = [lvl["s_probs"].detach().cpu() for lvl in out["levels"]]
    projected = compose_assignments_to_original(level_probs)
    fine_to_coarse = [p.argmax(dim=-1).numpy().astype(np.int64) for p in projected]
    coarse_to_fine = list(reversed(fine_to_coarse))

    artifacts = {"encoded_nodes": out["encoded_nodes"].detach().cpu().numpy()}
    for i, lvl in enumerate(out["levels"], start=1):
        artifacts[f"level{i}_s_probs"] = lvl["s_probs"].detach().cpu().numpy()
        artifacts[f"level{i}_s_logits"] = lvl["s_logits"].detach().cpu().numpy()
    return coarse_to_fine, out["history"], artifacts


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical graph clustering for graph-aware SID")

    parser.add_argument(
        "--method",
        type=str,
        default="stacked_dmon",
        choices=[
            "stacked_dmon",
            "recursive_spectral",
            "recursive_kmeans",
            "recursive_leiden",
            "recursive_dmon",
        ],
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--interactions_path", type=str, default=None)
    parser.add_argument("--graph_builder_dump_path", type=str, default=None)

    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--graph_type", type=str, default="adjacent_cooc", choices=["adjacent_cooc", "windowed", "rw"])
    parser.add_argument("--graph_building_kwargs", type=str, default="{}")
    parser.add_argument("--skip_n_last", type=int, default=2)
    parser.add_argument("--directed", action="store_true", help="Pass make_undirected=False to InteractionGraphBuilder")
    parser.add_argument("--drop_zero_degree_nodes", action="store_true")

    # Stacked DMoN explicit hierarchy.
    parser.add_argument(
        "--cluster_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Only for stacked_dmon. Fine-to-coarse sizes, e.g. --cluster_sizes 128 32",
    )

    # Recursive hierarchy.
    parser.add_argument("--recursive_factor", type=int, default=4)
    parser.add_argument("--recursive_levels", type=int, default=2)
    parser.add_argument("--recursive_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--spectral_embedding_method",
        choices=["semantic", "fastrp", "s3gc", "magi", "ns4gc"],
        default="semantic",
        help="Embedding source converted to a cosine kNN graph for recursive_spectral.",
    )
    parser.add_argument(
        "--spectral_fusion_alpha",
        "--spectral_semantic_weight",
        dest="spectral_fusion_alpha",
        type=float,
        default=0.0,
        help="Embedding-kNN adjacency weight in [0, 1] for recursive_spectral. Zero preserves graph-only clustering.",
    )
    parser.add_argument(
        "--spectral_knn",
        "--spectral_semantic_knn",
        dest="spectral_knn",
        type=int,
        default=20,
        help="Number of cosine-similarity neighbors used for the recursive_spectral embedding graph.",
    )
    parser.add_argument(
        "--graph_embedding_method",
        choices=["fastrp", "s3gc", "magi", "ns4gc"],
        default="fastrp",
        help="Graph-to-feature preprocessing used by recursive_kmeans.",
    )
    parser.add_argument("--fastrp_dim", type=int, default=256)
    parser.add_argument("--fastrp_feature_weight", type=float, default=1.0)
    parser.add_argument(
        "--fastrp_iteration_weights",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 1.0],
        help="FastRP hop weights, including the initial projection at index zero.",
    )
    parser.add_argument("--fastrp_seed", type=int, default=42)
    parser.add_argument("--s3gc_dim", type=int, default=256)
    parser.add_argument("--s3gc_epochs", type=int, default=200)
    parser.add_argument("--s3gc_batch_size", type=int, default=4096)
    parser.add_argument("--s3gc_lr", type=float, default=0.01)
    parser.add_argument("--s3gc_weight_decay", type=float, default=0.0)
    parser.add_argument("--s3gc_walk_length", type=int, default=3)
    parser.add_argument("--s3gc_context_size", type=int, default=3)
    parser.add_argument("--s3gc_walks_per_node", type=int, default=1)
    parser.add_argument("--s3gc_num_negative_samples", type=int, default=1)
    parser.add_argument("--s3gc_diffusion_hops", type=int, default=2)
    parser.add_argument("--s3gc_ppr_alpha", type=float, default=0.2)
    parser.add_argument(
        "--s3gc_scale",
        choices=["auto", "small", "medium", "large"],
        default="auto",
    )
    parser.add_argument("--s3gc_seed", type=int, default=42)
    parser.add_argument("--s3gc_print_every", type=int, default=10)
    parser.add_argument("--magi_dim", type=int, default=256)
    parser.add_argument("--magi_num_layers", type=int, default=2)
    parser.add_argument("--magi_epochs", type=int, default=200)
    parser.add_argument("--magi_batch_size", type=int, default=512)
    parser.add_argument("--magi_lr", type=float, default=0.001)
    parser.add_argument("--magi_weight_decay", type=float, default=0.0)
    parser.add_argument("--magi_num_neighbors", type=int, default=10)
    parser.add_argument("--magi_num_walks", type=int, default=20)
    parser.add_argument("--magi_walk_length", type=int, default=4)
    parser.add_argument("--magi_inference_batch_size", type=int, default=2048)
    parser.add_argument("--magi_inference_neighbors", type=int, default=15)
    parser.add_argument("--magi_temperature", type=float, default=0.5)
    parser.add_argument("--magi_seed", type=int, default=42)
    parser.add_argument("--magi_print_every", type=int, default=10)
    parser.add_argument("--ns4gc_dim", type=int, default=256)
    parser.add_argument("--ns4gc_num_layers", type=int, default=2)
    parser.add_argument("--ns4gc_epochs", type=int, default=200)
    parser.add_argument("--ns4gc_batch_size", type=int, default=1024)
    parser.add_argument("--ns4gc_lr", type=float, default=0.001)
    parser.add_argument("--ns4gc_weight_decay", type=float, default=0.0)
    parser.add_argument("--ns4gc_num_neighbors", type=int, default=15)
    parser.add_argument("--ns4gc_inference_batch_size", type=int, default=2048)
    parser.add_argument("--ns4gc_inference_neighbors", type=int, default=15)
    parser.add_argument("--ns4gc_feature_mask", type=float, default=0.2)
    parser.add_argument("--ns4gc_edge_drop", type=float, default=0.3)
    parser.add_argument("--ns4gc_similarity_threshold", type=float, default=0.6)
    parser.add_argument("--ns4gc_temperature", type=float, default=0.1)
    parser.add_argument("--ns4gc_neighbor_weight", type=float, default=1.0)
    parser.add_argument("--ns4gc_sparsity_weight", type=float, default=1.0)
    parser.add_argument(
        "--ns4gc_training_mode",
        choices=["sampled", "full"],
        default="sampled",
    )
    parser.add_argument("--ns4gc_max_full_nodes", type=int, default=10000)
    parser.add_argument("--ns4gc_seed", type=int, default=42)
    parser.add_argument("--ns4gc_print_every", type=int, default=10)
    parser.add_argument(
        "--leiden_embedding_method",
        choices=["semantic", "fastrp", "s3gc", "magi", "ns4gc"],
        default="semantic",
        help="Embedding source converted to a cosine kNN graph for recursive_leiden.",
    )
    parser.add_argument(
        "--leiden_fusion_alpha",
        type=float,
        default=0.0,
        help="Embedding-kNN adjacency weight in [0, 1] for recursive_leiden. Zero preserves graph-only clustering.",
    )
    parser.add_argument(
        "--leiden_knn",
        type=int,
        default=20,
        help="Number of cosine-similarity neighbors used for the recursive_leiden embedding graph.",
    )
    parser.add_argument(
        "--leiden_resolution_min",
        type=float,
        default=0.01,
        help="Lower bound of the recursive_leiden log-spaced resolution scan.",
    )
    parser.add_argument(
        "--leiden_resolution_max",
        type=float,
        default=100.0,
        help="Upper bound of the recursive_leiden log-spaced resolution scan.",
    )
    parser.add_argument(
        "--leiden_resolution_points",
        type=int,
        default=41,
        help="Number of resolutions evaluated by recursive_leiden.",
    )
    parser.add_argument(
        "--leiden_plateau_tolerance",
        type=float,
        default=0.05,
        help="Maximum relative community-count change between adjacent points in one stable plateau.",
    )
    parser.add_argument(
        "--leiden_early_stop_patience",
        type=int,
        default=10,
        help="Stop after this many unstable transitions once a stable nontrivial plateau has been found; zero disables.",
    )
    parser.add_argument(
        "--leiden_iterations",
        type=int,
        default=-1,
        help="Leiden optimization iterations; a negative value runs until convergence.",
    )
    parser.add_argument("--dmon_balance_weight", type=float, default=1.0)
    parser.add_argument("--dmon_entropy_weight", type=float, default=0.01)

    # Neural settings.
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--assign_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_dense_nodes", type=int, default=8000)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="graph_hierarchy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_building_kwargs = json.loads(args.graph_building_kwargs)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    x_graph, edge_index, edge_weight, item_ids, raw_embeddings, builder = build_graph_from_builder(
        embeddings_path=args.embeddings_path,
        interactions_path=args.interactions_path,
        graph_builder_dump_path=args.graph_builder_dump_path,
        graph_type=args.graph_type,
        graph_building_kwargs=graph_building_kwargs,
        make_undirected=not args.directed,
        skip_n_last=args.skip_n_last,
    )
    args.input_has_edge_weight = edge_weight is not None

    print(f"Method: {args.method}")
    print(f"Loaded graph nodes used: {x_graph.size(0)}, edges used: {edge_index.size(1)}, emb dim: {x_graph.size(1)}")

    if args.method == "stacked_dmon":
        labels_c2f, history, artifacts = run_stacked_dmon(args, x_graph, edge_index, edge_weight, device)
    elif args.method == "recursive_leiden":
        labels_c2f, history = recursive_leiden_hierarchy(
            x=x_graph,
            edge_index=edge_index,
            edge_weight=edge_weight,
            args=args,
            device=device,
        )
        artifacts = {}
    else:
        labels_c2f, history = recursive_hierarchy(
            method=args.method,
            x=x_graph,
            edge_index=edge_index,
            edge_weight=edge_weight,
            factor=args.recursive_factor,
            levels=args.recursive_levels,
            args=args,
            device=device,
        )
        artifacts = {}

    save_outputs(
        args=args,
        graph_labels_coarse_to_fine=labels_c2f,
        item_ids=item_ids,
        raw_embeddings=raw_embeddings,
        edge_index=edge_index,
        history=history,
        artifacts=artifacts,
    )


if __name__ == "__main__":
    main()
