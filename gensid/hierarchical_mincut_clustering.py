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

For recursive spectral and DMoN, --recursive_factor is the maximum child count.
Spectral proposes k >= 2 from the largest normalized-Laplacian eigengap and
validates the partition with local weighted modularity. DMoN trains every
k >= 2 candidate for a short screening budget and fully retrains only the
winner. With the soft next-item surrogate enabled, candidate selection and
adaptive stopping combine local modularity with held-out sequential gain.
Graph-embedding KMeans sweeps k >= 2 and selects the best embedding-space
silhouette score, without constructing an additional graph.

Branch-count selection and hierarchy depth are controlled independently by
--recursive_branch_mode and --recursive_depth_mode. Item-level depth maps each
training interaction count to an exact hierarchy depth and excludes an item
from deeper clustering as soon as it reaches that target.

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
from sklearn.metrics import silhouette_score
from torch_geometric.nn import APPNP, GCNConv

from graph_aware_sid.adjacency_fusion import build_embedding_knn_adj, fuse_adjacencies
from graph_aware_sid.fastrp import build_fastrp_embeddings
from graph_aware_sid.graph_utils import InteractionGraphBuilder, load_train_item_mask
from graph_aware_sid.magi import build_magi_embeddings
from graph_aware_sid.ns4gc import build_ns4gc_embeddings
from graph_aware_sid.s3gc import build_s3gc_embeddings


# -----------------------------------------------------------------------------
# Sequential transition loading
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexedTransitions:
    sources: np.ndarray
    targets: np.ndarray
    source_offsets: np.ndarray


@dataclass(frozen=True)
class TransitionPairs:
    train: IndexedTransitions
    evaluation: IndexedTransitions


@dataclass(frozen=True)
class LocalTransitionPairs:
    train_sources: np.ndarray
    train_targets: np.ndarray
    eval_sources: np.ndarray
    eval_targets: np.ndarray

    @property
    def available(self) -> bool:
        return self.train_sources.size > 0 and self.eval_sources.size > 0


def load_transition_pairs(
    interactions_path: str,
    item_ids: np.ndarray,
    num_items: int,
    skip_n_last: int | None,
    holdout_fraction: float,
    seed: int,
) -> TransitionPairs:
    """Load directed consecutive training pairs in graph-node index space."""
    with open(interactions_path, "r") as f:
        interactions = json.load(f)

    item_to_node = np.full(num_items, -1, dtype=np.int64)
    item_to_node[item_ids] = np.arange(len(item_ids), dtype=np.int64)
    sources: List[int] = []
    targets: List[int] = []
    for items in interactions.values():
        train_items = items if skip_n_last in (None, 0) else items[:-skip_n_last]
        for source_item, target_item in zip(train_items[:-1], train_items[1:]):
            source_item = int(source_item)
            target_item = int(target_item)
            if not (0 <= source_item < num_items and 0 <= target_item < num_items):
                raise ValueError(
                    "Sequential transition contains an item outside the embeddings range: "
                    f"({source_item}, {target_item}) vs [0, {num_items})"
                )
            source_node = int(item_to_node[source_item])
            target_node = int(item_to_node[target_item])
            if source_node >= 0 and target_node >= 0:
                sources.append(source_node)
                targets.append(target_node)

    source_array = np.asarray(sources, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    eval_mask = rng.random(source_array.size) < holdout_fraction

    def index_split(split_sources: np.ndarray, split_targets: np.ndarray):
        order = np.argsort(split_sources, kind="stable")
        split_sources = split_sources[order]
        split_targets = split_targets[order]
        counts = np.bincount(split_sources, minlength=len(item_ids))
        offsets = np.empty(len(item_ids) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return IndexedTransitions(split_sources, split_targets, offsets)

    return TransitionPairs(
        train=index_split(source_array[~eval_mask], target_array[~eval_mask]),
        evaluation=index_split(source_array[eval_mask], target_array[eval_mask]),
    )


def local_transition_pairs(
    transitions: TransitionPairs | None,
    subset: np.ndarray,
    num_nodes: int,
    min_cluster_size: int,
    min_pairs: int,
    max_pairs: int,
    seed: int,
) -> LocalTransitionPairs | None:
    """Filter global pairs to a parent and convert them to local row indices."""
    if transitions is None or len(subset) < min_cluster_size:
        return None

    global_to_local = np.full(num_nodes, -1, dtype=np.int64)
    global_to_local[subset] = np.arange(len(subset), dtype=np.int64)

    def select(indexed: IndexedTransitions, split_seed: int):
        counts = indexed.source_offsets[subset + 1] - indexed.source_offsets[subset]
        local_sources = np.repeat(np.arange(len(subset), dtype=np.int64), counts)
        target_parts = [
            indexed.targets[indexed.source_offsets[node] : indexed.source_offsets[node + 1]]
            for node, count in zip(subset, counts)
            if count > 0
        ]
        if not target_parts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        local_targets = global_to_local[np.concatenate(target_parts)]
        keep = local_targets >= 0
        local_sources = local_sources[keep]
        local_targets = local_targets[keep]
        if max_pairs > 0 and local_sources.size > max_pairs:
            rng = np.random.default_rng(split_seed)
            selected = rng.choice(local_sources.size, size=max_pairs, replace=False)
            local_sources = local_sources[selected]
            local_targets = local_targets[selected]
        return local_sources, local_targets

    train_sources, train_targets = select(transitions.train, seed)
    eval_sources, eval_targets = select(transitions.evaluation, seed + 1)
    if train_sources.size < min_pairs or eval_sources.size < 1:
        return None
    return LocalTransitionPairs(
        train_sources=train_sources,
        train_targets=train_targets,
        eval_sources=eval_sources,
        eval_targets=eval_targets,
    )


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


def load_item_frequencies(
    interactions_path: str,
    item_ids: np.ndarray,
    num_items: int,
    skip_n_last: int | None,
) -> np.ndarray:
    """Return raw training interaction counts in graph-node order."""
    with open(interactions_path, "r") as f:
        interactions = json.load(f)

    frequencies = np.zeros(num_items, dtype=np.int64)
    for items in interactions.values():
        train_items = items if skip_n_last in (None, 0) else items[:-skip_n_last]
        for item in train_items:
            item_id = int(item)
            if item_id < 0 or item_id >= num_items:
                raise ValueError(
                    f"Interaction item_id={item_id} is outside embeddings range [0, {num_items})"
                )
            frequencies[item_id] += 1

    return frequencies[item_ids].astype(np.float64)


def item_target_hierarchy_depths(
    item_frequencies: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Map dataset-normalized log training frequency to a hierarchy depth."""
    depth_min = int(args.recursive_item_depth_min)
    depth_max = int(args.recursive_item_depth_max)
    if depth_min < 1:
        raise ValueError("--recursive_item_depth_min must be at least 1")
    if depth_max < depth_min:
        raise ValueError(
            "--recursive_item_depth_max must be >= --recursive_item_depth_min"
        )

    log_counts = np.log1p(np.asarray(item_frequencies, dtype=np.float64))
    log_min = float(log_counts.min())
    log_max = float(log_counts.max())
    if log_max == log_min:
        return np.full(log_counts.shape, depth_max, dtype=np.int64)
    normalized = (log_counts - log_min) / (log_max - log_min)

    if args.recursive_item_depth_function == "sigmoid":
        if args.recursive_item_depth_alpha <= 0.0:
            raise ValueError("--recursive_item_depth_alpha must be > 0")
        if not 0.0 <= args.recursive_item_depth_beta <= 1.0:
            raise ValueError("--recursive_item_depth_beta must be between 0 and 1")
        logits = -args.recursive_item_depth_alpha * (
            normalized - args.recursive_item_depth_beta
        )
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        rare_logit = np.clip(
            args.recursive_item_depth_alpha * args.recursive_item_depth_beta,
            -60.0,
            60.0,
        )
        popular_logit = np.clip(
            -args.recursive_item_depth_alpha
            * (1.0 - args.recursive_item_depth_beta),
            -60.0,
            60.0,
        )
        rare_endpoint = 1.0 / (1.0 + np.exp(-rare_logit))
        popular_endpoint = 1.0 / (1.0 + np.exp(-popular_logit))
        scaled = (sigmoid - popular_endpoint) / (
            rare_endpoint - popular_endpoint
        )
    else:
        if args.recursive_item_depth_gamma <= 0.0:
            raise ValueError("--recursive_item_depth_gamma must be > 0")
        scaled = 1.0 - np.power(normalized, args.recursive_item_depth_gamma)

    continuous = depth_min + (depth_max - depth_min) * scaled

    # Round halves upward, then enforce the requested hierarchy-depth range.
    return np.clip(
        np.floor(continuous + 0.5), depth_min, depth_max
    ).astype(np.int64)


def load_item_log_frequencies(
    interactions_path: str,
    item_ids: np.ndarray,
    num_items: int,
    skip_n_last: int | None,
) -> np.ndarray:
    """Return log training frequency in graph-node order."""
    return np.log1p(load_item_frequencies(interactions_path, item_ids, num_items, skip_n_last))


def rank_quantile_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    """Assign near-equal-size global popularity bins from rarest to most popular."""
    order = np.argsort(values, kind="stable")
    bin_ids = np.empty(len(values), dtype=np.int64)
    ranks = np.arange(len(values), dtype=np.int64)
    bin_ids[order] = np.minimum(ranks * num_bins // max(len(values), 1), num_bins - 1)
    return bin_ids


def normalize_active_parent_popularity(
    parent_labels: np.ndarray,
    active_parents: set[int],
    item_log_frequencies: np.ndarray | None,
) -> Dict[int, float]:
    """Min-max normalize active parents' mean log-frequency within one level."""
    if item_log_frequencies is None:
        return {}
    active_parent_ids = [
        parent
        for parent in np.unique(parent_labels).tolist()
        if parent >= 0 and parent in active_parents
    ]
    raw_popularity = {
        parent: float(item_log_frequencies[parent_labels == parent].mean())
        for parent in active_parent_ids
    }
    if not raw_popularity:
        return {}
    lower = min(raw_popularity.values())
    upper = max(raw_popularity.values())
    if upper == lower:
        return {parent: 0.0 for parent in active_parent_ids}
    return {
        parent: (value - lower) / (upper - lower)
        for parent, value in raw_popularity.items()
    }


def popularity_quality_threshold(
    cluster_popularity: float | None,
    args: argparse.Namespace,
) -> float:
    normalized_popularity = 0.0 if cluster_popularity is None else cluster_popularity
    return float(
        args.recursive_popularity_epsilon_base
        + args.recursive_popularity_alpha * normalized_popularity
    )


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
    if method == "semantic":
        return {
            "semantic_embedding_source": "input_item_embeddings",
            "semantic_uses_graph": False,
        }
    if method == "appnp":
        return {
            "appnp_num_prop": int(args.appnp_num_prop),
            "appnp_alpha": float(args.appnp_alpha),
            "appnp_uses_edge_weights": True,
            "appnp_edge_weight_transform": "raw",
            "appnp_unit_weight_fallback": bool(not args.input_has_edge_weight),
        }
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
    target_sid_lengths: np.ndarray | None = None,
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
    if target_sid_lengths is not None:
        if target_sid_lengths.shape != (num_items,):
            raise ValueError(
                f"target_sid_lengths must have shape ({num_items},), "
                f"got {target_sid_lengths.shape}"
            )
        labels_json["item_target_sid_lengths"] = (
            target_sid_lengths.astype(np.int64).tolist()
        )

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
                "recursive_popularity_epsilon_base": float(
                    args.recursive_popularity_epsilon_base
                ),
                "recursive_popularity_alpha": float(args.recursive_popularity_alpha),
                "recursive_popularity_normalization": (
                    "per_level_minmax_parent_mean_log_training_frequency"
                ),
            }
        )
        if args.method == "recursive_dmon":
            metadata["dmon_popularity_balance_lambda"] = float(
                args.dmon_popularity_balance_lambda
            )
            metadata["dmon_popularity_mass_objective"] = (
                args.dmon_popularity_mass_objective
            )
            metadata["dmon_popularity_mass_weights"] = (
                "parent_local_smoothed_training_interaction_counts_plus_one"
            )
            metadata["dmon_popularity_separation_values"] = (
                "parent_local_minmax_log1p_training_interaction_counts"
            )
            metadata["dmon_popularity_purity_bins"] = int(
                args.dmon_popularity_purity_bins
            )
            metadata["dmon_popularity_purity_binning"] = (
                "global_rank_quantiles_of_training_interaction_counts"
            )
            metadata["dmon_prediction_lambda"] = float(args.dmon_prediction_lambda)
            metadata["dmon_prediction_holdout_fraction"] = float(
                args.dmon_prediction_holdout_fraction
            )
            metadata["dmon_prediction_min_cluster_size"] = int(
                args.dmon_prediction_min_cluster_size
            )
            metadata["dmon_prediction_min_pairs"] = int(
                args.dmon_prediction_min_pairs
            )
            metadata["dmon_prediction_max_pairs"] = int(
                args.dmon_prediction_max_pairs
            )
            metadata["dmon_modularity_score_weight"] = float(
                args.dmon_modularity_score_weight
            )
            metadata["dmon_prediction_gain_threshold"] = float(
                args.dmon_prediction_gain_threshold
            )
            metadata["dmon_prediction_baseline"] = (
                "held_out_marginal_next_soft_assignment_cross_entropy"
            )
        if args.method == "recursive_leiden":
            metadata["natural_depth"] = int(len(graph_labels_coarse_to_fine))
        elif args.method in {"recursive_spectral", "recursive_dmon", "recursive_kmeans"}:
            metadata["recursive_branch_mode"] = args.recursive_branch_mode
            metadata["recursive_depth_mode"] = args.recursive_depth_mode
            if args.recursive_branch_mode == "fixed":
                metadata["recursive_factor"] = int(args.recursive_factor)
                metadata["recursive_branch_selection"] = "fixed_factor"
            else:
                metadata["recursive_factor_max"] = int(args.recursive_factor)
            if args.recursive_depth_mode == "fixed":
                metadata["recursive_levels"] = int(args.recursive_levels)
            elif args.recursive_depth_mode == "item":
                if target_sid_lengths is None:
                    raise ValueError(
                        "Item-level depth metadata requires explicit target SID lengths"
                    )
                metadata.update(
                    {
                        "recursive_item_depth_function": args.recursive_item_depth_function,
                        "recursive_item_depth_min": int(args.recursive_item_depth_min),
                        "recursive_item_depth_max": int(args.recursive_item_depth_max),
                        "recursive_item_depth_alpha": float(args.recursive_item_depth_alpha),
                        "recursive_item_depth_beta": float(args.recursive_item_depth_beta),
                        "recursive_item_depth_gamma": float(args.recursive_item_depth_gamma),
                        "recursive_item_depth_counts": {
                            str(depth): int(
                                np.count_nonzero(target_sid_lengths - 1 == depth)
                            )
                            for depth in np.unique(target_sid_lengths - 1)
                        },
                        "recursive_item_final_sid_length_counts": {
                            str(length): int(
                                np.count_nonzero(target_sid_lengths == length)
                            )
                            for length in np.unique(target_sid_lengths)
                        },
                        "recursive_item_depth_includes_unique_leaf": False,
                        "recursive_item_depth_popularity_normalization": (
                            "dataset_minmax_log1p_training_interaction_count"
                        ),
                        "recursive_hierarchy_levels": int(
                            len(graph_labels_coarse_to_fine)
                        ),
                    }
                )
            else:
                metadata["natural_depth"] = int(len(graph_labels_coarse_to_fine))
            if args.recursive_branch_mode == "adaptive" and args.method in {"recursive_spectral", "recursive_dmon"}:
                metadata["recursive_branch_selection"] = (
                    "normalized_laplacian_largest_nontrivial_eigengap"
                    if args.method == "recursive_spectral"
                    else (
                        "dmon_candidate_sweep_combined_modularity_prediction"
                        if args.dmon_prediction_lambda > 0.0
                        else "dmon_candidate_sweep_highest_local_modularity"
                    )
                )
                if args.method == "recursive_dmon":
                    metadata["dmon_selection_epochs"] = int(
                        min(args.dmon_selection_epochs, args.recursive_epochs)
                    )
                    metadata["dmon_final_epochs"] = int(args.recursive_epochs)
            elif args.recursive_branch_mode == "adaptive" and args.method == "recursive_kmeans":
                metadata["recursive_branch_selection"] = "kmeans_candidate_sweep_highest_silhouette"
                metadata["recursive_kmeans_min_child_size"] = int(args.recursive_kmeans_min_child_size)
                metadata["recursive_kmeans_silhouette_sample_size"] = int(
                    args.recursive_kmeans_silhouette_sample_size
                )
            if args.recursive_depth_mode == "adaptive" and args.method == "recursive_kmeans":
                metadata["recursive_kmeans_min_child_size"] = int(
                    args.recursive_kmeans_min_child_size
                )
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
    """Return semantic features or run a graph-aware embedding backend."""
    if method == "semantic":
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    if method == "appnp":
        if args.appnp_num_prop < 0:
            raise ValueError("--appnp_num_prop must be >= 0")
        if not 0.0 <= args.appnp_alpha <= 1.0:
            raise ValueError("--appnp_alpha must be between 0 and 1")

        propagation = APPNP(K=args.appnp_num_prop, alpha=args.appnp_alpha, cached=False)
        propagation.eval()
        with torch.no_grad():
            propagated = propagation(x, edge_index, edge_weight)
        return propagated.detach().cpu().numpy().astype(np.float32, copy=False)
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


def adaptive_branch_count(
    A: sp.csr_matrix,
    subset: np.ndarray,
    max_branches: int,
    context: str = "adaptive split",
) -> int:
    """Propose k >= 2 from the largest nontrivial normalized-Laplacian eigengap."""
    n = int(len(subset))
    if n <= 1:
        print(f"  {context}: nodes={n}, selected k=1 (singleton)", flush=True)
        return 1
    upper = min(int(max_branches), n - 1)
    if upper <= 1:
        print(f"  {context}: nodes={n}, selected k=1 (no larger candidate)", flush=True)
        return 1
    A_sub = A[subset][:, subset].astype(np.float64).tocsr()
    if A_sub.nnz == 0:
        print(f"  {context}: nodes={n}, selected k=1 (no internal edges)", flush=True)
        return 1
    degree = np.asarray(A_sub.sum(axis=1)).reshape(-1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    laplacian = sp.eye(n, format="csr") - sp.diags(inv_sqrt) @ A_sub @ sp.diags(inv_sqrt)
    try:
        if n <= upper + 1:
            values = np.linalg.eigvalsh(laplacian.toarray())
        else:
            # Candidate k is the number of eigenvalues before the gap, so one
            # extra eigenvalue is needed to assess every k through `upper`.
            values = eigsh(laplacian, k=upper + 1, which="SM", return_eigenvectors=False)
        values = np.sort(values)
        gaps = np.diff(values)[:upper]
        if gaps.size:
            for candidate_idx, gap in enumerate(gaps):
                candidate_k = candidate_idx + 1
                print(
                    f"  {context}: k={candidate_k}, "
                    f"lambda_k={values[candidate_idx]:.6g}, "
                    f"lambda_k+1={values[candidate_idx + 1]:.6g}, "
                    f"eigengap={gap:.6g}",
                    flush=True,
                )
            nontrivial_gaps = gaps[1:]
            if not nontrivial_gaps.size:
                print(f"  {context}: no nontrivial k candidate; keeping one branch", flush=True)
                return 1
            selected_k = int(np.argmax(nontrivial_gaps)) + 2
            print(
                f"  {context}: proposed k={selected_k} "
                f"with eigengap={gaps[selected_k - 1]:.6g}",
                flush=True,
            )
            return selected_k
    except Exception as error:
        print(f"[warn] adaptive eigengap failed for subset size={n}: {error}. Keeping one branch.")
    return 1


def local_weighted_modularity_gain(
    A: sp.csr_matrix,
    subset: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Return hard-partition modularity relative to the unsplit local graph."""
    if len(subset) == 0 or len(np.unique(labels)) <= 1:
        return 0.0
    A_sub = A[subset][:, subset].astype(np.float64).tocsr()
    degrees = np.asarray(A_sub.sum(axis=1)).reshape(-1)
    m2 = float(degrees.sum())
    if m2 <= 0.0:
        return 0.0

    modularity = 0.0
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        internal_weight = float(A_sub[mask][:, mask].sum())
        degree_sum = float(degrees[mask].sum())
        modularity += internal_weight / m2 - (degree_sum / m2) ** 2
    return float(modularity)


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


def sampled_silhouette_score(
    features: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> float:
    """Compute deterministic, cluster-aware sampled Euclidean silhouette."""
    n = int(len(labels))
    if sample_size <= 0 or n <= sample_size:
        return float(silhouette_score(features, labels, metric="euclidean"))

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for cluster_id in np.unique(labels):
        members = np.flatnonzero(labels == cluster_id)
        keep = min(2, len(members))
        selected.extend(rng.choice(members, size=keep, replace=False).tolist())

    target_size = min(n, max(int(sample_size), len(selected)))
    selected_set = set(selected)
    if len(selected) < target_size:
        remaining = np.asarray([idx for idx in range(n) if idx not in selected_set], dtype=np.int64)
        selected.extend(
            rng.choice(remaining, size=target_size - len(selected), replace=False).tolist()
        )
    sample_indices = np.asarray(sorted(selected), dtype=np.int64)
    return float(
        silhouette_score(
            features[sample_indices],
            labels[sample_indices],
            metric="euclidean",
        )
    )


def adaptive_kmeans_split(
    graph_embeddings: np.ndarray,
    subset: np.ndarray,
    max_branches: int,
    min_silhouette: float,
    min_child_size: int,
    silhouette_sample_size: int,
    seed: int,
    context: str,
    cluster_popularity: float | None = None,
    require_quality: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Select k by embedding-space silhouette and reject weak local splits."""
    n = int(len(subset))
    if min_child_size < 1:
        raise ValueError("--recursive_kmeans_min_child_size must be >= 1")
    if silhouette_sample_size < 0:
        raise ValueError("--recursive_kmeans_silhouette_sample_size must be >= 0")

    upper = min(int(max_branches), n - 1, n // min_child_size)
    if upper < 2:
        print(f"  {context}: nodes={n}, no feasible K-means split candidate", flush=True)
        return np.zeros(n, dtype=np.int64), []

    features = graph_embeddings[subset]
    candidates: List[Tuple[int, np.ndarray, float, int, int, bool]] = []
    history: List[Dict[str, float]] = []
    for k in range(2, upper + 1):
        candidate_seed = seed + k * 100_003
        labels = KMeans(n_clusters=k, random_state=candidate_seed, n_init=10).fit_predict(features)
        labels = labels.astype(np.int64)
        child_sizes = np.bincount(labels, minlength=k)
        children = int(np.count_nonzero(child_sizes))
        smallest_child = int(child_sizes.min())
        valid = children == k and smallest_child >= min_child_size
        score = -1.0
        if valid:
            try:
                score = sampled_silhouette_score(
                    features=features,
                    labels=labels,
                    sample_size=silhouette_sample_size,
                    seed=candidate_seed,
                )
                if not np.isfinite(score):
                    valid = False
                    score = -1.0
            except ValueError as error:
                print(f"  {context}: silhouette failed for k={k}: {error}", flush=True)
                valid = False
        print(
            f"  {context}: k={k}, children={children}, min_child_size={smallest_child}, "
            f"silhouette={score:.6g}, valid={valid}",
            flush=True,
        )
        history.append(
            {
                "candidate_k": float(k),
                "candidate_children": float(children),
                "candidate_min_child_size": float(smallest_child),
                "candidate_silhouette": float(score),
                "candidate_valid": float(valid),
            }
        )
        candidates.append((k, labels, score, children, smallest_child, valid))

    valid_candidates = [candidate for candidate in candidates if candidate[5]]
    if not valid_candidates:
        print(f"  {context}: all K-means candidates invalid; keeping one branch", flush=True)
        return np.zeros(n, dtype=np.int64), history

    selected_k, selected_labels, selected_score, selected_children, selected_min_child, _ = max(
        valid_candidates,
        key=lambda candidate: (candidate[2], -candidate[0]),
    )
    split_accepted = not require_quality or selected_score > min_silhouette
    print(
        f"  {context}: selected k={selected_k}, children={selected_children}, "
        f"min_child_size={selected_min_child}, silhouette={selected_score:.6g}, "
        f"popularity={0.0 if cluster_popularity is None else cluster_popularity:.6g}, "
        f"threshold={min_silhouette if require_quality else 'disabled'}, "
        f"split={'accepted' if split_accepted else 'rejected'}",
        flush=True,
    )
    history.append(
        {
            "selected_k": float(selected_k),
            "selected_children": float(selected_children),
            "selected_min_child_size": float(selected_min_child),
            "selected_silhouette": float(selected_score),
            "cluster_popularity": float(
                0.0 if cluster_popularity is None else cluster_popularity
            ),
            "silhouette_threshold": float(min_silhouette),
            "split_accepted": float(split_accepted),
        }
    )
    if not split_accepted:
        return np.zeros(n, dtype=np.int64), history
    return selected_labels, history


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


class SoftAssignmentPredictor(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.linear = nn.Linear(k, k)

    def forward(self, assignments: torch.Tensor) -> torch.Tensor:
        return self.linear(assignments)


def soft_next_item_prediction_loss(
    assignments: torch.Tensor,
    predictor: SoftAssignmentPredictor,
    sources: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    predicted_log_probabilities = F.log_softmax(
        predictor(assignments[sources]), dim=-1
    )
    target_assignments = assignments[targets]
    return -(target_assignments * predicted_log_probabilities).sum(dim=-1).mean()


def evaluate_soft_next_item_prediction(
    assignments: torch.Tensor,
    predictor: SoftAssignmentPredictor,
    sources: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    target_assignments = assignments[targets]
    predicted_log_probabilities = F.log_softmax(
        predictor(assignments[sources]), dim=-1
    )
    conditional_loss = -(
        target_assignments * predicted_log_probabilities
    ).sum(dim=-1).mean()
    marginal = target_assignments.mean(dim=0)
    baseline_loss = -(
        target_assignments * torch.log(marginal.clamp_min(1e-12))
    ).sum(dim=-1).mean()
    raw_gain = baseline_loss - conditional_loss
    normalized_gain = torch.where(
        baseline_loss > 1e-8,
        raw_gain / baseline_loss,
        torch.zeros_like(raw_gain),
    )
    return {
        "prediction_baseline_loss": float(baseline_loss.item()),
        "prediction_conditional_loss": float(conditional_loss.item()),
        "prediction_gain": float(raw_gain.item()),
        "prediction_normalized_gain": float(normalized_gain.item()),
    }


def dmon_quality_score(
    modularity_gain: float,
    prediction_normalized_gain: float | None,
    modularity_threshold: float,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    if prediction_normalized_gain is None or args.dmon_prediction_lambda <= 0.0:
        return modularity_gain, modularity_threshold
    beta = float(args.dmon_modularity_score_weight)
    score = beta * modularity_gain + (1.0 - beta) * prediction_normalized_gain
    threshold = (
        beta * modularity_threshold
        + (1.0 - beta) * args.dmon_prediction_gain_threshold
    )
    return score, threshold


def dense_dmon_local_loss(
    adj: torch.Tensor,
    logits: torch.Tensor,
    balance_weight: float = 1.0,
    entropy_weight: float = 0.01,
    popularity_weights: torch.Tensor | None = None,
    popularity_values: torch.Tensor | None = None,
    popularity_bin_membership: torch.Tensor | None = None,
    popularity_balance_lambda: float = 0.0,
    popularity_mass_objective: str = "balance",
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
    uniform_balance = ((p - 1.0 / k) ** 2).sum()
    balance = uniform_balance
    popularity_mass_loss = logits.new_zeros(())
    if popularity_balance_lambda > 0.0:
        if popularity_mass_objective in {"balance", "concentrate"}:
            if popularity_weights is None:
                raise ValueError(
                    "Popularity weights are required for balance and concentrate objectives"
                )
            popularity_mass = s.t() @ popularity_weights
        if popularity_mass_objective == "balance":
            popularity_mass_loss = ((popularity_mass - 1.0 / k) ** 2).sum()
            balance = (
                (1.0 - popularity_balance_lambda) * uniform_balance
                + popularity_balance_lambda * popularity_mass_loss
            )
        elif popularity_mass_objective == "concentrate":
            popularity_mass_loss = -(
                popularity_mass * torch.log(popularity_mass.clamp_min(1e-12))
            ).sum()
        elif popularity_mass_objective == "separate":
            if popularity_values is None:
                raise ValueError(
                    "Popularity values are required for the separate objective"
                )
            child_mass = s.sum(dim=0).clamp_min(1e-12)
            child_mean_popularity = (s.t() @ popularity_values) / child_mass
            popularity_mass_loss = -child_mean_popularity.var(unbiased=False)
        elif popularity_mass_objective == "purity":
            if popularity_bin_membership is None:
                raise ValueError(
                    "Popularity-bin membership is required for the purity objective"
                )
            child_bin_mass = s.t() @ popularity_bin_membership
            child_mass = child_bin_mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
            child_bin_probability = child_bin_mass / child_mass
            child_bin_entropy = -(
                child_bin_probability
                * torch.log(child_bin_probability.clamp_min(1e-12))
            ).sum(dim=1)
            child_weight = child_mass.squeeze(1) / child_mass.sum().clamp_min(1e-12)
            popularity_mass_loss = (child_weight * child_bin_entropy).sum()
        else:
            raise ValueError(
                f"Unknown DMoN popularity-mass objective: {popularity_mass_objective}"
            )

    # Positive entropy term encourages sharper assignments when minimized.
    entropy = -(s * torch.log(s.clamp_min(1e-12))).sum(dim=1).mean()

    loss = -modularity + balance_weight * balance + entropy_weight * entropy
    if popularity_mass_objective in {"concentrate", "separate", "purity"}:
        loss = loss + popularity_balance_lambda * popularity_mass_loss
    metrics = {
        "modularity": modularity.detach(),
        "balance": balance.detach(),
        "entropy": entropy.detach(),
        "popularity_mass_loss": popularity_mass_loss.detach(),
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
    seed: int | None = None,
    epochs: int | None = None,
    item_frequencies: np.ndarray | None = None,
    transition_pairs: LocalTransitionPairs | None = None,
) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, float] | None]:
    n = int(len(subset))
    if n == 0:
        return np.empty((0,), dtype=np.int64), [], None
    if k <= 1:
        return np.zeros((n,), dtype=np.int64), [], None
    if n <= k:
        return np.arange(n, dtype=np.int64), [], None
    if n > args.max_dense_nodes:
        raise ValueError(
            f"Recursive {method} local subset has {n} nodes, above --max_dense_nodes={args.max_dense_nodes}."
        )
    training_epochs = int(args.recursive_epochs if epochs is None else epochs)
    if training_epochs < 1:
        raise ValueError("Recursive DMoN training epochs must be >= 1")

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    A_sub_np = A[subset][:, subset].astype(np.float32).toarray()
    x_sub = x_all[torch.as_tensor(subset, dtype=torch.long)].to(device)
    adj = torch.as_tensor(A_sub_np, dtype=torch.float32, device=device)
    popularity_weights = None
    popularity_values = None
    popularity_bin_membership = None
    if args.dmon_popularity_balance_lambda > 0.0:
        if item_frequencies is None:
            raise ValueError("Item frequencies are required for popularity-aware DMoN")
        local_counts = torch.as_tensor(
            item_frequencies[subset], dtype=torch.float32, device=device
        )
        smoothed_counts = local_counts + 1.0
        popularity_weights = smoothed_counts / smoothed_counts.sum().clamp_min(1e-12)
        local_log_popularity = torch.log1p(local_counts)
        popularity_min = local_log_popularity.min()
        popularity_range = local_log_popularity.max() - popularity_min
        popularity_values = (
            local_log_popularity - popularity_min
        ) / popularity_range.clamp_min(1e-12)
        if args.dmon_popularity_mass_objective == "purity":
            local_bin_ids = torch.as_tensor(
                args.dmon_popularity_bin_ids[subset], dtype=torch.long, device=device
            )
            popularity_bin_membership = F.one_hot(
                local_bin_ids, num_classes=args.dmon_popularity_purity_bins
            ).to(dtype=torch.float32)

    model = LocalAssignNet(x_sub.size(1), args.assign_hidden_dim, k, args.dropout).to(device)
    prediction_enabled = (
        args.dmon_prediction_lambda > 0.0
        and transition_pairs is not None
        and transition_pairs.available
    )
    predictor = SoftAssignmentPredictor(k).to(device) if prediction_enabled else None
    parameters = list(model.parameters())
    if predictor is not None:
        parameters.extend(predictor.parameters())
    opt = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if prediction_enabled:
        train_sources = torch.as_tensor(
            transition_pairs.train_sources, dtype=torch.long, device=device
        )
        train_targets = torch.as_tensor(
            transition_pairs.train_targets, dtype=torch.long, device=device
        )
        eval_sources = torch.as_tensor(
            transition_pairs.eval_sources, dtype=torch.long, device=device
        )
        eval_targets = torch.as_tensor(
            transition_pairs.eval_targets, dtype=torch.long, device=device
        )
    history = []

    for epoch in range(1, training_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x_sub)
        if method == "recursive_dmon":
            dmon_loss, metrics = dense_dmon_local_loss(
                adj=adj,
                logits=logits,
                balance_weight=args.dmon_balance_weight,
                entropy_weight=args.dmon_entropy_weight,
                popularity_weights=popularity_weights,
                popularity_values=popularity_values,
                popularity_bin_membership=popularity_bin_membership,
                popularity_balance_lambda=args.dmon_popularity_balance_lambda,
                popularity_mass_objective=args.dmon_popularity_mass_objective,
            )
            prediction_loss = logits.new_zeros(())
            if predictor is not None:
                assignments = F.softmax(logits, dim=-1)
                prediction_loss = soft_next_item_prediction_loss(
                    assignments=assignments,
                    predictor=predictor,
                    sources=train_sources,
                    targets=train_targets,
                )
            loss = dmon_loss + args.dmon_prediction_lambda * prediction_loss
            metrics["dmon_loss"] = dmon_loss.detach()
            metrics["prediction_loss"] = prediction_loss.detach()
        else:
            raise ValueError(f"Unknown neural recursive method: {method}")

        if torch.isnan(loss):
            raise ValueError(f"NaN loss in {method} local split")
        loss.backward()
        if args.grad_clip is not None:
            nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        opt.step()

        if epoch == 1 or epoch == training_epochs:
            row = {"epoch": float(epoch), "loss": float(loss.item()), "subset_size": float(n), "k": float(k)}
            row.update({name: float(value.item()) for name, value in metrics.items()})
            history.append(row)

    model.eval()
    if predictor is not None:
        predictor.eval()
    prediction_metrics = None
    with torch.no_grad():
        final_assignments = F.softmax(model(x_sub), dim=-1)
        labels = final_assignments.argmax(dim=-1).cpu().numpy().astype(np.int64)
        if predictor is not None:
            prediction_metrics = evaluate_soft_next_item_prediction(
                assignments=final_assignments,
                predictor=predictor,
                sources=eval_sources,
                targets=eval_targets,
            )
            prediction_metrics["prediction_train_pairs"] = float(
                train_sources.numel()
            )
            prediction_metrics["prediction_eval_pairs"] = float(
                eval_sources.numel()
            )
            if history:
                history[-1].update(prediction_metrics)
    return labels, history, prediction_metrics


def adaptive_dmon_split(
    x_all: torch.Tensor,
    A: sp.csr_matrix,
    subset: np.ndarray,
    max_branches: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    context: str,
    cluster_popularity: float | None = None,
    require_quality: bool = True,
    item_frequencies: np.ndarray | None = None,
    transitions: TransitionPairs | None = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Select a local DMoN split using modularity and held-out predictability."""
    n = int(len(subset))
    upper = min(int(max_branches), n - 1)
    if upper < 2:
        print(f"  {context}: nodes={n}, no DMoN split candidate", flush=True)
        return np.zeros(n, dtype=np.int64), []

    A_sub = A[subset][:, subset]
    if A_sub.nnz == 0 or float(A_sub.sum()) <= 0.0:
        print(f"  {context}: no positive internal edge weight; keeping one branch", flush=True)
        return np.zeros(n, dtype=np.int64), []

    screening_epochs = min(int(args.dmon_selection_epochs), int(args.recursive_epochs))
    if screening_epochs < 1:
        raise ValueError("--dmon_selection_epochs must be >= 1")

    local_pairs = local_transition_pairs(
        transitions=transitions,
        subset=subset,
        num_nodes=x_all.size(0),
        min_cluster_size=args.dmon_prediction_min_cluster_size,
        min_pairs=args.dmon_prediction_min_pairs,
        max_pairs=args.dmon_prediction_max_pairs,
        seed=seed,
    )
    if args.dmon_prediction_lambda > 0.0 and local_pairs is None:
        print(
            f"  {context}: insufficient local transitions; using modularity-only scoring",
            flush=True,
        )

    modularity_threshold = popularity_quality_threshold(cluster_popularity, args)
    candidates: List[
        Tuple[int, np.ndarray, float, int, float | None, float]
    ] = []
    history: List[Dict[str, float]] = []
    for k in range(2, upper + 1):
        print(
            f"  {context}: screening DMoN candidate k={k} "
            f"for {screening_epochs} epochs",
            flush=True,
        )
        candidate_labels, candidate_history, prediction_metrics = neural_cluster_subset(
            method="recursive_dmon",
            x_all=x_all,
            A=A,
            subset=subset,
            k=k,
            args=args,
            device=device,
            seed=seed + k * 100_003,
            epochs=screening_epochs,
            item_frequencies=item_frequencies,
            transition_pairs=local_pairs,
        )
        for row in candidate_history:
            row["selection_screening"] = 1.0
        history.extend(candidate_history)
        children = int(len(np.unique(candidate_labels)))
        gain = local_weighted_modularity_gain(A, subset, candidate_labels)
        prediction_gain = (
            None
            if prediction_metrics is None
            else prediction_metrics["prediction_normalized_gain"]
        )
        quality_score, quality_threshold = dmon_quality_score(
            modularity_gain=gain,
            prediction_normalized_gain=prediction_gain,
            modularity_threshold=modularity_threshold,
            args=args,
        )
        valid = children > 1
        print(
            f"  {context}: k={k}, children={children}, "
            f"local_modularity_gain={gain:.6g}, "
            f"prediction_gain={prediction_gain if prediction_gain is not None else 'unavailable'}, "
            f"quality_score={quality_score:.6g}, valid={valid}",
            flush=True,
        )
        history.append(
            {
                "candidate_k": float(k),
                "candidate_children": float(children),
                "candidate_modularity_gain": float(gain),
                "candidate_prediction_normalized_gain": float(
                    prediction_gain if prediction_gain is not None else 0.0
                ),
                "candidate_prediction_available": float(prediction_gain is not None),
                "candidate_quality_score": float(quality_score),
                "candidate_quality_threshold": float(quality_threshold),
                "candidate_valid": float(valid),
            }
        )
        candidates.append(
            (k, candidate_labels, gain, children, prediction_gain, quality_score)
        )

    valid_candidates = [candidate for candidate in candidates if candidate[3] > 1]
    if not valid_candidates:
        print(f"  {context}: all DMoN candidates collapsed; keeping one branch", flush=True)
        return np.zeros(n, dtype=np.int64), history

    # Exact score ties prefer the simpler partition with the smaller requested k.
    (
        selected_k,
        screened_labels,
        screened_gain,
        screened_children,
        screened_prediction_gain,
        screened_quality_score,
    ) = max(
        valid_candidates,
        key=lambda candidate: (candidate[5], -candidate[0]),
    )
    print(
        f"  {context}: screening selected k={selected_k}, children={screened_children}, "
        f"local_modularity_gain={screened_gain:.6g}, "
        f"prediction_gain={screened_prediction_gain if screened_prediction_gain is not None else 'unavailable'}, "
        f"quality_score={screened_quality_score:.6g}",
        flush=True,
    )
    history.append(
        {
            "screening_selected_k": float(selected_k),
            "screening_selected_children": float(screened_children),
            "screening_selected_modularity_gain": float(screened_gain),
            "screening_selected_prediction_normalized_gain": float(
                screened_prediction_gain
                if screened_prediction_gain is not None
                else 0.0
            ),
            "screening_selected_prediction_available": float(
                screened_prediction_gain is not None
            ),
            "screening_selected_quality_score": float(screened_quality_score),
            "screening_epochs": float(screening_epochs),
        }
    )

    selected_seed = seed + selected_k * 100_003
    if screening_epochs < args.recursive_epochs:
        print(
            f"  {context}: retraining selected k={selected_k} "
            f"for {args.recursive_epochs} epochs",
            flush=True,
        )
        selected_labels, final_history, selected_prediction_metrics = neural_cluster_subset(
            method="recursive_dmon",
            x_all=x_all,
            A=A,
            subset=subset,
            k=selected_k,
            args=args,
            device=device,
            seed=selected_seed,
            epochs=args.recursive_epochs,
            item_frequencies=item_frequencies,
            transition_pairs=local_pairs,
        )
        for row in final_history:
            row["selection_screening"] = 0.0
        history.extend(final_history)
    else:
        selected_labels = screened_labels
        selected_prediction_metrics = (
            None
            if screened_prediction_gain is None
            else {"prediction_normalized_gain": screened_prediction_gain}
        )

    selected_children = int(len(np.unique(selected_labels)))
    selected_gain = local_weighted_modularity_gain(A, subset, selected_labels)
    selected_prediction_gain = (
        None
        if selected_prediction_metrics is None
        else selected_prediction_metrics["prediction_normalized_gain"]
    )
    normalized_popularity = 0.0 if cluster_popularity is None else cluster_popularity
    quality_score, quality_threshold = dmon_quality_score(
        modularity_gain=selected_gain,
        prediction_normalized_gain=selected_prediction_gain,
        modularity_threshold=modularity_threshold,
        args=args,
    )
    split_accepted = (
        selected_children > 1
        and (not require_quality or quality_score > quality_threshold)
    )
    print(
        f"  {context}: final k={selected_k}, children={selected_children}, "
        f"local_modularity_gain={selected_gain:.6g}, "
        f"prediction_gain={selected_prediction_gain if selected_prediction_gain is not None else 'unavailable'}, "
        f"quality_score={quality_score:.6g}, "
        f"popularity={normalized_popularity:.6g}, "
        f"threshold={quality_threshold if require_quality else 'disabled'}, "
        f"split={'accepted' if split_accepted else 'rejected'}",
        flush=True,
    )
    history.append(
        {
            "selected_k": float(selected_k),
            "selected_children": float(selected_children),
            "modularity_gain": float(selected_gain),
            "prediction_normalized_gain": float(
                selected_prediction_gain
                if selected_prediction_gain is not None
                else 0.0
            ),
            "prediction_available": float(selected_prediction_gain is not None),
            "quality_score": float(quality_score),
            "cluster_popularity": float(normalized_popularity),
            "modularity_threshold": float(modularity_threshold),
            "quality_threshold": float(quality_threshold),
            "split_accepted": float(split_accepted),
        }
    )
    if not split_accepted:
        return np.zeros(n, dtype=np.int64), history
    return selected_labels, history


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
    item_log_frequencies: np.ndarray | None = None,
    item_frequencies: np.ndarray | None = None,
    target_hierarchy_depths: np.ndarray | None = None,
    required_hierarchy_depth: int | None = None,
    transitions: TransitionPairs | None = None,
) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    if factor < 2:
        raise ValueError("--recursive_factor must be >= 2")
    adaptive_branch = args.recursive_branch_mode == "adaptive"
    cluster_adaptive_depth = args.recursive_depth_mode == "adaptive" and method in {
        "recursive_spectral",
        "recursive_dmon",
        "recursive_kmeans",
    }
    item_adaptive_depth = args.recursive_depth_mode == "item" and method in {
        "recursive_spectral",
        "recursive_dmon",
        "recursive_kmeans",
    }
    adaptive_depth = cluster_adaptive_depth or item_adaptive_depth
    if item_adaptive_depth:
        if target_hierarchy_depths is None or target_hierarchy_depths.shape != (x.size(0),):
            raise ValueError(
                "Item-level depth requires one target hierarchy depth per graph node"
            )
        max_target_depth = int(target_hierarchy_depths.max())
        if required_hierarchy_depth is not None:
            if required_hierarchy_depth < 1:
                raise ValueError("required_hierarchy_depth must be at least 1")
            max_target_depth = max(max_target_depth, int(required_hierarchy_depth))
    if not adaptive_depth and levels < 1:
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
            f"Prepared {args.graph_embedding_method} clustering embeddings: "
            f"shape={graph_embeddings.shape}"
        )

    parent_labels = np.zeros(n, dtype=np.int64)
    hierarchy: List[np.ndarray] = []
    history: List[Dict[str, float]] = []
    active_parents = {0}

    rng_seed = int(args.seed)

    level = 1
    while (item_adaptive_depth and level <= max_target_depth) or (
        not item_adaptive_depth and (adaptive_depth or level <= levels)
    ):
        new_labels = np.full(n, -1, dtype=np.int64)
        next_cluster_id = 0
        parents = [p for p in np.unique(parent_labels).tolist() if p >= 0]
        parent_popularity = normalize_active_parent_popularity(
            parent_labels,
            active_parents,
            item_log_frequencies,
        )
        next_active_parents = set()
        any_split = False

        for parent in parents:
            subset = np.where(parent_labels == parent)[0].astype(np.int64)
            if item_adaptive_depth:
                split_subset = subset[target_hierarchy_depths[subset] >= level]
                stopped_subset = subset[target_hierarchy_depths[subset] < level]
            else:
                split_subset = subset
                stopped_subset = np.empty(0, dtype=np.int64)
            if parent not in active_parents:
                local = np.zeros(len(split_subset), dtype=np.int64)
                local_history = []
            elif len(split_subset) <= 1:
                local = np.zeros(len(split_subset), dtype=np.int64)
                local_history = []
            else:
                split_context = f"{method} level {level}, parent {parent}"
                if not adaptive_branch:
                    k = min(factor, len(split_subset))
                elif method == "recursive_spectral":
                    k = adaptive_branch_count(
                        A,
                        split_subset,
                        factor,
                        context=split_context,
                    )
                    if not adaptive_depth and k <= 1 and len(split_subset) > 1:
                        k = min(2, factor, len(split_subset))
                        print(
                            f"  {split_context}: fixed depth forces fallback k={k}",
                            flush=True,
                        )

                if method == "recursive_spectral":
                    local = spectral_cluster_subset(
                        A=A,
                        A_embedding=A_embedding,
                        fusion_alpha=args.spectral_fusion_alpha,
                        subset=split_subset,
                        k=k,
                        seed=rng_seed + level * 1009 + int(parent),
                    )
                    local_history = []
                elif method == "recursive_kmeans":
                    if adaptive_branch:
                        cluster_popularity = parent_popularity.get(parent)
                        silhouette_threshold = popularity_quality_threshold(
                            cluster_popularity,
                            args,
                        )
                        local, local_history = adaptive_kmeans_split(
                            graph_embeddings=graph_embeddings,
                            subset=split_subset,
                            max_branches=factor,
                            min_silhouette=silhouette_threshold,
                            min_child_size=args.recursive_kmeans_min_child_size,
                            silhouette_sample_size=args.recursive_kmeans_silhouette_sample_size,
                            seed=rng_seed + level * 1009 + int(parent),
                            context=split_context,
                            cluster_popularity=cluster_popularity,
                            require_quality=cluster_adaptive_depth,
                        )
                    else:
                        local = embedding_cluster_subset(
                            graph_embeddings=graph_embeddings,
                            subset=split_subset,
                            k=k,
                            seed=rng_seed + level * 1009 + int(parent),
                        )
                        local_history = []
                elif method == "recursive_dmon":
                    if adaptive_branch:
                        local, local_history = adaptive_dmon_split(
                            x_all=x,
                            A=A,
                            subset=split_subset,
                            max_branches=factor,
                            args=args,
                            device=device,
                            seed=rng_seed + level * 1009 + int(parent),
                            context=split_context,
                            cluster_popularity=parent_popularity.get(parent),
                            require_quality=cluster_adaptive_depth,
                            item_frequencies=item_frequencies,
                            transitions=transitions,
                        )
                    else:
                        fixed_pairs = local_transition_pairs(
                            transitions=transitions,
                            subset=split_subset,
                            num_nodes=n,
                            min_cluster_size=args.dmon_prediction_min_cluster_size,
                            min_pairs=args.dmon_prediction_min_pairs,
                            max_pairs=args.dmon_prediction_max_pairs,
                            seed=rng_seed + level * 1009 + int(parent),
                        )
                        local, local_history, fixed_prediction_metrics = neural_cluster_subset(
                            method=method,
                            x_all=x,
                            A=A,
                            subset=split_subset,
                            k=k,
                            args=args,
                            device=device,
                            seed=rng_seed + level * 1009 + int(parent),
                            item_frequencies=item_frequencies,
                            transition_pairs=fixed_pairs,
                        )
                else:
                    raise ValueError(f"Unsupported recursive method: {method}")

                if cluster_adaptive_depth and not adaptive_branch and method == "recursive_kmeans" and k > 1:
                    child_sizes = np.bincount(local, minlength=k)
                    proposed_children = int(np.count_nonzero(child_sizes))
                    smallest_child = int(child_sizes.min())
                    valid = (
                        proposed_children == k
                        and smallest_child >= args.recursive_kmeans_min_child_size
                    )
                    silhouette = -1.0
                    if valid:
                        try:
                            silhouette = sampled_silhouette_score(
                                features=graph_embeddings[split_subset],
                                labels=local,
                                sample_size=args.recursive_kmeans_silhouette_sample_size,
                                seed=rng_seed + level * 1009 + int(parent),
                            )
                        except ValueError as error:
                            print(
                                f"  {split_context}: silhouette failed: {error}",
                                flush=True,
                            )
                            valid = False
                    cluster_popularity = parent_popularity.get(parent)
                    normalized_popularity = (
                        0.0 if cluster_popularity is None else cluster_popularity
                    )
                    silhouette_threshold = popularity_quality_threshold(
                        cluster_popularity,
                        args,
                    )
                    split_accepted = valid and silhouette > silhouette_threshold
                    print(
                        f"  {split_context}: fixed k={k}, children={proposed_children}, "
                        f"min_child_size={smallest_child}, silhouette={silhouette:.6g}, "
                        f"popularity={normalized_popularity:.6g}, "
                        f"threshold={silhouette_threshold:.6g}, "
                        f"split={'accepted' if split_accepted else 'rejected'}",
                        flush=True,
                    )
                    local_history.append(
                        {
                            "fixed_k": float(k),
                            "proposed_children": float(proposed_children),
                            "selected_silhouette": float(silhouette),
                            "cluster_popularity": float(normalized_popularity),
                            "silhouette_threshold": float(silhouette_threshold),
                            "split_accepted": float(split_accepted),
                        }
                    )
                    if not split_accepted:
                        local = np.zeros(len(split_subset), dtype=np.int64)

                if (
                    cluster_adaptive_depth
                    and not adaptive_branch
                    and method == "recursive_dmon"
                    and k > 1
                ):
                    proposed_children = int(len(np.unique(local)))
                    modularity_gain = local_weighted_modularity_gain(
                        A, split_subset, local
                    )
                    prediction_gain = (
                        None
                        if fixed_prediction_metrics is None
                        else fixed_prediction_metrics["prediction_normalized_gain"]
                    )
                    cluster_popularity = parent_popularity.get(parent)
                    normalized_popularity = (
                        0.0 if cluster_popularity is None else cluster_popularity
                    )
                    modularity_threshold = popularity_quality_threshold(
                        cluster_popularity,
                        args,
                    )
                    quality_score, quality_threshold = dmon_quality_score(
                        modularity_gain=modularity_gain,
                        prediction_normalized_gain=prediction_gain,
                        modularity_threshold=modularity_threshold,
                        args=args,
                    )
                    split_accepted = (
                        proposed_children > 1
                        and quality_score > quality_threshold
                    )
                    print(
                        f"  {split_context}: fixed k={k}, children={proposed_children}, "
                        f"local_modularity_gain={modularity_gain:.6g}, "
                        f"prediction_gain={prediction_gain if prediction_gain is not None else 'unavailable'}, "
                        f"quality_score={quality_score:.6g}, "
                        f"popularity={normalized_popularity:.6g}, "
                        f"threshold={quality_threshold:.6g}, "
                        f"split={'accepted' if split_accepted else 'rejected'}",
                        flush=True,
                    )
                    local_history.append(
                        {
                            "fixed_k": float(k),
                            "proposed_children": float(proposed_children),
                            "modularity_gain": float(modularity_gain),
                            "prediction_normalized_gain": float(
                                prediction_gain
                                if prediction_gain is not None
                                else 0.0
                            ),
                            "prediction_available": float(prediction_gain is not None),
                            "quality_score": float(quality_score),
                            "cluster_popularity": float(normalized_popularity),
                            "modularity_threshold": float(modularity_threshold),
                            "quality_threshold": float(quality_threshold),
                            "split_accepted": float(split_accepted),
                        }
                    )
                    if not split_accepted:
                        local = np.zeros(len(split_subset), dtype=np.int64)

                if cluster_adaptive_depth and method == "recursive_spectral" and k > 1:
                    proposed_children = int(len(np.unique(local)))
                    modularity_gain = local_weighted_modularity_gain(A, split_subset, local)
                    cluster_popularity = parent_popularity.get(parent)
                    normalized_popularity = (
                        0.0 if cluster_popularity is None else cluster_popularity
                    )
                    modularity_threshold = popularity_quality_threshold(
                        cluster_popularity,
                        args,
                    )
                    split_accepted = (
                        proposed_children > 1
                        and modularity_gain > modularity_threshold
                    )
                    print(
                        f"  {split_context}: proposed_children={proposed_children}, "
                        f"local_modularity_gain={modularity_gain:.6g}, "
                        f"popularity={normalized_popularity:.6g}, "
                        f"threshold={modularity_threshold:.6g}, "
                        f"split={'accepted' if split_accepted else 'rejected'}",
                        flush=True,
                    )
                    local_history.append(
                        {
                            "proposed_k": float(k),
                            "proposed_children": float(proposed_children),
                            "modularity_gain": float(modularity_gain),
                            "cluster_popularity": float(normalized_popularity),
                            "modularity_threshold": float(modularity_threshold),
                            "split_accepted": float(split_accepted),
                        }
                    )
                    if not split_accepted:
                        local = np.zeros(len(split_subset), dtype=np.int64)

            local_ids = sorted(np.unique(local).tolist())
            parent_split = len(local_ids) > 1 or (
                len(stopped_subset) > 0 and len(split_subset) > 0
            )
            any_split = any_split or parent_split

            if len(stopped_subset) > 0:
                new_labels[stopped_subset] = next_cluster_id
                next_cluster_id += 1

            for local_id in local_ids:
                mask = local == local_id
                child_items = split_subset[mask]
                new_labels[child_items] = next_cluster_id
                if item_adaptive_depth:
                    if np.any(target_hierarchy_depths[child_items] > level):
                        next_active_parents.add(next_cluster_id)
                elif parent_split and int(mask.sum()) > 1:
                    next_active_parents.add(next_cluster_id)
                next_cluster_id += 1

            for row in local_history:
                row = dict(row)
                row["level"] = float(level)
                row["parent"] = float(parent)
                history.append(row)

        new_labels = relabel_compact(new_labels)
        if cluster_adaptive_depth and not any_split:
            if not hierarchy:
                hierarchy.append(new_labels)
                print(f"{method} stopped at root: clusters=1")
            break
        hierarchy.append(new_labels)
        print(f"recursive level {level}: clusters={len(np.unique(new_labels[new_labels >= 0]))}")
        parent_labels = new_labels
        active_parents = next_active_parents
        level += 1

    return hierarchy, history


def recursive_leiden_hierarchy(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    item_log_frequencies: np.ndarray | None = None,
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
        parent_popularity = normalize_active_parent_popularity(
            parent_labels,
            active_parents,
            item_log_frequencies,
        )

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
                proposed_children = int(len(np.unique(local)))
                if proposed_children > 1:
                    modularity_gain = local_weighted_modularity_gain(A, subset, local)
                    cluster_popularity = parent_popularity.get(parent)
                    normalized_popularity = (
                        0.0 if cluster_popularity is None else cluster_popularity
                    )
                    modularity_threshold = popularity_quality_threshold(
                        cluster_popularity,
                        args,
                    )
                    split_accepted = modularity_gain > modularity_threshold
                    print(
                        f"Leiden level {level}, parent {parent}: "
                        f"children={proposed_children}, "
                        f"local_modularity_gain={modularity_gain:.6g}, "
                        f"popularity={normalized_popularity:.6g}, "
                        f"threshold={modularity_threshold:.6g}, "
                        f"split={'accepted' if split_accepted else 'rejected'}",
                        flush=True,
                    )
                    local_history.append(
                        {
                            "proposed_children": float(proposed_children),
                            "modularity_gain": float(modularity_gain),
                            "cluster_popularity": float(normalized_popularity),
                            "modularity_threshold": float(modularity_threshold),
                            "split_accepted": float(split_accepted),
                        }
                    )
                    if not split_accepted:
                        local = np.zeros(len(subset), dtype=np.int64)
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
    parser.add_argument(
        "--recursive_branch_mode",
        choices=["adaptive", "fixed"],
        default="adaptive",
        help="Select the child count adaptively or use --recursive_factor.",
    )
    parser.add_argument(
        "--recursive_depth_mode",
        choices=["adaptive", "item", "fixed"],
        default="adaptive",
        help=(
            "Stop clusters by split quality, stop each item at its popularity-derived "
            "target SID length, or run exactly --recursive_levels levels."
        ),
    )
    parser.add_argument("--recursive_factor", type=int, default=4, help="Maximum child clusters per parent for recursive methods.")
    parser.add_argument(
        "--recursive_levels",
        type=int,
        default=2,
        help="Exact hierarchy depth in fixed mode; ignored by adaptive and item modes.",
    )
    parser.add_argument("--recursive_epochs", type=int, default=100)
    parser.add_argument(
        "--recursive_item_depth_function",
        choices=["sigmoid", "linear"],
        default="sigmoid",
        help="Popularity-to-hierarchy-depth function used when --recursive_depth_mode=item.",
    )
    parser.add_argument(
        "--recursive_item_depth_min",
        type=int,
        default=2,
        help="Minimum number of hierarchy levels, excluding the unique-leaf token.",
    )
    parser.add_argument(
        "--recursive_item_depth_max",
        type=int,
        default=8,
        help="Maximum number of hierarchy levels, excluding the unique-leaf token.",
    )
    parser.add_argument("--recursive_item_depth_alpha", type=float, default=1.0)
    parser.add_argument(
        "--recursive_item_depth_beta",
        type=float,
        default=0.5,
        help="Sigmoid midpoint in normalized dataset popularity space [0, 1].",
    )
    parser.add_argument(
        "--recursive_item_depth_gamma",
        type=float,
        default=1.0,
        help="Positive exponent controlling curvature of the normalized linear mapping.",
    )
    parser.add_argument(
        "--dmon_selection_epochs",
        type=int,
        default=50,
        help="Short training budget for each adaptive DMoN k candidate before retraining the winner.",
    )
    parser.add_argument(
        "--dmon_prediction_lambda",
        type=float,
        default=0.0,
        help="Weight of held-in soft next-item assignment prediction loss; zero disables it.",
    )
    parser.add_argument(
        "--dmon_prediction_holdout_fraction",
        type=float,
        default=0.2,
        help="Fraction of training transitions reserved for split-quality scoring.",
    )
    parser.add_argument(
        "--dmon_prediction_min_cluster_size",
        type=int,
        default=20,
        help="Minimum parent size on which to train the sequential surrogate.",
    )
    parser.add_argument(
        "--dmon_prediction_min_pairs",
        type=int,
        default=20,
        help="Minimum held-in local transitions required to train the surrogate.",
    )
    parser.add_argument(
        "--dmon_prediction_max_pairs",
        type=int,
        default=50000,
        help="Maximum held-in and held-out local transitions per parent; zero disables sampling.",
    )
    parser.add_argument(
        "--dmon_modularity_score_weight",
        type=float,
        default=0.5,
        help="Beta in beta*modularity_gain + (1-beta)*normalized_prediction_gain.",
    )
    parser.add_argument(
        "--dmon_prediction_gain_threshold",
        type=float,
        default=0.01,
        help="Minimum normalized held-out prediction gain used in combined adaptive stopping.",
    )
    parser.add_argument(
        "--recursive_min_modularity_gain",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option; use --recursive_popularity_epsilon_base."
        ),
    )
    parser.add_argument(
        "--recursive_popularity_epsilon_base",
        type=float,
        default=0.001,
        help="Base quality threshold for popularity-aware recursive stopping.",
    )
    parser.add_argument(
        "--recursive_popularity_alpha",
        type=float,
        default=0.0,
        help=(
            "Popularity coefficient in threshold = epsilon_base + alpha * per-level "
            "min-max normalized parent mean log-frequency."
        ),
    )
    parser.add_argument(
        "--recursive_kmeans_min_silhouette",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; use --recursive_popularity_epsilon_base.",
    )
    parser.add_argument(
        "--recursive_kmeans_min_child_size",
        type=int,
        default=2,
        help="Minimum number of items required in every adaptive K-means child.",
    )
    parser.add_argument(
        "--recursive_kmeans_silhouette_sample_size",
        type=int,
        default=2000,
        help="Maximum items used for silhouette scoring per branch; zero scores the full branch.",
    )
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
        choices=["semantic", "appnp", "fastrp", "s3gc", "magi", "ns4gc"],
        default="semantic",
        help="Embedding source used by recursive_kmeans; semantic uses the input item embeddings directly.",
    )
    parser.add_argument(
        "--appnp_num_prop",
        type=int,
        default=1,
        help="Number of APPNP propagation steps used to prepare recursive K-means embeddings.",
    )
    parser.add_argument(
        "--appnp_alpha",
        type=float,
        default=0.5,
        help="APPNP teleport probability used to prepare recursive K-means embeddings.",
    )
    parser.add_argument("--fastrp_dim", type=int, default=768)
    parser.add_argument("--fastrp_feature_weight", type=float, default=1.0)
    parser.add_argument(
        "--fastrp_iteration_weights",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 1.0],
        help="FastRP hop weights, including the initial projection at index zero.",
    )
    parser.add_argument("--fastrp_seed", type=int, default=42)
    parser.add_argument("--s3gc_dim", type=int, default=768)
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
    parser.add_argument("--magi_dim", type=int, default=768)
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
    parser.add_argument("--ns4gc_dim", type=int, default=768)
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
    parser.add_argument(
        "--dmon_popularity_balance_lambda",
        type=float,
        default=0.0,
        help=(
            "Popularity-mass strength. In balance mode, blend equal popularity mass "
            "into the balance regularizer. In concentrate mode, weight the "
            "popularity-mass entropy loss. In separate mode, weight negative variance "
            "of child mean popularity. In purity mode, weight conditional popularity-bin "
            "entropy. Zero disables all popularity objectives."
        ),
    )
    parser.add_argument(
        "--dmon_popularity_mass_objective",
        choices=["balance", "concentrate", "separate", "purity"],
        default="balance",
        help=(
            "Whether recursive DMoN encourages equal popularity mass across children "
            "concentrates it by minimizing popularity-mass entropy, or separates "
            "children by maximizing variance in their mean popularity or minimizing "
            "popularity-bin entropy within children."
        ),
    )
    parser.add_argument(
        "--dmon_popularity_purity_bins",
        type=int,
        default=3,
        help="Number of global rank-quantile popularity bins used by the purity objective.",
    )

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
    if args.recursive_depth_mode == "item" and args.method not in {
        "recursive_spectral",
        "recursive_dmon",
        "recursive_kmeans",
    }:
        raise ValueError(
            "--recursive_depth_mode=item is supported by recursive_spectral, "
            "recursive_dmon, and recursive_kmeans"
        )
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
    item_log_frequencies = None
    item_frequencies = None
    target_hierarchy_depths_all = None
    target_sid_lengths = None
    transitions = None
    if args.recursive_popularity_alpha < 0.0:
        raise ValueError("--recursive_popularity_alpha must be >= 0")
    if args.recursive_popularity_epsilon_base < 0.0:
        raise ValueError("--recursive_popularity_epsilon_base must be >= 0")
    if not 0.0 <= args.dmon_popularity_balance_lambda <= 1.0:
        raise ValueError("--dmon_popularity_balance_lambda must be between 0 and 1")
    if args.dmon_popularity_purity_bins < 2:
        raise ValueError("--dmon_popularity_purity_bins must be at least 2")
    if args.dmon_prediction_lambda < 0.0:
        raise ValueError("--dmon_prediction_lambda must be >= 0")
    if not 0.0 < args.dmon_prediction_holdout_fraction < 1.0:
        raise ValueError("--dmon_prediction_holdout_fraction must be between 0 and 1")
    if args.dmon_prediction_min_cluster_size < 2:
        raise ValueError("--dmon_prediction_min_cluster_size must be >= 2")
    if args.dmon_prediction_min_pairs < 1:
        raise ValueError("--dmon_prediction_min_pairs must be >= 1")
    if args.dmon_prediction_max_pairs < 0:
        raise ValueError("--dmon_prediction_max_pairs must be >= 0")
    if not 0.0 <= args.dmon_modularity_score_weight <= 1.0:
        raise ValueError("--dmon_modularity_score_weight must be between 0 and 1")
    popularity_methods = {
        "recursive_dmon",
        "recursive_spectral",
        "recursive_kmeans",
        "recursive_leiden",
    }
    if args.method in popularity_methods:
        all_item_frequencies = load_item_frequencies(
            interactions_path=args.interactions_path,
            item_ids=np.arange(raw_embeddings.shape[0], dtype=np.int64),
            num_items=raw_embeddings.shape[0],
            skip_n_last=args.skip_n_last,
        )
        item_frequencies = all_item_frequencies[item_ids]
        item_log_frequencies = np.log1p(item_frequencies)
        args.dmon_popularity_bin_ids = rank_quantile_bins(
            item_frequencies, args.dmon_popularity_purity_bins
        )
        print(
            f"Popularity-aware {args.method} threshold: "
            f"epsilon_base={args.recursive_popularity_epsilon_base}, "
            f"alpha={args.recursive_popularity_alpha}, "
            "normalization=per_level_minmax_parent_mean_log_frequency"
        )
        if args.recursive_depth_mode == "item":
            target_hierarchy_depths_all = item_target_hierarchy_depths(
                all_item_frequencies, args
            )
            target_sid_lengths = target_hierarchy_depths_all + 1
            target_counts = {
                int(depth): int(np.count_nonzero(target_hierarchy_depths_all == depth))
                for depth in np.unique(target_hierarchy_depths_all)
            }
            print(
                "Item-level target hierarchy depths: "
                f"function={args.recursive_item_depth_function}, "
                f"min={args.recursive_item_depth_min}, "
                f"max={args.recursive_item_depth_max}, "
                "normalization=dataset_minmax_log1p_training_frequency, "
                f"counts={target_counts}"
            )
    if args.method == "recursive_dmon" and args.dmon_prediction_lambda > 0.0:
        transitions = load_transition_pairs(
            interactions_path=args.interactions_path,
            item_ids=item_ids,
            num_items=raw_embeddings.shape[0],
            skip_n_last=args.skip_n_last,
            holdout_fraction=args.dmon_prediction_holdout_fraction,
            seed=args.seed,
        )
        print(
            "Loaded sequential surrogate pairs: "
            f"train={len(transitions.train.sources)}, "
            f"eval={len(transitions.evaluation.sources)}, "
            f"holdout={args.dmon_prediction_holdout_fraction}"
        )

    print(f"Method: {args.method}")
    if args.method in {"recursive_spectral", "recursive_dmon", "recursive_kmeans"}:
        print(
            f"Recursive modes: branch={args.recursive_branch_mode}, "
            f"depth={args.recursive_depth_mode}, factor={args.recursive_factor}, "
            f"levels={args.recursive_levels}"
        )
    if args.method == "recursive_dmon":
        print(
            "DMoN popularity-mass objective: "
            f"{args.dmon_popularity_mass_objective}, "
            f"lambda={args.dmon_popularity_balance_lambda}"
        )
        print(
            "DMoN next-item surrogate: "
            f"lambda={args.dmon_prediction_lambda}, "
            f"modularity_weight={args.dmon_modularity_score_weight}, "
            f"prediction_threshold={args.dmon_prediction_gain_threshold}"
        )
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
            item_log_frequencies=item_log_frequencies,
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
            item_log_frequencies=item_log_frequencies,
            item_frequencies=item_frequencies,
            target_hierarchy_depths=(
                target_hierarchy_depths_all[item_ids]
                if target_hierarchy_depths_all is not None
                else None
            ),
            required_hierarchy_depth=(
                int(target_hierarchy_depths_all.max())
                if target_hierarchy_depths_all is not None
                else None
            ),
            transitions=transitions,
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
        target_sid_lengths=target_sid_lengths,
    )


if __name__ == "__main__":
    main()
