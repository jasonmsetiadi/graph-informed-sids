"""Locally weighted NS4GC graph embedding preprocessing."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


class _GraphAugmentation:
    """Device-aware feature masking and aligned edge/weight dropping."""

    def __init__(self, feature_mask: float, edge_drop: float):
        self.feature_mask = feature_mask
        self.edge_drop = edge_drop

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        **kwargs,
    ) -> dict:
        if self.feature_mask > 0.0:
            x = x.masked_fill(torch.rand_like(x) < self.feature_mask, 0.0)
        if self.edge_drop > 0.0 and edge_index.numel() > 0:
            keep = torch.rand(edge_index.size(1), device=edge_index.device) >= self.edge_drop
            edge_index = edge_index[:, keep]
            edge_weight = edge_weight[keep]
        return {
            "x": x,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        }


def weighted_ns4gc_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    similarity_threshold: float,
    temperature: float,
    neighbor_weight: float,
    sparsity_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute NS4GC with a raw-weighted, scale-invariant neighbor term."""
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    similarity = z1 @ z2.T
    num_nodes = z1.size(0)

    alignment_loss = -torch.diag(similarity).mean()
    src, dst = edge_index
    if edge_weight.numel() == 0:
        neighborhood_loss = similarity.new_zeros(())
    else:
        total_edge_weight = edge_weight.sum().clamp_min(1e-15)
        neighborhood_loss = -(edge_weight * similarity[src, dst]).sum() / total_edge_weight

    non_edge_mask = torch.ones(
        (num_nodes, num_nodes),
        device=similarity.device,
        dtype=torch.bool,
    )
    non_edge_mask.fill_diagonal_(False)
    non_edge_mask[src, dst] = False
    non_edge_similarity = similarity[non_edge_mask]
    if non_edge_similarity.numel() == 0:
        sparsity_loss = similarity.new_zeros(())
    else:
        sparsity_loss = torch.sigmoid(
            (non_edge_similarity - similarity_threshold) / temperature
        ).mean()

    total_loss = (
        alignment_loss
        + neighbor_weight * neighborhood_loss
        + sparsity_weight * sparsity_loss
    )
    return total_loss, alignment_loss, neighborhood_loss, sparsity_loss


def _prepare_edge_weights(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
) -> Tuple[torch.Tensor, bool]:
    """Validate raw graph weights and apply the DMoN-style unit fallback."""
    used_unit_fallback = edge_weight is None
    if edge_weight is None:
        weights = torch.ones(edge_index.size(1), dtype=torch.float32)
    else:
        weights = edge_weight.detach().cpu().float()

    if weights.ndim != 1 or weights.numel() != edge_index.size(1):
        raise ValueError(
            "NS4GC edge_weight must have one value per edge, got "
            f"shape={tuple(weights.shape)} and edges={edge_index.size(1)}"
        )
    if not torch.isfinite(weights).all():
        raise ValueError("NS4GC edge_weight contains non-finite values")
    if torch.any(weights < 0.0):
        raise ValueError("NS4GC edge_weight must be non-negative")
    if weights.numel() == 0 or not torch.any(weights > 0.0):
        raise ValueError("NS4GC requires at least one positive edge weight")
    return weights, used_unit_fallback


def build_ns4gc_embeddings(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    node_features: torch.Tensor | np.ndarray,
    embedding_dim: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_neighbors: int,
    inference_batch_size: int,
    inference_neighbors: int,
    feature_mask: float,
    edge_drop: float,
    similarity_threshold: float,
    temperature: float,
    neighbor_weight: float,
    sparsity_weight: float,
    training_mode: str,
    max_full_nodes: int,
    seed: int,
    device: torch.device,
    print_every: int,
) -> np.ndarray:
    """Train weighted NS4GC and return graph-aware node embeddings."""
    features = (
        node_features.detach().cpu().float()
        if isinstance(node_features, torch.Tensor)
        else torch.as_tensor(node_features, dtype=torch.float32)
    )
    edge_index = edge_index.detach().cpu().long()

    if features.ndim != 2:
        raise ValueError(f"NS4GC node_features must be 2D, got {tuple(features.shape)}")
    num_nodes = int(features.size(0))
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"NS4GC edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    weights, used_unit_fallback = _prepare_edge_weights(edge_index, edge_weight)
    if edge_index.numel() > 0 and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError("NS4GC edge_index contains a node outside node_features")
    if not torch.isfinite(features).all():
        raise ValueError("NS4GC node_features contain non-finite values")
    if embedding_dim < 1 or num_layers < 1 or epochs < 1:
        raise ValueError("NS4GC dimensions, layers, and epochs must be >= 1")
    if batch_size < 2 or inference_batch_size < 1:
        raise ValueError("NS4GC batch_size must be >= 2 and inference_batch_size >= 1")
    if num_neighbors < 1 or inference_neighbors < 1:
        raise ValueError("NS4GC neighbor counts must be >= 1")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("NS4GC learning_rate must be > 0 and weight_decay >= 0")
    if not 0.0 <= feature_mask <= 1.0 or not 0.0 <= edge_drop <= 1.0:
        raise ValueError("NS4GC feature_mask and edge_drop must be in [0, 1]")
    if temperature <= 0.0 or print_every < 1:
        raise ValueError("NS4GC temperature and print_every must be > 0")
    if training_mode not in {"sampled", "full"}:
        raise ValueError(f"Unknown NS4GC training_mode: {training_mode!r}")
    if max_full_nodes < 1:
        raise ValueError("NS4GC max_full_nodes must be >= 1")
    if training_mode == "full" and num_nodes > max_full_nodes:
        raise ValueError(
            f"Full-graph NS4GC received {num_nodes} nodes, above "
            f"--ns4gc_max_full_nodes={max_full_nodes}. Use sampled mode or raise the limit."
        )

    try:
        from pyagc.encoders import create_tuned_gnn
        from pyagc.models.base import LossOutput, TrainableModel
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
    except ImportError as exc:
        raise ImportError(
            "NS4GC graph embeddings require pyagc==1.1.2 and PyG sampling dependencies. "
            "Install the dependencies from requirements.txt and README.md."
        ) from exc

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    class WeightedNS4GC(TrainableModel):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.transform1 = _GraphAugmentation(feature_mask, edge_drop)
            self.transform2 = _GraphAugmentation(feature_mask, edge_drop)

        def embed(self, x, edge_index, edge_weight=None, **kwargs):
            return self.encoder(x, edge_index, edge_weight=edge_weight)

        def forward(self, x, edge_index, edge_weight, **kwargs):
            view1 = self.transform1(x, edge_index, edge_weight)
            view2 = self.transform2(x, edge_index, edge_weight)
            return self.encoder(**view1), self.encoder(**view2)

        def _loss(self, z1, z2, local_edge_index, local_edge_weight):
            total, alignment, neighborhood, sparsity = weighted_ns4gc_loss(
                z1=z1,
                z2=z2,
                edge_index=local_edge_index,
                edge_weight=local_edge_weight,
                similarity_threshold=similarity_threshold,
                temperature=temperature,
                neighbor_weight=neighbor_weight,
                sparsity_weight=sparsity_weight,
            )
            return LossOutput(
                total=total,
                components={
                    "ali": alignment.item(),
                    "nei": neighborhood.item(),
                    "spa": sparsity.item(),
                },
            )

        def loss(self, x, edge_index, edge_weight, **kwargs):
            z1, z2 = self(x, edge_index, edge_weight)
            return self._loss(z1, z2, edge_index, edge_weight)

        def loss_batch(self, batch):
            z1, z2 = self(batch.x, batch.edge_index, batch.edge_weight)
            batch_size_actual = int(batch.batch_size)
            z1 = z1[:batch_size_actual]
            z2 = z2[:batch_size_actual]
            edge_mask = (
                (batch.edge_index[0] < batch_size_actual)
                & (batch.edge_index[1] < batch_size_actual)
            )
            return self._loss(
                z1,
                z2,
                batch.edge_index[:, edge_mask],
                batch.edge_weight[edge_mask],
            )

    data = Data(x=features, edge_index=edge_index, edge_weight=weights)
    encoder = create_tuned_gnn(
        "gcn",
        in_channels=int(features.size(1)),
        hidden_channels=int(embedding_dim),
        out_channels=int(embedding_dim),
        num_layers=num_layers,
    )
    model = WeightedNS4GC(encoder=encoder).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if training_mode == "full":
        full_data = data.to(device)
        for epoch in range(1, epochs + 1):
            loss = model.train_full(full_data, optimizer, epoch, verbose=False)
            if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
                print(f"NS4GC[full] epoch={epoch:04d} loss={loss:.6f}")
        model.eval()
        with torch.no_grad():
            embeddings_tensor = model.embed(
                full_data.x,
                full_data.edge_index,
                full_data.edge_weight,
            )
    else:
        train_loader = NeighborLoader(
            data,
            num_neighbors=[num_neighbors] * num_layers,
            batch_size=min(batch_size, num_nodes),
            shuffle=True,
            num_workers=0,
        )
        for epoch in range(1, epochs + 1):
            loss = model.train_batch(train_loader, optimizer, epoch, verbose=False)
            if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
                print(f"NS4GC[sampled] epoch={epoch:04d} loss={loss:.6f}")

        inference_loader = NeighborLoader(
            data,
            num_neighbors=[inference_neighbors] * num_layers,
            batch_size=min(inference_batch_size, num_nodes),
            shuffle=False,
            num_workers=0,
        )
        embeddings_tensor = model.infer_batch(inference_loader, verbose=False)

    embeddings = embeddings_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    expected_shape = (num_nodes, int(embedding_dim))
    if embeddings.shape != expected_shape:
        raise ValueError(f"NS4GC returned shape {embeddings.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("NS4GC returned non-finite embeddings")

    fallback_label = "unit" if used_unit_fallback else "raw"
    print(f"NS4GC edge weights: source={fallback_label}, edges={weights.numel()}")
    return embeddings
