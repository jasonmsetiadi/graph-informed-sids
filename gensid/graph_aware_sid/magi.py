"""PyAGC MAGI graph embedding preprocessing."""

from __future__ import annotations

import numpy as np
import torch


def build_magi_embeddings(
    edge_index: torch.Tensor,
    node_features: torch.Tensor | np.ndarray,
    embedding_dim: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_neighbors: int,
    num_walks: int,
    walk_length: int,
    inference_batch_size: int,
    inference_neighbors: int,
    temperature: float,
    seed: int,
    device: torch.device,
    print_every: int,
) -> np.ndarray:
    """Train PyAGC MAGI with random-walk sampling and return node embeddings."""
    try:
        from pyagc.encoders import create_tuned_gnn
        from pyagc.models import MAGI
        from pyagc.models.magi import MAGINeighborLoader
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
    except ImportError as exc:
        raise ImportError(
            "MAGI graph embeddings require pyagc==1.1.2, torch-sparse, and PyG sampling dependencies. "
            "Install the dependencies from requirements.txt and README.md."
        ) from exc

    features = (
        node_features.detach().cpu().float()
        if isinstance(node_features, torch.Tensor)
        else torch.as_tensor(node_features, dtype=torch.float32)
    )
    edge_index = edge_index.detach().cpu().long()
    num_nodes = int(features.size(0))

    if features.ndim != 2:
        raise ValueError(f"MAGI node_features must be 2D, got {tuple(features.shape)}")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"MAGI edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError("MAGI node_features contain non-finite values")
    if embedding_dim < 1 or num_layers < 1 or epochs < 1:
        raise ValueError("MAGI dimensions, layers, and epochs must be >= 1")
    if batch_size < 2 or inference_batch_size < 1:
        raise ValueError("MAGI batch_size must be >= 2 and inference_batch_size >= 1")
    if min(num_neighbors, num_walks, walk_length, inference_neighbors) < 1:
        raise ValueError("MAGI neighbor and walk settings must be >= 1")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("MAGI learning_rate must be > 0 and weight_decay >= 0")
    if temperature <= 0.0 or print_every < 1:
        raise ValueError("MAGI temperature and print_every must be > 0")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = Data(x=features, edge_index=edge_index)
    encoder = create_tuned_gnn(
        "gcn",
        in_channels=int(features.size(1)),
        hidden_channels=int(embedding_dim),
        out_channels=int(embedding_dim),
        num_layers=num_layers,
    )
    model = MAGI(encoder=encoder, tau=temperature).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_loader = MAGINeighborLoader(
        data,
        num_neighbors=[num_neighbors] * num_layers,
        num_walks=num_walks,
        walk_length=walk_length,
        batch_size=min(batch_size, num_nodes),
        shuffle=True,
        num_workers=0,
    )

    for epoch in range(1, epochs + 1):
        loss = model.train_batch(train_loader, optimizer, epoch, verbose=False)
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"MAGI epoch={epoch:04d} loss={loss:.6f}")

    inference_loader = NeighborLoader(
        data,
        num_neighbors=[inference_neighbors] * num_layers,
        batch_size=min(inference_batch_size, num_nodes),
        shuffle=False,
        num_workers=0,
    )
    embeddings = model.infer_batch(inference_loader, verbose=False).detach().cpu().numpy()
    embeddings = embeddings.astype(np.float32, copy=False)

    expected_shape = (num_nodes, int(embedding_dim))
    if embeddings.shape != expected_shape:
        raise ValueError(f"MAGI returned shape {embeddings.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("MAGI returned non-finite embeddings")
    return embeddings
