"""PyAGC S3GC graph embedding preprocessing."""

from __future__ import annotations

import numpy as np
import torch


def build_s3gc_embeddings(
    edge_index: torch.Tensor,
    node_features: torch.Tensor | np.ndarray,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    walk_length: int,
    context_size: int,
    walks_per_node: int,
    num_negative_samples: int,
    diffusion_hops: int,
    ppr_alpha: float,
    scale: str,
    seed: int,
    device: torch.device,
    print_every: int,
) -> np.ndarray:
    """Train PyAGC S3GC and return graph-aware node embeddings."""
    try:
        from pyagc.models.s3gc import CompositeOptimizer, S3GC, precompute_features
    except ImportError as exc:
        raise ImportError(
            "S3GC graph embeddings require pyagc==1.1.2 and pyg-lib>=0.6.0. "
            "Install the dependencies from requirements.txt and README.md."
        ) from exc

    if isinstance(node_features, torch.Tensor):
        features = node_features.detach().cpu().float()
    else:
        features = torch.as_tensor(node_features, dtype=torch.float32)
    edge_index = edge_index.detach().cpu().long()
    num_nodes = int(features.size(0))

    if features.ndim != 2:
        raise ValueError(f"S3GC node_features must be 2D, got shape {tuple(features.shape)}")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"S3GC edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.numel() > 0 and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError("S3GC edge_index contains a node outside node_features")
    if not torch.isfinite(features).all():
        raise ValueError("S3GC node_features contain non-finite values")
    if embedding_dim < 1:
        raise ValueError("S3GC embedding_dim must be >= 1")
    if epochs < 1:
        raise ValueError("S3GC epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("S3GC batch_size must be >= 1")
    if learning_rate <= 0.0 or not np.isfinite(learning_rate):
        raise ValueError("S3GC learning_rate must be finite and > 0")
    if weight_decay < 0.0 or not np.isfinite(weight_decay):
        raise ValueError("S3GC weight_decay must be finite and >= 0")
    if walk_length < context_size or context_size < 2:
        raise ValueError("S3GC requires walk_length >= context_size >= 2")
    if walks_per_node < 1 or num_negative_samples < 1:
        raise ValueError("S3GC walk and negative sample counts must be >= 1")
    if diffusion_hops < 0:
        raise ValueError("S3GC diffusion_hops must be >= 0")
    if not 0.0 < ppr_alpha <= 1.0:
        raise ValueError("S3GC ppr_alpha must be in (0, 1]")
    if scale not in {"auto", "small", "medium", "large"}:
        raise ValueError(f"Unknown S3GC scale: {scale!r}")
    if print_every < 1:
        raise ValueError("S3GC print_every must be >= 1")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # PyAGC's S3GC preprocessing and random walks currently consume only
    # connectivity; edge weights are not part of its public API.
    ax, sx = precompute_features(
        features,
        edge_index,
        num_nodes,
        method="ppr",
        k_hop=diffusion_hops,
        alpha=ppr_alpha,
        verbose=False,
    )

    model = S3GC(
        edge_index=edge_index,
        num_nodes=num_nodes,
        in_channels=int(features.size(1)),
        hidden_channels=int(embedding_dim),
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        scale=scale,
    ).to(device)
    model.set_precomputed_features(ax, sx)
    loader = model.loader(batch_size=min(batch_size, num_nodes), shuffle=True)

    if model.scale == "large":
        optimizer = CompositeOptimizer(
            dense=torch.optim.Adam(
                model.encoder.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            ),
            sparse=torch.optim.SparseAdam(
                model.iden_embedding.parameters(),
                lr=learning_rate,
            ),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    for epoch in range(1, epochs + 1):
        loss = model.train_epoch(loader, optimizer, epoch, verbose=False)
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"S3GC epoch={epoch:04d} loss={loss:.6f}")

    model.eval()
    output_batches = []
    with torch.no_grad():
        for start in range(0, num_nodes, batch_size):
            indices = torch.arange(start, min(start + batch_size, num_nodes), device=device)
            output_batches.append(model.embed(indices).cpu())
    embeddings = torch.cat(output_batches, dim=0).numpy().astype(np.float32, copy=False)

    expected_shape = (num_nodes, int(embedding_dim))
    if embeddings.shape != expected_shape:
        raise ValueError(f"S3GC returned shape {embeddings.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("S3GC returned non-finite embeddings")
    return embeddings
