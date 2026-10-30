#!/usr/bin/env bash
set -euo pipefail


GRAPH_EMBEDDING_METHOD="${1:-${GRAPH_EMBEDDING_METHOD:-fastrp}}"
METHOD_NAME="${METHOD_NAME:-RecKMeans_t5}"

case "${GRAPH_EMBEDDING_METHOD}" in
  fastrp|s3gc) ;;
  *)
    echo "Unsupported graph embedding method: ${GRAPH_EMBEDDING_METHOD}" >&2
    echo "Choose fastrp or s3gc." >&2
    exit 2
    ;;
esac

RECURSIVE_FACTOR=6
RECURSIVE_LEVELS=4

FASTRP_DIM=256
FASTRP_FEATURE_WEIGHT=1.0
FASTRP_ITERATION_WEIGHTS=(0.0 1.0 1.0)
FASTRP_SEED=42

S3GC_DIM=256
S3GC_EPOCHS=200
S3GC_BATCH_SIZE=4096
S3GC_LR=0.01
S3GC_WEIGHT_DECAY=0.0
S3GC_WALK_LENGTH=3
S3GC_CONTEXT_SIZE=3
S3GC_WALKS_PER_NODE=1
S3GC_NUM_NEGATIVE_SAMPLES=1
S3GC_DIFFUSION_HOPS=2
S3GC_PPR_ALPHA=0.2
S3GC_SCALE=auto
S3GC_SEED=42
S3GC_PRINT_EVERY=10

RECKMEANS_CLUSTERING_ARGS=(
  --recursive_factor "${RECURSIVE_FACTOR}"
  --recursive_levels "${RECURSIVE_LEVELS}"
  --graph_embedding_method "${GRAPH_EMBEDDING_METHOD}"
)

if [[ "${GRAPH_EMBEDDING_METHOD}" == "fastrp" ]]; then
  RECKMEANS_CLUSTERING_ARGS+=(
    --fastrp_dim "${FASTRP_DIM}"
    --fastrp_feature_weight "${FASTRP_FEATURE_WEIGHT}"
    --fastrp_iteration_weights "${FASTRP_ITERATION_WEIGHTS[@]}"
    --fastrp_seed "${FASTRP_SEED}"
  )
else
  RECKMEANS_CLUSTERING_ARGS+=(
    --s3gc_dim "${S3GC_DIM}"
    --s3gc_epochs "${S3GC_EPOCHS}"
    --s3gc_batch_size "${S3GC_BATCH_SIZE}"
    --s3gc_lr "${S3GC_LR}"
    --s3gc_weight_decay "${S3GC_WEIGHT_DECAY}"
    --s3gc_walk_length "${S3GC_WALK_LENGTH}"
    --s3gc_context_size "${S3GC_CONTEXT_SIZE}"
    --s3gc_walks_per_node "${S3GC_WALKS_PER_NODE}"
    --s3gc_num_negative_samples "${S3GC_NUM_NEGATIVE_SAMPLES}"
    --s3gc_diffusion_hops "${S3GC_DIFFUSION_HOPS}"
    --s3gc_ppr_alpha "${S3GC_PPR_ALPHA}"
    --s3gc_scale "${S3GC_SCALE}"
    --s3gc_seed "${S3GC_SEED}"
    --s3gc_print_every "${S3GC_PRINT_EVERY}"
  )
fi

RECKMEANS_CONFIGURED=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the common dataset, SID generation, mapping, training, and evaluation
# pipeline without keeping K-means backend configuration in run_recdmon.sh.
# shellcheck source=run_recdmon.sh
source "${SCRIPT_DIR}/run_recdmon.sh"
