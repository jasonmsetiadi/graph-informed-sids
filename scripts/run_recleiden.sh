#!/usr/bin/env bash
set -euo pipefail


LEIDEN_EMBEDDING_METHOD="${1:-${LEIDEN_EMBEDDING_METHOD:-semantic}}"
METHOD_NAME="${METHOD_NAME:-RecLeiden_t5}"

case "${LEIDEN_EMBEDDING_METHOD}" in
  semantic|fastrp|s3gc) ;;
  *)
    echo "Unsupported Leiden embedding method: ${LEIDEN_EMBEDDING_METHOD}" >&2
    echo "Choose semantic, fastrp, or s3gc." >&2
    exit 2
    ;;
esac

if [[ "${LEIDEN_EMBEDDING_METHOD}" == "semantic" ]]; then
  DEFAULT_LEIDEN_FUSION_ALPHA=0.5
else
  DEFAULT_LEIDEN_FUSION_ALPHA=1.0
fi
LEIDEN_FUSION_ALPHA="${LEIDEN_FUSION_ALPHA:-${DEFAULT_LEIDEN_FUSION_ALPHA}}"
LEIDEN_KNN="${LEIDEN_KNN:-20}"
LEIDEN_RESOLUTION_MIN=0.01
LEIDEN_RESOLUTION_MAX=100.0
LEIDEN_RESOLUTION_POINTS=41
LEIDEN_PLATEAU_TOLERANCE=0.05
LEIDEN_EARLY_STOP_PATIENCE=10
LEIDEN_ITERATIONS=-1

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

RECLEIDEN_CLUSTERING_ARGS=(
  --leiden_embedding_method "${LEIDEN_EMBEDDING_METHOD}"
  --leiden_fusion_alpha "${LEIDEN_FUSION_ALPHA}"
  --leiden_knn "${LEIDEN_KNN}"
  --leiden_resolution_min "${LEIDEN_RESOLUTION_MIN}"
  --leiden_resolution_max "${LEIDEN_RESOLUTION_MAX}"
  --leiden_resolution_points "${LEIDEN_RESOLUTION_POINTS}"
  --leiden_plateau_tolerance "${LEIDEN_PLATEAU_TOLERANCE}"
  --leiden_early_stop_patience "${LEIDEN_EARLY_STOP_PATIENCE}"
  --leiden_iterations "${LEIDEN_ITERATIONS}"
)

if [[ "${LEIDEN_EMBEDDING_METHOD}" == "fastrp" ]]; then
  RECLEIDEN_CLUSTERING_ARGS+=(
    --fastrp_dim "${FASTRP_DIM}"
    --fastrp_feature_weight "${FASTRP_FEATURE_WEIGHT}"
    --fastrp_iteration_weights "${FASTRP_ITERATION_WEIGHTS[@]}"
    --fastrp_seed "${FASTRP_SEED}"
  )
elif [[ "${LEIDEN_EMBEDDING_METHOD}" == "s3gc" ]]; then
  RECLEIDEN_CLUSTERING_ARGS+=(
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

RECLEIDEN_CONFIGURED=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the common dataset, SID generation, mapping, training, and evaluation
# pipeline without keeping Leiden configuration in run_recdmon.sh.
# shellcheck source=run_recdmon.sh
source "${SCRIPT_DIR}/run_recdmon.sh"
