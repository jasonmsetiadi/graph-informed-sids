#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=graph_embedding_config.sh
source "${SCRIPT_DIR}/graph_embedding_config.sh"

LEIDEN_EMBEDDING_METHOD="${1:-${LEIDEN_EMBEDDING_METHOD:-semantic}}"
METHOD_NAME="${METHOD_NAME:-RecLeiden_t5}"

case "${LEIDEN_EMBEDDING_METHOD}" in
  semantic|fastrp|s3gc|magi|ns4gc) ;;
  *)
    echo "Unsupported Leiden embedding method: ${LEIDEN_EMBEDDING_METHOD}" >&2
    echo "Choose semantic, fastrp, s3gc, magi, or ns4gc." >&2
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
LEIDEN_RESOLUTION_MIN="${LEIDEN_RESOLUTION_MIN:-0.01}"
LEIDEN_RESOLUTION_MAX="${LEIDEN_RESOLUTION_MAX:-100.0}"
LEIDEN_RESOLUTION_POINTS="${LEIDEN_RESOLUTION_POINTS:-41}"
LEIDEN_PLATEAU_TOLERANCE="${LEIDEN_PLATEAU_TOLERANCE:-0.05}"
LEIDEN_EARLY_STOP_PATIENCE="${LEIDEN_EARLY_STOP_PATIENCE:-10}"
LEIDEN_ITERATIONS="${LEIDEN_ITERATIONS:--1}"

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
if [[ "${LEIDEN_EMBEDDING_METHOD}" != "semantic" ]]; then
  configure_graph_embedding_args "${LEIDEN_EMBEDDING_METHOD}"
  RECLEIDEN_CLUSTERING_ARGS+=("${GRAPH_EMBEDDING_ARGS[@]}")
fi

RECLEIDEN_CONFIGURED=1

# Reuse the common dataset, SID generation, mapping, training, and evaluation
# pipeline without keeping Leiden configuration in run_recdmon.sh.
# shellcheck source=run_recdmon.sh
source "${SCRIPT_DIR}/run_recdmon.sh"
