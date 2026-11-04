#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=graph_embedding_config.sh
source "${SCRIPT_DIR}/graph_embedding_config.sh"

GRAPH_EMBEDDING_METHOD="${1:-${GRAPH_EMBEDDING_METHOD:-semantic}}"
METHOD_NAME="${METHOD_NAME:-RecKMeans_t5}"

case "${GRAPH_EMBEDDING_METHOD}" in
  semantic|fastrp|s3gc|magi|ns4gc) ;;
  *)
    echo "Unsupported graph embedding method: ${GRAPH_EMBEDDING_METHOD}" >&2
    echo "Choose semantic, fastrp, s3gc, magi, or ns4gc." >&2
    exit 2
    ;;
esac

RECURSIVE_KMEANS_MIN_SILHOUETTE="${RECURSIVE_KMEANS_MIN_SILHOUETTE:-0.0}"
RECURSIVE_KMEANS_MIN_CHILD_SIZE="${RECURSIVE_KMEANS_MIN_CHILD_SIZE:-2}"
RECURSIVE_KMEANS_SILHOUETTE_SAMPLE_SIZE="${RECURSIVE_KMEANS_SILHOUETTE_SAMPLE_SIZE:-2000}"
GRAPH_EMBEDDING_ARGS=()
if [[ "${GRAPH_EMBEDDING_METHOD}" != "semantic" ]]; then
  configure_graph_embedding_args "${GRAPH_EMBEDDING_METHOD}"
fi

RECKMEANS_CLUSTERING_ARGS=(
  --recursive_kmeans_min_silhouette "${RECURSIVE_KMEANS_MIN_SILHOUETTE}"
  --recursive_kmeans_min_child_size "${RECURSIVE_KMEANS_MIN_CHILD_SIZE}"
  --recursive_kmeans_silhouette_sample_size "${RECURSIVE_KMEANS_SILHOUETTE_SAMPLE_SIZE}"
  --graph_embedding_method "${GRAPH_EMBEDDING_METHOD}"
  "${GRAPH_EMBEDDING_ARGS[@]}"
)

RECKMEANS_CONFIGURED=1

# Reuse the common dataset, SID generation, mapping, training, and evaluation
# pipeline without keeping K-means backend configuration in run_recdmon.sh.
# shellcheck source=run_recdmon.sh
source "${SCRIPT_DIR}/run_recdmon.sh"
