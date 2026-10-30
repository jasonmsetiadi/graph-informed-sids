#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=graph_embedding_config.sh
source "${SCRIPT_DIR}/graph_embedding_config.sh"

GRAPH_EMBEDDING_METHOD="${1:-${GRAPH_EMBEDDING_METHOD:-fastrp}}"
METHOD_NAME="${METHOD_NAME:-RecKMeans_t5}"

case "${GRAPH_EMBEDDING_METHOD}" in
  fastrp|s3gc|magi|ns4gc) ;;
  *)
    echo "Unsupported graph embedding method: ${GRAPH_EMBEDDING_METHOD}" >&2
    echo "Choose fastrp, s3gc, magi, or ns4gc." >&2
    exit 2
    ;;
esac

RECURSIVE_FACTOR=6
RECURSIVE_LEVELS=4
configure_graph_embedding_args "${GRAPH_EMBEDDING_METHOD}"

RECKMEANS_CLUSTERING_ARGS=(
  --recursive_factor "${RECURSIVE_FACTOR}"
  --recursive_levels "${RECURSIVE_LEVELS}"
  --graph_embedding_method "${GRAPH_EMBEDDING_METHOD}"
  "${GRAPH_EMBEDDING_ARGS[@]}"
)

RECKMEANS_CONFIGURED=1

# Reuse the common dataset, SID generation, mapping, training, and evaluation
# pipeline without keeping K-means backend configuration in run_recdmon.sh.
# shellcheck source=run_recdmon.sh
source "${SCRIPT_DIR}/run_recdmon.sh"
