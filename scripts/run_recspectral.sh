#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=graph_embedding_config.sh
source "${SCRIPT_DIR}/graph_embedding_config.sh"

SPECTRAL_EMBEDDING_METHOD="${1:-${SPECTRAL_EMBEDDING_METHOD:-semantic}}"
METHOD_NAME="${METHOD_NAME:-RecSpectral_t5}"

case "${SPECTRAL_EMBEDDING_METHOD}" in
  semantic|fastrp|s3gc|magi|ns4gc) ;;
  *)
    echo "Unsupported spectral embedding method: ${SPECTRAL_EMBEDDING_METHOD}" >&2
    echo "Choose semantic, fastrp, s3gc, magi, or ns4gc." >&2
    exit 2
    ;;
esac

RECURSIVE_FACTOR=6
RECURSIVE_LEVELS=4
if [[ "${SPECTRAL_EMBEDDING_METHOD}" == "semantic" ]]; then
  DEFAULT_SPECTRAL_FUSION_ALPHA=0.5
else
  DEFAULT_SPECTRAL_FUSION_ALPHA=1.0
fi
SPECTRAL_FUSION_ALPHA="${SPECTRAL_FUSION_ALPHA:-${DEFAULT_SPECTRAL_FUSION_ALPHA}}"
SPECTRAL_KNN="${SPECTRAL_KNN:-20}"

RECSPECTRAL_CLUSTERING_ARGS=(
  --recursive_factor "${RECURSIVE_FACTOR}"
  --recursive_levels "${RECURSIVE_LEVELS}"
  --spectral_embedding_method "${SPECTRAL_EMBEDDING_METHOD}"
  --spectral_fusion_alpha "${SPECTRAL_FUSION_ALPHA}"
  --spectral_knn "${SPECTRAL_KNN}"
)
if [[ "${SPECTRAL_EMBEDDING_METHOD}" != "semantic" ]]; then
  configure_graph_embedding_args "${SPECTRAL_EMBEDDING_METHOD}"
  RECSPECTRAL_CLUSTERING_ARGS+=("${GRAPH_EMBEDDING_ARGS[@]}")
fi

RECSPECTRAL_CONFIGURED=1

# Reuse the common dataset, SID generation, mapping, training, and evaluation
# pipeline without keeping spectral configuration in run_recdmon.sh.
# shellcheck source=run_recdmon.sh
source "${SCRIPT_DIR}/run_recdmon.sh"
