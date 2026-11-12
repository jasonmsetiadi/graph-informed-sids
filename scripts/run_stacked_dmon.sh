#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${DATASET:-}" ]]; then
  echo "Set DATASET to an Amazon category, MIND, or Yelp." >&2
  echo "Example: DATASET=Beauty bash scripts/run_stacked_dmon.sh" >&2
  exit 2
fi

# CLUSTER_SIZES is a whitespace-separated list, ordered from the coarsest
# level to the leaf level. Defaults match the original stacked DMoN setup.
export CLUSTER_SIZES="${CLUSTER_SIZES:-216 36 6 1}"

exec bash "${SCRIPT_DIR}/run_recdmon.sh" stacked_dmon
