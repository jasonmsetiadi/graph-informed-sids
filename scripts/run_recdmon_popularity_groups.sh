#!/usr/bin/env bash
set -euo pipefail

# Fixed 6-way branching and four hierarchy levels. Report metrics for all test
# examples and separately for head-, middle-, and tail-target examples.
export RECURSIVE_BRANCH_MODE="fixed"
export RECURSIVE_DEPTH_MODE="fixed"
export RECURSIVE_FACTOR="${RECURSIVE_FACTOR:-6}"
export RECURSIVE_LEVELS="${RECURSIVE_LEVELS:-4}"

# Match SID popularity statistics: top 20% head, bottom 20% tail, and compute
# popularity from training interactions after withholding validation and test.
export EVAL_TARGET_POPULARITY_GROUPS="true"
export EVAL_POPULARITY_EXTREME_FRACTION="${EVAL_POPULARITY_EXTREME_FRACTION:-0.2}"
export SKIP_N_LAST="${SKIP_N_LAST:-2}"

# Keep grouped result artifacts separate from the standard RecDMoN runs.
export METHOD_NAME="${METHOD_NAME:-RecDMoN_t5_popgroups}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_recdmon.sh" recursive_dmon
