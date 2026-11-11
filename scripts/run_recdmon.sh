#!/usr/bin/env bash
set -euo pipefail


if [[ "${RECKMEANS_CONFIGURED:-0}" == "1" ]]; then
  CLUSTERING_METHOD="recursive_kmeans"
elif [[ "${RECSPECTRAL_CONFIGURED:-0}" == "1" ]]; then
  CLUSTERING_METHOD="recursive_spectral"
elif [[ "${RECLEIDEN_CONFIGURED:-0}" == "1" ]]; then
  CLUSTERING_METHOD="recursive_leiden"
else
  CLUSTERING_METHOD="${1:-${CLUSTERING_METHOD:-recursive_dmon}}"
fi

case "${CLUSTERING_METHOD}" in
  stacked_dmon|recursive_dmon) ;;
  recursive_leiden)
    if [[ "${RECLEIDEN_CONFIGURED:-0}" != "1" ]]; then
      echo "Use scripts/run_recleiden.sh for recursive Leiden clustering." >&2
      exit 2
    fi
    ;;
  recursive_spectral)
    if [[ "${RECSPECTRAL_CONFIGURED:-0}" != "1" ]]; then
      echo "Use scripts/run_recspectral.sh for recursive spectral clustering." >&2
      exit 2
    fi
    ;;
  recursive_kmeans)
    if [[ "${RECKMEANS_CONFIGURED:-0}" != "1" ]]; then
      echo "Use scripts/run_reckmeans.sh for recursive K-means." >&2
      exit 2
    fi
    ;;
  *)
    echo "Unsupported clustering method: ${CLUSTERING_METHOD}" >&2
    echo "Choose stacked_dmon or recursive_dmon." >&2
    exit 2
    ;;
esac


PIPELINE_FAILED=0

run_step () {
  local step_name="$1"
  shift

  # After first failure, skip all remaining steps.
  [[ "${PIPELINE_FAILED}" -ne 0 ]] && return 0

  echo -e "\n=== ${step_name} ==="

  if "$@"; then
    echo "✅ Step OK: ${step_name}"
  else
    local status=$?
    echo "❌ Step FAILED (exit code ${status}): ${step_name}"
    PIPELINE_FAILED=1
  fi
}

DATASET="${DATASET}"
# METHOD_NAME="RecDMoN_qwen"
# EMB_MODEL="Qwen3-Embedding-0.6B"
METHOD_NAME="${METHOD_NAME:-RecDMoN_t5}"
EMB_MODEL="sentence-t5-base"
SEEDS=(42 43 44)

CLUSTERING_METHODS=("${CLUSTERING_METHOD}")
CLUSTER_SIZES=(216 36 6 1)
if [[ -n "${RECURSIVE_MODE:-}" ]]; then
  echo "RECURSIVE_MODE has been removed; set RECURSIVE_BRANCH_MODE and RECURSIVE_DEPTH_MODE." >&2
  exit 2
fi
RECURSIVE_LEVELS="${RECURSIVE_LEVELS:-4}"
RECURSIVE_BRANCH_MODE="${RECURSIVE_BRANCH_MODE:-adaptive}"
RECURSIVE_DEPTH_MODE="${RECURSIVE_DEPTH_MODE:-adaptive}"
RECURSIVE_ITEM_DEPTH_FUNCTION="${RECURSIVE_ITEM_DEPTH_FUNCTION:-sigmoid}"
RECURSIVE_ITEM_DEPTH_MIN="${RECURSIVE_ITEM_DEPTH_MIN:-4}"
RECURSIVE_ITEM_DEPTH_MAX="${RECURSIVE_ITEM_DEPTH_MAX:-10}"
RECURSIVE_ITEM_DEPTH_ALPHA="${RECURSIVE_ITEM_DEPTH_ALPHA:-1.0}"
RECURSIVE_ITEM_DEPTH_BETA="${RECURSIVE_ITEM_DEPTH_BETA:-0.5}"
RECURSIVE_ITEM_DEPTH_GAMMA="${RECURSIVE_ITEM_DEPTH_GAMMA:-1.0}"
RECURSIVE_POPULARITY_EPSILON_BASE="${RECURSIVE_POPULARITY_EPSILON_BASE:-0.001}"
RECURSIVE_POPULARITY_ALPHA="${RECURSIVE_POPULARITY_ALPHA:-0}"
DMON_SELECTION_EPOCHS="${DMON_SELECTION_EPOCHS:-50}"
DMON_POPULARITY_BALANCE_LAMBDA="${DMON_POPULARITY_BALANCE_LAMBDA:-0}"
DMON_POPULARITY_MASS_OBJECTIVE="${DMON_POPULARITY_MASS_OBJECTIVE:-balance}"
DMON_POPULARITY_PURITY_BINS="${DMON_POPULARITY_PURITY_BINS:-3}"
SKIP_N_LAST="${SKIP_N_LAST:-2}"
EVAL_TARGET_POPULARITY_GROUPS="${EVAL_TARGET_POPULARITY_GROUPS:-false}"
EVAL_POPULARITY_EXTREME_FRACTION="${EVAL_POPULARITY_EXTREME_FRACTION:-0.2}"
case "${RECURSIVE_BRANCH_MODE}" in
  adaptive|fixed) ;;
  *)
    echo "Unsupported RECURSIVE_BRANCH_MODE='${RECURSIVE_BRANCH_MODE}'. Choose adaptive or fixed." >&2
    exit 2
    ;;
esac
case "${RECURSIVE_DEPTH_MODE}" in
  adaptive|item|fixed) ;;
  *)
    echo "Unsupported RECURSIVE_DEPTH_MODE='${RECURSIVE_DEPTH_MODE}'. Choose adaptive, item, or fixed." >&2
    exit 2
    ;;
esac
case "${RECURSIVE_ITEM_DEPTH_FUNCTION}" in
  sigmoid|linear) ;;
  *)
    echo "Unsupported RECURSIVE_ITEM_DEPTH_FUNCTION='${RECURSIVE_ITEM_DEPTH_FUNCTION}'. Choose sigmoid or linear." >&2
    exit 2
    ;;
esac
case "${DMON_POPULARITY_MASS_OBJECTIVE}" in
  balance|concentrate|separate|purity) ;;
  *)
    echo "Unsupported DMON_POPULARITY_MASS_OBJECTIVE='${DMON_POPULARITY_MASS_OBJECTIVE}'. Choose balance, concentrate, separate, or purity." >&2
    exit 2
    ;;
esac
if [[ "${RECURSIVE_BRANCH_MODE}" == "fixed" ]]; then
  DEFAULT_RECURSIVE_FACTOR=6
else
  DEFAULT_RECURSIVE_FACTOR=10
fi
RECURSIVE_FACTOR="${RECURSIVE_FACTOR:-${DEFAULT_RECURSIVE_FACTOR}}"

GENSID_DIR="./gensid"

get_dataset_prefix() {
  local dataset="$1"
  case "${dataset}" in
    Musical_Instruments|Beauty|Toys_and_Games|Sports_and_Outdoors|Books) echo AmazonReviews2014/ ;;
    *) echo "" ;;
  esac
}

GENSID_CACHE_DIR="cache/gensid"
GR_CACHE_DIR="cache"
DATA_PATH="${GENSID_CACHE_DIR}/${DATASET}/${DATASET}.emb-${EMB_MODEL}-td.npy"
INTERACTIONS_PATH="${GENSID_CACHE_DIR}/${DATASET}/${DATASET}.inter.json"
GENSID_CKPT_PATH="ckpt/gensid"
CF_EMB="${GENSID_CKPT_PATH}/${DATASET}-32d-SASRec.pt"

GR_PATH="${GR_CACHE_DIR}/$(get_dataset_prefix "${DATASET}")${DATASET}/processed"

# Amazon data is selected by category; MIND and Yelp are standalone datasets.
case "${DATASET}" in
  Musical_Instruments|Beauty|Toys_and_Games|Sports_and_Outdoors|Books)
    GENREC_DATASET_ARGS="--dataset=AmazonReviews2014 --category='${DATASET}'"
    ;;
  MIND|Yelp)
    GENREC_DATASET_ARGS="--dataset='${DATASET}'"
    ;;
  *)
    echo "Unsupported DATASET='${DATASET}'"
    echo "Choose an Amazon category, MIND, or Yelp."
    exit 2
    ;;
esac

DEVICE_NO=0
DEVICE="cuda:${DEVICE_NO}"

mkdir -p "logs/env"

for SEED in "${SEEDS[@]}"; do
  [[ "${PIPELINE_FAILED}" -ne 0 ]] && break
  for CLUSTERING_METHOD in "${CLUSTERING_METHODS[@]}"; do
  [[ "${PIPELINE_FAILED}" -ne 0 ]] && break

  # Alias identifies every artifact from this run.
  CLUSTERING_LABEL="${CLUSTERING_METHOD}"
  if [[ "${CLUSTERING_METHOD}" == "recursive_kmeans" ]]; then
    CLUSTERING_LABEL="${GRAPH_EMBEDDING_METHOD}"
    if [[ "${GRAPH_EMBEDDING_METHOD}" == "appnp" ]]; then
      CLUSTERING_LABEL="appnp_k${APPNP_NUM_PROP}_alpha${APPNP_ALPHA}"
    elif [[ "${GRAPH_EMBEDDING_METHOD}" == "ns4gc" ]]; then
      CLUSTERING_LABEL="${CLUSTERING_LABEL}_${NS4GC_TRAINING_MODE}"
    fi
  elif [[ "${CLUSTERING_METHOD}" == "recursive_spectral" ]]; then
    CLUSTERING_LABEL="${SPECTRAL_EMBEDDING_METHOD}_alpha${SPECTRAL_FUSION_ALPHA}_k${SPECTRAL_KNN}"
    if [[ "${SPECTRAL_EMBEDDING_METHOD}" == "ns4gc" ]]; then
      CLUSTERING_LABEL="ns4gc_${NS4GC_TRAINING_MODE}_alpha${SPECTRAL_FUSION_ALPHA}_k${SPECTRAL_KNN}"
    fi
  elif [[ "${CLUSTERING_METHOD}" == "recursive_leiden" ]]; then
    CLUSTERING_LABEL="${LEIDEN_EMBEDDING_METHOD}_alpha${LEIDEN_FUSION_ALPHA}_k${LEIDEN_KNN}"
    if [[ "${LEIDEN_EMBEDDING_METHOD}" == "ns4gc" ]]; then
      CLUSTERING_LABEL="ns4gc_${NS4GC_TRAINING_MODE}_alpha${LEIDEN_FUSION_ALPHA}_k${LEIDEN_KNN}"
    fi
  elif [[ "${CLUSTERING_METHOD}" == "recursive_dmon" ]]; then
    CLUSTERING_LABEL=""
  fi
  if [[ "${CLUSTERING_METHOD}" == "recursive_kmeans" || "${CLUSTERING_METHOD}" == "recursive_spectral" || "${CLUSTERING_METHOD}" == "recursive_dmon" ]]; then
    MODE_LABEL="b${RECURSIVE_BRANCH_MODE}_d${RECURSIVE_DEPTH_MODE}"
    if [[ "${RECURSIVE_BRANCH_MODE}" == "fixed" ]]; then
      MODE_LABEL="${MODE_LABEL}_f${RECURSIVE_FACTOR}"
    else
      MODE_LABEL="${MODE_LABEL}_max${RECURSIVE_FACTOR}"
    fi
    if [[ "${RECURSIVE_DEPTH_MODE}" == "fixed" ]]; then
      MODE_LABEL="${MODE_LABEL}_l${RECURSIVE_LEVELS}"
    elif [[ "${RECURSIVE_DEPTH_MODE}" == "item" ]]; then
      ITEM_DEPTH_MIN_LABEL="${RECURSIVE_ITEM_DEPTH_MIN//./p}"
      ITEM_DEPTH_MAX_LABEL="${RECURSIVE_ITEM_DEPTH_MAX//./p}"
      if [[ "${RECURSIVE_ITEM_DEPTH_FUNCTION}" == "sigmoid" ]]; then
        ITEM_DEPTH_ALPHA_LABEL="${RECURSIVE_ITEM_DEPTH_ALPHA//./p}"
        ITEM_DEPTH_BETA_LABEL="${RECURSIVE_ITEM_DEPTH_BETA//./p}"
        MODE_LABEL="${MODE_LABEL}_sig${ITEM_DEPTH_ALPHA_LABEL}b${ITEM_DEPTH_BETA_LABEL}_lmin${ITEM_DEPTH_MIN_LABEL}_lmax${ITEM_DEPTH_MAX_LABEL}"
      else
        ITEM_DEPTH_GAMMA_LABEL="${RECURSIVE_ITEM_DEPTH_GAMMA//./p}"
        MODE_LABEL="${MODE_LABEL}_lin${ITEM_DEPTH_GAMMA_LABEL}_lmin${ITEM_DEPTH_MIN_LABEL}_lmax${ITEM_DEPTH_MAX_LABEL}"
      fi
    fi
    if [[ -n "${CLUSTERING_LABEL}" ]]; then
      CLUSTERING_LABEL="${CLUSTERING_LABEL}_${MODE_LABEL}"
    else
      CLUSTERING_LABEL="${MODE_LABEL}"
    fi
  fi
  if [[ "${CLUSTERING_METHOD}" == "recursive_kmeans" || "${CLUSTERING_METHOD}" == "recursive_spectral" || "${CLUSTERING_METHOD}" == "recursive_leiden" || "${CLUSTERING_METHOD}" == "recursive_dmon" ]]; then
    POPULARITY_BASE_LABEL="${RECURSIVE_POPULARITY_EPSILON_BASE//./p}"
    POPULARITY_ALPHA_LABEL="${RECURSIVE_POPULARITY_ALPHA//./p}"
    CLUSTERING_LABEL="${CLUSTERING_LABEL}_popb${POPULARITY_BASE_LABEL}_a${POPULARITY_ALPHA_LABEL}"
  fi
  if [[ "${CLUSTERING_METHOD}" == "recursive_dmon" ]]; then
    DMON_POPULARITY_BALANCE_LABEL="${DMON_POPULARITY_BALANCE_LAMBDA//./p}"
    if [[ "${DMON_POPULARITY_MASS_OBJECTIVE}" == "concentrate" ]]; then
      CLUSTERING_LABEL="${CLUSTERING_LABEL}_popconc${DMON_POPULARITY_BALANCE_LABEL}"
    elif [[ "${DMON_POPULARITY_MASS_OBJECTIVE}" == "separate" ]]; then
      CLUSTERING_LABEL="${CLUSTERING_LABEL}_popsep${DMON_POPULARITY_BALANCE_LABEL}"
    elif [[ "${DMON_POPULARITY_MASS_OBJECTIVE}" == "purity" ]]; then
      CLUSTERING_LABEL="${CLUSTERING_LABEL}_poppure${DMON_POPULARITY_PURITY_BINS}_${DMON_POPULARITY_BALANCE_LABEL}"
    else
      CLUSTERING_LABEL="${CLUSTERING_LABEL}_bal${DMON_POPULARITY_BALANCE_LABEL}"
    fi
  fi
  if [[ -n "${CLUSTERING_LABEL}" ]]; then
    EXPERIMENT_ALIAS="${METHOD_NAME}_${CLUSTERING_LABEL}_${DATASET}"
  else
    EXPERIMENT_ALIAS="${METHOD_NAME}_${DATASET}"
  fi
  ALIAS="${EXPERIMENT_ALIAS}_seed${SEED}"
  INDEX_FILE=".${ALIAS}.index.json"

  OUT_DIR="${GENSID_CACHE_DIR}/mincut_runs/${ALIAS}"
  LABELS_PATH="${OUT_DIR}/graph_hierarchy_labels.json"

  # Path is relative to GR_CODEBASE.
  GR_SID_MAPPING_PATH="${GENSID_CACHE_DIR}/${DATASET}.${ALIAS}.json"

  # Mapper writes shell variables here; parent script sources it later.
  MAP_ENV="logs/env/${ALIAS}_sid_mapping.env"

  LOG_FILE="logs/${ALIAS,,}.log"
  case "${CLUSTERING_METHOD}" in
    recursive_dmon|stacked_dmon) RESULTS_METHOD="dmon" ;;
    recursive_spectral) RESULTS_METHOD="spectral" ;;
    recursive_kmeans) RESULTS_METHOD="kmeans" ;;
    recursive_leiden) RESULTS_METHOD="leiden" ;;
    *) RESULTS_METHOD="${CLUSTERING_METHOD}" ;;
  esac
  RESULTS_DIR="results/${RESULTS_METHOD}/${EXPERIMENT_ALIAS,,}"
  FINAL_METRICS_FILE="${RESULTS_DIR}/seed${SEED}.metrics.json"
  SID_STATS_FILE="${RESULTS_DIR}/seed${SEED}.sid_stats.json"
  mkdir -p "${RESULTS_DIR}"

  echo -e "\n########################################"
  echo "Dataset: ${DATASET}"
  echo "Method:  ${METHOD_NAME}"
  echo "Seed:    ${SEED}"
  echo "Alias:   ${ALIAS}"
  echo "########################################"

  CLUSTERING_ARGS=(--method "${CLUSTERING_METHOD}")
  SID_INDEX_ARGS=()
  MAPPER_VARIABLE_LENGTH_ARG=""
  if [[ "${CLUSTERING_METHOD}" == "recursive_leiden" || "${RECURSIVE_DEPTH_MODE}" != "fixed" && ( "${CLUSTERING_METHOD}" == "recursive_spectral" || "${CLUSTERING_METHOD}" == "recursive_dmon" || "${CLUSTERING_METHOD}" == "recursive_kmeans" ) ]]; then
    SID_INDEX_ARGS+=(--variable_length)
    MAPPER_VARIABLE_LENGTH_ARG="--variable_length_sids"
  fi
  if [[ "${CLUSTERING_METHOD}" == "recursive_spectral" || "${CLUSTERING_METHOD}" == "recursive_dmon" || "${CLUSTERING_METHOD}" == "recursive_kmeans" ]]; then
    CLUSTERING_ARGS+=(
      --recursive_branch_mode "${RECURSIVE_BRANCH_MODE}"
      --recursive_depth_mode "${RECURSIVE_DEPTH_MODE}"
      --recursive_factor "${RECURSIVE_FACTOR}"
      --recursive_levels "${RECURSIVE_LEVELS}"
    )
    if [[ "${RECURSIVE_DEPTH_MODE}" == "item" ]]; then
      CLUSTERING_ARGS+=(
        --recursive_item_depth_function "${RECURSIVE_ITEM_DEPTH_FUNCTION}"
        --recursive_item_depth_min "${RECURSIVE_ITEM_DEPTH_MIN}"
        --recursive_item_depth_max "${RECURSIVE_ITEM_DEPTH_MAX}"
        --recursive_item_depth_alpha "${RECURSIVE_ITEM_DEPTH_ALPHA}"
        --recursive_item_depth_beta "${RECURSIVE_ITEM_DEPTH_BETA}"
        --recursive_item_depth_gamma "${RECURSIVE_ITEM_DEPTH_GAMMA}"
      )
    fi
  fi
  if [[ "${CLUSTERING_METHOD}" == "recursive_kmeans" || "${CLUSTERING_METHOD}" == "recursive_spectral" || "${CLUSTERING_METHOD}" == "recursive_leiden" || "${CLUSTERING_METHOD}" == "recursive_dmon" ]]; then
    CLUSTERING_ARGS+=(
      --recursive_popularity_epsilon_base "${RECURSIVE_POPULARITY_EPSILON_BASE}"
      --recursive_popularity_alpha "${RECURSIVE_POPULARITY_ALPHA}"
    )
  fi
  if [[ "${CLUSTERING_METHOD}" == "stacked_dmon" ]]; then
    CLUSTERING_ARGS+=(--cluster_sizes "${CLUSTER_SIZES[@]}")
  elif [[ "${CLUSTERING_METHOD}" == "recursive_spectral" ]]; then
    CLUSTERING_ARGS+=("${RECSPECTRAL_CLUSTERING_ARGS[@]}")
  elif [[ "${CLUSTERING_METHOD}" == "recursive_kmeans" ]]; then
    CLUSTERING_ARGS+=("${RECKMEANS_CLUSTERING_ARGS[@]}")
  elif [[ "${CLUSTERING_METHOD}" == "recursive_leiden" ]]; then
    CLUSTERING_ARGS+=("${RECLEIDEN_CLUSTERING_ARGS[@]}")
  elif [[ "${CLUSTERING_METHOD}" == "recursive_dmon" ]]; then
    CLUSTERING_ARGS+=(
      --dmon_selection_epochs "${DMON_SELECTION_EPOCHS}"
      --dmon_popularity_balance_lambda "${DMON_POPULARITY_BALANCE_LAMBDA}"
      --dmon_popularity_mass_objective "${DMON_POPULARITY_MASS_OBJECTIVE}"
      --dmon_popularity_purity_bins "${DMON_POPULARITY_PURITY_BINS}"
    )
  fi

  # =========================
  # STEP 1.1: GRAPH CLUSTERING
  # =========================
  run_step "Graph clustering (MinCut)" \
    python -u "${GENSID_DIR}/hierarchical_mincut_clustering.py" \
      --interactions_path "${INTERACTIONS_PATH}" \
      --embeddings_path "${DATA_PATH}" \
      "${CLUSTERING_ARGS[@]}" \
      --graph_type adjacent_cooc \
      --skip_n_last "${SKIP_N_LAST}" \
      --lr 3e-4 \
      --assign_hidden_dim 256 \
      --dropout 0.0 \
      --recursive_epochs 1000 \
      --device "${DEVICE}" \
      --output_dir "${OUT_DIR}" \
      --output_prefix graph_hierarchy \
      --max_dense_nodes 100000

  # =========================
  # STEP 1.2: INDEX GENERATION
  # =========================
  run_step "Generate graph-cluster SID index" \
    python "${GENSID_DIR}/generate_graph_cluster_indices.py" \
      --labels_path "${LABELS_PATH}" \
      --output_file "${GENSID_CACHE_DIR}/${DATASET}/${DATASET}${INDEX_FILE}" \
      --stats_file "${SID_STATS_FILE}" \
      --label_group coarse_to_fine_item_labels \
      --unclustered_strategy special \
      --interactions_path "${INTERACTIONS_PATH}" \
      --popularity_skip_n_last "${SKIP_N_LAST}" \
      "${SID_INDEX_ARGS[@]}" \
      --append_unique_leaf

  # =========================
  # 2. Copy SID mapping to GR / LETTER cache
  # =========================

  run_step "Copy SID mapping to GR LETTER cache | ${ALIAS}" \
  cp "${GENSID_CACHE_DIR}/${DATASET}/${DATASET}${INDEX_FILE}" "${GR_SID_MAPPING_PATH}"

  # =========================
  # 3. Remap LETTER-style SID mapping to GR .sem_ids
  # =========================

  run_step "Remap SID mapping to GR format | ${ALIAS}" bash -c "
  python map_gensid_to_genrec.py \
    --sid_mapping_path '${GR_SID_MAPPING_PATH}' \
    --gr_path '${GR_PATH}' \
    --dataset '${DATASET}' \
    ${MAPPER_VARIABLE_LENGTH_ARG} \
    --env_file '${MAP_ENV}'
  "

  if [[ "${PIPELINE_FAILED}" -eq 0 ]]; then
    # Import mapper outputs into this shell.
    # shellcheck disable=SC1090
    source "${MAP_ENV}"

    # Fail early if mapper did not write required variables.
    : "${GR_SEM_IDS_PATH:?Missing GR_SEM_IDS_PATH in ${MAP_ENV}}"
    : "${GR_SEM_IDS_DEPTH:?Missing GR_SEM_IDS_DEPTH in ${MAP_ENV}}"
    : "${GR_SEM_IDS_CODEBOOK_SIZES:?Missing GR_SEM_IDS_CODEBOOK_SIZES in ${MAP_ENV}}"
    : "${GR_SEM_IDS_VARIABLE_LENGTH:?Missing GR_SEM_IDS_VARIABLE_LENGTH in ${MAP_ENV}}"

    echo "Using GR SID mapping: ${GR_SEM_IDS_PATH}"
    echo "SID depth: ${GR_SEM_IDS_DEPTH}"
    echo "Codebook sizes: ${GR_SEM_IDS_CODEBOOK_SIZES}"
    echo "Variable-length SIDs: ${GR_SEM_IDS_VARIABLE_LENGTH}"
  fi

  # =========================
  # 4. Finetune GR
  # =========================

  if [[ "${PIPELINE_FAILED}" -eq 0 ]]; then
    run_step "Finetune MemGen-GR with mapped SIDs | ${ALIAS}" bash -c "
    CUDA_VISIBLE_DEVICES=${DEVICE_NO} accelerate launch \
      --mixed_precision bf16 \
      --dynamo_backend inductor \
      --num_processes 1 main_genrec.py \
      --final_metrics_file='${FINAL_METRICS_FILE}'\
      --model=TIGER \
      ${GENREC_DATASET_ARGS} \
      --max_item_seq_len=20 \
      --val_eval_user_limit=0.2 \
      --train_batch_size=512 \
      --eval_batch_size=32 \
      --eval_target_popularity_groups='${EVAL_TARGET_POPULARITY_GROUPS}' \
      --eval_popularity_extreme_fraction='${EVAL_POPULARITY_EXTREME_FRACTION}' \
      --eval_popularity_skip_n_last='${SKIP_N_LAST}' \
      --seed='${SEED}' \
      --sem_ids_path='${GR_SEM_IDS_PATH}' \
      --rq_n_codebooks='${GR_SEM_IDS_DEPTH}' \
      --rq_codebook_size='${GR_SEM_IDS_CODEBOOK_SIZES}' \
      --variable_length_sids='${GR_SEM_IDS_VARIABLE_LENGTH}' \
      --config=aux_configs/mmgrec.yaml \
      2>&1 | tee '${LOG_FILE}'
    "
  fi
  done
done

echo
if [[ "${PIPELINE_FAILED}" -ne 0 ]]; then
  echo "🎯 Pipeline stopped after first failure"
else
  echo "🎯 Pipeline finished successfully"
fi
