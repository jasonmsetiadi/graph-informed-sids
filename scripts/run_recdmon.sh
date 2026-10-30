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
RECURSIVE_FACTOR=6
RECURSIVE_LEVELS=4

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
  elif [[ "${CLUSTERING_METHOD}" == "recursive_spectral" ]]; then
    CLUSTERING_LABEL="${SPECTRAL_EMBEDDING_METHOD}_alpha${SPECTRAL_FUSION_ALPHA}_k${SPECTRAL_KNN}"
  elif [[ "${CLUSTERING_METHOD}" == "recursive_leiden" ]]; then
    CLUSTERING_LABEL="${LEIDEN_EMBEDDING_METHOD}_alpha${LEIDEN_FUSION_ALPHA}_k${LEIDEN_KNN}"
  elif [[ "${CLUSTERING_METHOD}" == "recursive_dmon" ]]; then
    CLUSTERING_LABEL=""
  fi
  if [[ -n "${CLUSTERING_LABEL}" ]]; then
    ALIAS="${METHOD_NAME}_${CLUSTERING_LABEL}_${DATASET}_seed${SEED}"
  else
    ALIAS="${METHOD_NAME}_${DATASET}_seed${SEED}"
  fi
  INDEX_FILE=".${ALIAS}.index.json"

  OUT_DIR="${GENSID_CACHE_DIR}/mincut_runs/${ALIAS}"
  LABELS_PATH="${OUT_DIR}/graph_hierarchy_labels.json"

  # Path is relative to GR_CODEBASE.
  GR_SID_MAPPING_PATH="${GENSID_CACHE_DIR}/${DATASET}.${ALIAS}.json"

  # Mapper writes shell variables here; parent script sources it later.
  MAP_ENV="logs/env/${ALIAS}_sid_mapping.env"

  LOG_FILE="logs/${ALIAS,,}.log"
  FINAL_METRICS_FILE="results/${ALIAS,,}.json"

  echo -e "\n########################################"
  echo "Dataset: ${DATASET}"
  echo "Method:  ${METHOD_NAME}"
  echo "Seed:    ${SEED}"
  echo "Alias:   ${ALIAS}"
  echo "########################################"

  CLUSTERING_ARGS=(--method "${CLUSTERING_METHOD}")
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
      --recursive_factor "${RECURSIVE_FACTOR}"
      --recursive_levels "${RECURSIVE_LEVELS}"
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
      --label_group coarse_to_fine_item_labels \
      --unclustered_strategy special \
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

    echo "Using fixed GR SID mapping: ${GR_SEM_IDS_PATH}"
    echo "SID depth: ${GR_SEM_IDS_DEPTH}"
    echo "Codebook sizes: ${GR_SEM_IDS_CODEBOOK_SIZES}"
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
      --seed='${SEED}' \
      --sem_ids_path='${GR_SEM_IDS_PATH}' \
      --rq_n_codebooks='${GR_SEM_IDS_DEPTH}' \
      --rq_codebook_size='${GR_SEM_IDS_CODEBOOK_SIZES}' \
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
