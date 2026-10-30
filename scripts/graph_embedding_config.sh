#!/usr/bin/env bash

# Populate GRAPH_EMBEDDING_ARGS for one graph-aware embedding backend.
configure_graph_embedding_args() {
  local embedding_method="$1"
  GRAPH_EMBEDDING_ARGS=()

  case "${embedding_method}" in
    fastrp)
      FASTRP_DIM="${FASTRP_DIM:-256}"
      FASTRP_FEATURE_WEIGHT="${FASTRP_FEATURE_WEIGHT:-1.0}"
      FASTRP_SEED="${FASTRP_SEED:-42}"
      if ! declare -p FASTRP_ITERATION_WEIGHTS >/dev/null 2>&1; then
        FASTRP_ITERATION_WEIGHTS=(0.0 1.0 1.0)
      fi
      GRAPH_EMBEDDING_ARGS=(
        --fastrp_dim "${FASTRP_DIM}"
        --fastrp_feature_weight "${FASTRP_FEATURE_WEIGHT}"
        --fastrp_iteration_weights "${FASTRP_ITERATION_WEIGHTS[@]}"
        --fastrp_seed "${FASTRP_SEED}"
      )
      ;;
    s3gc)
      S3GC_DIM="${S3GC_DIM:-256}"
      S3GC_EPOCHS="${S3GC_EPOCHS:-200}"
      S3GC_BATCH_SIZE="${S3GC_BATCH_SIZE:-4096}"
      S3GC_LR="${S3GC_LR:-0.01}"
      S3GC_WEIGHT_DECAY="${S3GC_WEIGHT_DECAY:-0.0}"
      S3GC_WALK_LENGTH="${S3GC_WALK_LENGTH:-3}"
      S3GC_CONTEXT_SIZE="${S3GC_CONTEXT_SIZE:-3}"
      S3GC_WALKS_PER_NODE="${S3GC_WALKS_PER_NODE:-1}"
      S3GC_NUM_NEGATIVE_SAMPLES="${S3GC_NUM_NEGATIVE_SAMPLES:-1}"
      S3GC_DIFFUSION_HOPS="${S3GC_DIFFUSION_HOPS:-2}"
      S3GC_PPR_ALPHA="${S3GC_PPR_ALPHA:-0.2}"
      S3GC_SCALE="${S3GC_SCALE:-auto}"
      S3GC_SEED="${S3GC_SEED:-42}"
      S3GC_PRINT_EVERY="${S3GC_PRINT_EVERY:-10}"
      GRAPH_EMBEDDING_ARGS=(
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
      ;;
    magi)
      MAGI_DIM="${MAGI_DIM:-256}"
      MAGI_NUM_LAYERS="${MAGI_NUM_LAYERS:-2}"
      MAGI_EPOCHS="${MAGI_EPOCHS:-200}"
      MAGI_BATCH_SIZE="${MAGI_BATCH_SIZE:-512}"
      MAGI_LR="${MAGI_LR:-0.001}"
      MAGI_WEIGHT_DECAY="${MAGI_WEIGHT_DECAY:-0.0}"
      MAGI_NUM_NEIGHBORS="${MAGI_NUM_NEIGHBORS:-10}"
      MAGI_NUM_WALKS="${MAGI_NUM_WALKS:-20}"
      MAGI_WALK_LENGTH="${MAGI_WALK_LENGTH:-4}"
      MAGI_INFERENCE_BATCH_SIZE="${MAGI_INFERENCE_BATCH_SIZE:-2048}"
      MAGI_INFERENCE_NEIGHBORS="${MAGI_INFERENCE_NEIGHBORS:-15}"
      MAGI_TEMPERATURE="${MAGI_TEMPERATURE:-0.5}"
      MAGI_SEED="${MAGI_SEED:-42}"
      MAGI_PRINT_EVERY="${MAGI_PRINT_EVERY:-10}"
      GRAPH_EMBEDDING_ARGS=(
        --magi_dim "${MAGI_DIM}"
        --magi_num_layers "${MAGI_NUM_LAYERS}"
        --magi_epochs "${MAGI_EPOCHS}"
        --magi_batch_size "${MAGI_BATCH_SIZE}"
        --magi_lr "${MAGI_LR}"
        --magi_weight_decay "${MAGI_WEIGHT_DECAY}"
        --magi_num_neighbors "${MAGI_NUM_NEIGHBORS}"
        --magi_num_walks "${MAGI_NUM_WALKS}"
        --magi_walk_length "${MAGI_WALK_LENGTH}"
        --magi_inference_batch_size "${MAGI_INFERENCE_BATCH_SIZE}"
        --magi_inference_neighbors "${MAGI_INFERENCE_NEIGHBORS}"
        --magi_temperature "${MAGI_TEMPERATURE}"
        --magi_seed "${MAGI_SEED}"
        --magi_print_every "${MAGI_PRINT_EVERY}"
      )
      ;;
    ns4gc)
      NS4GC_DIM="${NS4GC_DIM:-256}"
      NS4GC_NUM_LAYERS="${NS4GC_NUM_LAYERS:-2}"
      NS4GC_EPOCHS="${NS4GC_EPOCHS:-200}"
      NS4GC_BATCH_SIZE="${NS4GC_BATCH_SIZE:-1024}"
      NS4GC_LR="${NS4GC_LR:-0.001}"
      NS4GC_WEIGHT_DECAY="${NS4GC_WEIGHT_DECAY:-0.0}"
      NS4GC_NUM_NEIGHBORS="${NS4GC_NUM_NEIGHBORS:-15}"
      NS4GC_INFERENCE_BATCH_SIZE="${NS4GC_INFERENCE_BATCH_SIZE:-2048}"
      NS4GC_INFERENCE_NEIGHBORS="${NS4GC_INFERENCE_NEIGHBORS:-15}"
      NS4GC_FEATURE_MASK="${NS4GC_FEATURE_MASK:-0.2}"
      NS4GC_EDGE_DROP="${NS4GC_EDGE_DROP:-0.3}"
      NS4GC_SIMILARITY_THRESHOLD="${NS4GC_SIMILARITY_THRESHOLD:-0.6}"
      NS4GC_TEMPERATURE="${NS4GC_TEMPERATURE:-0.1}"
      NS4GC_NEIGHBOR_WEIGHT="${NS4GC_NEIGHBOR_WEIGHT:-1.0}"
      NS4GC_SPARSITY_WEIGHT="${NS4GC_SPARSITY_WEIGHT:-1.0}"
      NS4GC_TRAINING_MODE="${NS4GC_TRAINING_MODE:-sampled}"
      NS4GC_MAX_FULL_NODES="${NS4GC_MAX_FULL_NODES:-10000}"
      NS4GC_SEED="${NS4GC_SEED:-42}"
      NS4GC_PRINT_EVERY="${NS4GC_PRINT_EVERY:-10}"
      GRAPH_EMBEDDING_ARGS=(
        --ns4gc_dim "${NS4GC_DIM}"
        --ns4gc_num_layers "${NS4GC_NUM_LAYERS}"
        --ns4gc_epochs "${NS4GC_EPOCHS}"
        --ns4gc_batch_size "${NS4GC_BATCH_SIZE}"
        --ns4gc_lr "${NS4GC_LR}"
        --ns4gc_weight_decay "${NS4GC_WEIGHT_DECAY}"
        --ns4gc_num_neighbors "${NS4GC_NUM_NEIGHBORS}"
        --ns4gc_inference_batch_size "${NS4GC_INFERENCE_BATCH_SIZE}"
        --ns4gc_inference_neighbors "${NS4GC_INFERENCE_NEIGHBORS}"
        --ns4gc_feature_mask "${NS4GC_FEATURE_MASK}"
        --ns4gc_edge_drop "${NS4GC_EDGE_DROP}"
        --ns4gc_similarity_threshold "${NS4GC_SIMILARITY_THRESHOLD}"
        --ns4gc_temperature "${NS4GC_TEMPERATURE}"
        --ns4gc_neighbor_weight "${NS4GC_NEIGHBOR_WEIGHT}"
        --ns4gc_sparsity_weight "${NS4GC_SPARSITY_WEIGHT}"
        --ns4gc_training_mode "${NS4GC_TRAINING_MODE}"
        --ns4gc_max_full_nodes "${NS4GC_MAX_FULL_NODES}"
        --ns4gc_seed "${NS4GC_SEED}"
        --ns4gc_print_every "${NS4GC_PRINT_EVERY}"
      )
      ;;
    *)
      echo "Unsupported graph embedding method: ${embedding_method}" >&2
      return 2
      ;;
  esac
}
