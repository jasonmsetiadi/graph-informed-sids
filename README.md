# Graph-Informed Semantic IDs (GrIS Framework)

Official implementation of "Neither Black nor White: Balancing Semantic and Collaborative Signals with Graph-Informed Semantic IDs (GrIS)".

## Project Structure

- `gensid`: Based on the [LETTER codebase](https://github.com/HonghuiBao2000/LETTER), extended to generate graph-based semantic IDs via non-parametric message passing.
- `genrec`: Based on [MemGen-GR](https://github.com/Jamesding000/MemGen-GR) and [phonism's](https://github.com/phonism/genrec) codebases, contains the GR and atomic baselines (SASRec, HSTU, and TIGER). Several improvements were added, including fixed evaluation in case of SID collisions, enhanced logging/results saving, a tokenizer abstraction to easily add new models, improved config handling, extended dataset and tokenizer classes to work with extra sequence columns (e.g. time and rating), and support for new datasets ([MIND-small](https://msnews.github.io/#getting-started) and [Yelp](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/download?datasetVersionNumber=1)).

## Installation

```shell
conda create -n gr_env python=3.11 -y
conda activate gr_env
pip install -r requirements.txt
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.11.0+cu126.html
```

The pinned PyTorch CUDA 12.6 build supports NVIDIA V100 GPUs (compute capability
7.0). On a machine without a compatible NVIDIA GPU, install the CPU-only PyTorch
build instead of using the CUDA wheel index above.

## Reproducing Experiments

The MIND dataset requires signing Microsoft's license terms, and Yelp 2018 is only available via Kaggle. Download both manually and place them under `cache/{dataset}/raw/`. The dataset classes handle processing the raw files.

1. **Prepare the dataset** — downloads Amazon datasets (if applicable), prepares item text representations, and generates sentence embeddings for SID generation:

```shell
python prepare_dataset.py --category=Beauty \
  --sent_emb_model=sentence-transformers/sentence-t5-base \
  --sent_emb_batch_size=64

# Other Amazon categories:
# python prepare_dataset.py --category=Toys_and_Games
# python prepare_dataset.py --category=Sports_and_Outdoors
# python prepare_dataset.py --category=Books
# python prepare_dataset.py --dataset=Yelp
# python prepare_dataset.py --dataset=MIND
```

2. **Run the experiments:**

```shell
bash scripts/run_sasrec.sh
bash scripts/run_hstu.sh
bash scripts/run_letter.sh                           # Pretrains a CF model for LETTER CF-loss regularization
bash scripts/run_s2gr.sh
bash scripts/run_mmgrec.sh
bash scripts/run_gae_components_ablation.sh          # TIGER, RQ-GAE, RQ-GAE w/o graph-loss, RQ-GAE w/o graph-input
bash scripts/run_recdmon.sh
bash scripts/run_reckmeans.sh semantic                # Recursive K-means over the original semantic embeddings
bash scripts/run_reckmeans.sh appnp                   # Recursive K-means over APPNP-propagated semantic embeddings
bash scripts/run_reckmeans.sh fastrp                 # Recursive K-means over FastRP embeddings
bash scripts/run_recspectral.sh semantic             # Recursive spectral with semantic-embedding kNN fusion
bash scripts/run_recspectral.sh fastrp               # Recursive spectral with FastRP-embedding kNN fusion
bash scripts/run_recspectral.sh s3gc                 # Recursive spectral with S3GC-embedding kNN fusion
bash scripts/run_recleiden.sh semantic               # Recursive Leiden with semantic-embedding kNN fusion
bash scripts/run_recleiden.sh fastrp                 # Recursive Leiden with FastRP-embedding kNN fusion
bash scripts/run_recleiden.sh s3gc                   # Recursive Leiden with S3GC-embedding kNN fusion
bash scripts/run_reckmeans.sh magi                   # MAGI and NS4GC are also available to all three runners
bash scripts/run_recspectral.sh ns4gc                # Local weighted NS4GC, sampled mode by default
NS4GC_TRAINING_MODE=full bash scripts/run_reckmeans.sh ns4gc  # Exact small-graph NS4GC
```

Recursive K-means embedding choices are `semantic`, `appnp`, `fastrp`, `s3gc`, `magi`, and `ns4gc`.
APPNP defaults to the full RQ-GAE propagation settings (`APPNP_NUM_PROP=1`,
`APPNP_ALPHA=0.5`) and uses graph edge weights when available.
FastRP, S3GC, MAGI, and NS4GC default to 768 output dimensions to match the
default `sentence-t5-base` semantic embeddings; their `*_DIM` variables remain configurable.
Spectral and Leiden embedding choices are `semantic`, `fastrp`, `s3gc`, `magi`, and `ns4gc`.
For recursive K-means, `semantic` clusters the original item embeddings directly. For
recursive spectral and Leiden, it converts those embeddings into a cosine kNN graph.
All three recursive clustering runners default to `semantic`.
Recursive DMoN, spectral, and K-means control branch count and depth independently:

```shell
# Adaptive branching and adaptive depth (default; variable-length SIDs)
RECURSIVE_BRANCH_MODE=adaptive RECURSIVE_DEPTH_MODE=adaptive \
  bash scripts/run_recdmon.sh

# Fixed branching and fixed depth (fixed-length SIDs)
RECURSIVE_BRANCH_MODE=fixed RECURSIVE_DEPTH_MODE=fixed \
  bash scripts/run_recdmon.sh

# Adaptive branching at exactly four levels (fixed-length SIDs)
RECURSIVE_BRANCH_MODE=adaptive RECURSIVE_DEPTH_MODE=fixed \
  bash scripts/run_recdmon.sh

# Factor-six branching with natural stopping (variable-length SIDs)
RECURSIVE_BRANCH_MODE=fixed RECURSIVE_DEPTH_MODE=adaptive \
  bash scripts/run_recdmon.sh
```

The same controls work with `run_recspectral.sh` and `run_reckmeans.sh`.
Adaptive branching defaults to maximum factor 10; fixed branching defaults to
factor 6; fixed depth defaults to four levels. SID storage follows depth mode:
adaptive depth is variable-length and fixed depth is fixed-length. Recursive
Leiden remains adaptive and variable-length. Both SID modes use the same
catalog-trie-constrained TIGER beam-search decoder; fixed mode simply has
equal-depth trie paths.
Each generated SID stats JSON also follows the adjacent `varlen_semantic_ids`
naming for length/popularity analysis: `length.mean_uniform/mean_unigram`,
`popularity_length.pop_stats_by_length`, and `head_longtail`. Per-length
popularity statistics include count, minimum, maximum, mean, median, and standard
deviation. Length includes the terminal leaf. The head/middle/long-tail split is
the most-popular 20%, middle 60%, and least-popular 20% by default, and frequency
uses the graph's `SKIP_N_LAST` policy (default `2`).
The `ns4gc` backend is a local weighted variant rather than the standard unweighted
PyAGC objective. It uses raw interaction edge weights in GCN propagation and the
neighbor-consistency loss, falling back to unit weights only when weights are absent.
Use `NS4GC_TRAINING_MODE=sampled` for scalable training or `full` for exact
full-graph training up to `NS4GC_MAX_FULL_NODES` (default `10000`).

The `scripts/` directory contains all experiments needed to reproduce the results in the paper. Results are saved under `results/`, named by the alias specified in each script.

## Acknowledgements

We sincerely thank the authors of [LETTER](https://github.com/HonghuiBao2000/LETTER), [MemGen-GR](https://github.com/Jamesding000/MemGen-GR), and [phonism](https://github.com/phonism/genrec) repositories for their valuable contributions to the open-source community.
