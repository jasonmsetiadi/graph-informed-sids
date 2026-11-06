#!/usr/bin/env python3
"""
Generate semantic-ID index mapping directly from hierarchical graph-cluster labels.

This script converts graph clustering output, e.g. from hierarchical_mincut_clustering.py,
into the JSON format used by downstream generative recommendation training:

    {
      "0": ["<a_3>", "<b_17>", "<c_5>"],
      "1": ["<a_3>", "<b_18>", "<c_2>"],
      ...
    }

The expected clustering labels are item-id indexed arrays, usually stored as:

    {
      "coarse_to_fine_item_labels": {
        "level_1": [...],
        "level_2": [...]
      }
    }

For RQ-VAE-style prefix semantics, use the coarse-to-fine group. Each level becomes
one SID token. Optionally, append a uniqueness token to eliminate full-SID collisions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np


def make_default_prefixes(num_levels: int) -> list[str]:
    """Create alphabetic token prefixes without imposing a hierarchy-depth limit."""
    prefixes = []
    for index in range(num_levels):
        value = index + 1
        letters = ""
        while value > 0:
            value, remainder = divmod(value - 1, 26)
            letters = chr(ord("a") + remainder) + letters
        prefixes.append(f"<{letters}_{{}}>")
    return prefixes


def _is_label_vector(x: Any) -> bool:
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], int))


def _is_list_of_label_vectors(x: Any) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(isinstance(v, list) for v in x)


def _level_sort_key(key: str):
    m = re.search(r"(\d+)$", str(key))
    if m is not None:
        return int(m.group(1))
    return str(key)


def _normalize_level_name(level: str | int) -> str:
    if isinstance(level, int):
        return f"level_{level}"
    if isinstance(level, str):
        if level.isdigit():
            return f"level_{int(level)}"
        return level
    raise ValueError(f"Invalid level specifier: {level!r}")


def load_cluster_label_levels(
    labels_path: str,
    label_group: str = "coarse_to_fine_item_labels",
    levels: list[str] | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Load one or more item-id indexed cluster-label arrays from a clustering JSON.

    Supports:
      1. grouped dict: {"coarse_to_fine_item_labels": {"level_1": [...], ...}}
      2. flat dict: {"level_1": [...], "level_2": [...]}
      3. list of levels: [[...], [...]]
      4. single vector: [...]
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    with open(labels_path, "r") as f:
        payload = json.load(f)

    if _is_label_vector(payload):
        level_dict = {"level_1": payload}
    elif _is_list_of_label_vectors(payload):
        level_dict = {f"level_{i + 1}": v for i, v in enumerate(payload)}
    elif isinstance(payload, dict):
        if label_group in payload:
            level_dict = payload[label_group]
        elif any(str(k).startswith("level_") for k in payload.keys()):
            level_dict = payload
        else:
            candidate_groups = {
                k: v
                for k, v in payload.items()
                if isinstance(v, dict) and any(str(kk).startswith("level_") for kk in v.keys())
            }
            if len(candidate_groups) == 1:
                level_dict = next(iter(candidate_groups.values()))
            else:
                raise ValueError(
                    f"Could not select label group. Requested {label_group!r}; "
                    f"available candidate groups: {sorted(candidate_groups.keys())}"
                )
    else:
        raise ValueError(f"Unsupported JSON root type: {type(payload).__name__}")

    if not isinstance(level_dict, dict):
        raise ValueError(f"Selected label group must be dict, got {type(level_dict).__name__}")

    available = sorted(level_dict.keys(), key=_level_sort_key)
    selected = available if levels is None else [_normalize_level_name(l) for l in levels]

    out_names: list[str] = []
    out_arrays: list[np.ndarray] = []
    for name in selected:
        if name not in level_dict:
            raise ValueError(f"Requested level {name!r} not found. Available: {available}")
        arr = np.asarray(level_dict[name], dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(f"Level {name!r} must be 1-D, got shape {arr.shape}")
        out_names.append(name)
        out_arrays.append(arr)

    lengths = {len(a) for a in out_arrays}
    if len(lengths) != 1:
        raise ValueError(f"All levels must have the same length, got {sorted(lengths)}")

    return out_names, out_arrays


def make_contiguous_per_level(labels: list[np.ndarray], keep_negative: bool = True) -> tuple[list[np.ndarray], list[dict[int, int]]]:
    """Remap arbitrary cluster ids to compact 0..K-1 ids per level."""
    remapped = []
    maps = []
    for arr in labels:
        valid = np.unique(arr[arr >= 0] if keep_negative else arr)
        mapping = {int(old): int(new) for new, old in enumerate(sorted(map(int, valid)))}
        new_arr = np.full_like(arr, -1)
        for old, new in mapping.items():
            new_arr[arr == old] = new
        if not keep_negative and np.any(arr < 0):
            # Put negative labels after valid labels if user really asked not to keep them.
            neg_values = sorted(map(int, np.unique(arr[arr < 0])))
            offset = len(mapping)
            for j, old in enumerate(neg_values):
                mapping[old] = offset + j
                new_arr[arr == old] = offset + j
        remapped.append(new_arr)
        maps.append(mapping)
    return remapped, maps


def infer_effective_lengths(labels: list[np.ndarray]) -> np.ndarray:
    """Infer each item's last real split level from rectangular hierarchy labels."""
    if not labels:
        raise ValueError("No label levels were loaded")

    n_items = len(labels[0])
    lengths = np.ones(n_items, dtype=np.int64)
    active = labels[0] >= 0

    for level_idx in range(len(labels) - 1):
        current = labels[level_idx]
        following = labels[level_idx + 1]

        # A refinement may divide a parent, but it may never merge two parents.
        valid = (current >= 0) & (following >= 0)
        for child in np.unique(following[valid]):
            parents = np.unique(current[valid & (following == child)])
            if len(parents) != 1:
                raise ValueError(
                    f"level_{level_idx + 2} cluster {int(child)} merges "
                    f"level_{level_idx + 1} parents {parents.tolist()}"
                )

        if np.any((current < 0) & (following >= 0)):
            item_id = int(np.where((current < 0) & (following >= 0))[0][0])
            raise ValueError(
                f"item_id={item_id} re-enters the hierarchy at level_{level_idx + 2}"
            )
        if np.any((current >= 0) & (following < 0)):
            item_id = int(np.where((current >= 0) & (following < 0))[0][0])
            raise ValueError(
                f"item_id={item_id} leaves the hierarchy at level_{level_idx + 2}"
            )

        split_at_transition = np.zeros(n_items, dtype=bool)
        for parent in np.unique(current[current >= 0]):
            members = current == parent
            children = np.unique(following[members & (following >= 0)])
            if len(children) > 1:
                split_at_transition[members] = True

        resumed = (~active) & split_at_transition
        if np.any(resumed):
            item_id = int(np.where(resumed)[0][0])
            raise ValueError(
                f"Stopped branch containing item_id={item_id} resumes splitting "
                f"at level_{level_idx + 2}"
            )

        continued = active & split_at_transition
        lengths[continued] = level_idx + 2
        active &= split_at_transition

    return lengths


def build_codes(
    labels: list[np.ndarray],
    prefixes: list[str],
    unclustered_strategy: str,
    append_unique_leaf: bool,
    unique_prefix: str,
    effective_lengths: np.ndarray | None = None,
) -> dict[str, list[str]]:
    """
    Convert level labels into token sequences.

    unclustered_strategy:
      - special: use a shared special id at every level (non-negative in
        variable-length mode so it cannot become downstream padding).
      - unique: use item id as the code for unclustered values.
      - error: fail if any -1 is present.
    append_unique_leaf:
      If True, append a final per-path rank token to make full SIDs unique.
    """
    if len(labels) == 0:
        raise ValueError("No label levels were loaded")
    if len(prefixes) < len(labels):
        raise ValueError(f"Need at least {len(labels)} token prefixes, got {len(prefixes)}")

    n_items = len(labels[0])
    variable_length = effective_lengths is not None
    if effective_lengths is None:
        effective_lengths = np.full(n_items, len(labels), dtype=np.int64)
    if effective_lengths.shape != (n_items,):
        raise ValueError(
            f"effective_lengths must have shape ({n_items},), got {effective_lengths.shape}"
        )
    if np.any((effective_lengths < 1) | (effective_lengths > len(labels))):
        raise ValueError("Every effective SID length must be between 1 and the number of levels")

    special_ids = []
    for arr in labels:
        valid = arr[arr >= 0]
        special_ids.append(int(valid.max()) + 1 if valid.size else 0)

    codes: dict[str, list[str]] = {}

    for item_id in range(n_items):
        item_tokens = []
        for level_idx, arr in enumerate(labels[: int(effective_lengths[item_id])]):
            cid = int(arr[item_id])
            if cid < 0:
                if unclustered_strategy == "error":
                    raise ValueError(f"item_id={item_id} has unclustered label {cid} at level {level_idx + 1}")
                if unclustered_strategy == "unique":
                    cid = item_id
                elif unclustered_strategy == "special":
                    cid = special_ids[level_idx] if variable_length else -1
                else:
                    raise ValueError(f"Unknown unclustered_strategy={unclustered_strategy!r}")
            item_tokens.append(prefixes[level_idx].format(cid))
        codes[str(item_id)] = item_tokens

    if append_unique_leaf:
        path_to_items: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for item_id, toks in codes.items():
            path_to_items[tuple(toks)].append(item_id)

        for toks, item_ids in path_to_items.items():
            if len(item_ids) == 1:
                # rank 0 still makes all items have same SID length.
                codes[item_ids[0]].append(unique_prefix.format(0))
            else:
                for rank, item_id in enumerate(sorted(item_ids, key=lambda x: int(x))):
                    codes[item_id].append(unique_prefix.format(rank))

    return codes


def collision_stats(codes: dict[str, list[str]]) -> dict[str, Any]:
    paths = [tuple(v) for v in codes.values()]
    c = Counter(paths)
    n = len(paths)
    n_unique = len(c)
    return {
        "num_items": n,
        "num_unique_sids": n_unique,
        "num_colliding_items": int(sum(v for v in c.values() if v > 1)),
        "collision_rate": float((n - n_unique) / n) if n else 0.0,
        "max_collision_group_size": int(max(c.values())) if c else 0,
    }


def load_interaction_frequencies(
    interactions_path: str,
    skip_n_last: int,
) -> Counter:
    """Count training interaction occurrences per item using the graph split policy."""
    if skip_n_last < 0:
        raise ValueError("--popularity_skip_n_last must be >= 0")
    with open(interactions_path, "r") as f:
        interactions = json.load(f)

    frequencies: Counter = Counter()
    for items in interactions.values():
        train_items = items if skip_n_last == 0 else items[:-skip_n_last]
        for item_id in train_items:
            frequencies[str(int(item_id))] += 1
    return frequencies


def popularity_length_stats(
    codes: dict[str, list[str]],
    frequencies: Counter,
    extreme_fraction: float,
) -> dict[str, Any]:
    """Match varlen_semantic_ids length/popularity statistic naming."""
    if not 0.0 < extreme_fraction < 0.5:
        raise ValueError("--popularity_extreme_fraction must be between 0 and 0.5")

    item_ids = sorted(codes, key=lambda item_id: int(item_id))
    lengths = np.asarray([len(codes[item_id]) for item_id in item_ids], dtype=np.int64)
    popularity = np.asarray(
        [int(frequencies.get(str(int(item_id)), 0)) for item_id in item_ids],
        dtype=np.float64,
    )

    mean_uniform = float(lengths.mean()) if lengths.size else float("nan")
    popularity_sum = float(popularity.sum())
    mean_unigram = (
        float(np.dot(lengths, popularity) / popularity_sum)
        if popularity_sum > 0.0
        else mean_uniform
    )

    grouped: dict[int, list[float]] = defaultdict(list)
    for length, pop in zip(lengths.tolist(), popularity.tolist()):
        grouped[int(length)].append(float(pop))

    head_mask = np.zeros(lengths.size, dtype=bool)
    tail_mask = np.zeros(lengths.size, dtype=bool)
    if lengths.size > 1:
        group_size = min(
            lengths.size // 2,
            max(1, int(np.ceil(lengths.size * extreme_fraction))),
        )
        numeric_item_ids = np.asarray([int(item_id) for item_id in item_ids], dtype=np.int64)
        popularity_order = np.lexsort((numeric_item_ids, -popularity))
        head_mask[popularity_order[:group_size]] = True
        tail_mask[popularity_order[-group_size:]] = True
    middle_mask = ~(head_mask | tail_mask)

    per_length = {}
    for sid_length, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        length_mask = lengths == sid_length
        count = int(length_mask.sum())
        count_head = int((length_mask & head_mask).sum())
        count_middle = int((length_mask & middle_mask).sum())
        count_longtail = int((length_mask & tail_mask).sum())
        per_length[str(sid_length)] = {
            "count": count,
            "pop_mean": float(array.mean()),
            "pop_median": float(np.median(array)),
            "pop_std": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
            "pop_min": float(array.min()),
            "pop_max": float(array.max()),
            "count_head": count_head,
            "count_middle": count_middle,
            "count_longtail": count_longtail,
            "pct_head": float(100.0 * count_head / count),
            "pct_middle": float(100.0 * count_middle / count),
            "pct_longtail": float(100.0 * count_longtail / count),
        }

    if lengths.size > 1:
        head_longtail = {
            "mean_length_head": float(lengths[head_mask].mean()),
            "mean_length_middle": (
                float(lengths[middle_mask].mean()) if middle_mask.any() else float("nan")
            ),
            "mean_length_longtail": float(lengths[tail_mask].mean()),
            "count_head": int(head_mask.sum()),
            "count_middle": int(middle_mask.sum()),
            "count_longtail": int(tail_mask.sum()),
        }
    else:
        head_longtail = {}

    return {
        "length": {
            "mean_uniform": mean_uniform,
            "mean_unigram": mean_unigram,
        },
        "popularity_length": {
            "pop_stats_by_length": per_length,
        },
        "head_longtail": head_longtail,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Generate GR SID mapping directly from graph-cluster hierarchy labels.")
    p.add_argument("--labels_path", required=True, help="Path to clustering labels JSON.")
    p.add_argument("--output_file", required=True, help="Where to save item_id -> token-list JSON.")
    p.add_argument("--label_group", default="coarse_to_fine_item_labels", help="Label group in clustering JSON.")
    p.add_argument("--levels", nargs="*", default=None, help="Levels to use, e.g. level_1 level_2. Default: all.")
    p.add_argument("--prefixes", nargs="*", default=None, help="Token format strings, e.g. '<a_{}>' '<b_{}>'.")
    p.add_argument("--reindex_contiguous", action="store_true", help="Remap cluster ids per level to compact 0..K-1 ids.")
    p.add_argument(
        "--unclustered_strategy",
        choices=["special", "unique", "error"],
        default="unique",
        help="How to handle label -1/unclustered items. Default gives each such item unique per-level ids.",
    )
    p.add_argument(
        "--append_unique_leaf",
        action="store_true",
        help="Append final per-path rank token so full SID is unique even if leaf clusters contain multiple items.",
    )
    p.add_argument(
        "--variable_length",
        action="store_true",
        help="Infer stopped recursive branches and omit repeated suffix levels from each SID.",
    )
    p.add_argument("--unique_prefix", default="<z_{}>", help="Format for appended uniqueness token.")
    p.add_argument("--stats_file", default=None, help="Optional path to save stats JSON. Default: output_file + '.stats.json'.")
    p.add_argument(
        "--interactions_path",
        default=None,
        help="Optional user-to-item-sequence JSON used to summarize item frequency by SID length.",
    )
    p.add_argument(
        "--popularity_skip_n_last",
        type=int,
        default=2,
        help="Interactions excluded from the end of every user sequence when computing frequency.",
    )
    p.add_argument(
        "--popularity_extreme_fraction",
        type=float,
        default=0.2,
        help="Fraction assigned to each popularity extreme: top head and bottom long-tail.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.variable_length and not args.append_unique_leaf:
        raise ValueError("--variable_length requires --append_unique_leaf as the terminal SID token")

    level_names, labels = load_cluster_label_levels(
        labels_path=args.labels_path,
        label_group=args.label_group,
        levels=args.levels,
    )

    remap_maps = None
    if args.reindex_contiguous:
        labels, remap_maps = make_contiguous_per_level(labels, keep_negative=True)

    prefixes = args.prefixes if args.prefixes is not None else make_default_prefixes(len(labels))
    effective_lengths = infer_effective_lengths(labels) if args.variable_length else None

    codes = build_codes(
        labels=labels,
        prefixes=prefixes,
        unclustered_strategy=args.unclustered_strategy,
        append_unique_leaf=args.append_unique_leaf,
        unique_prefix=args.unique_prefix,
        effective_lengths=effective_lengths,
    )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(codes, f)

    stats = collision_stats(codes)
    stats.update(
        {
            "labels_path": args.labels_path,
            "label_group": args.label_group,
            "levels": level_names,
            "num_levels": len(level_names),
            "append_unique_leaf": bool(args.append_unique_leaf),
            "variable_length": bool(args.variable_length),
            "unclustered_strategy": args.unclustered_strategy,
            "level_cluster_counts_excluding_minus1": {
                name: int(len(set(map(int, arr[arr >= 0])))) for name, arr in zip(level_names, labels)
            },
            "level_unclustered_counts": {
                name: int((arr < 0).sum()) for name, arr in zip(level_names, labels)
            },
        }
    )
    if args.interactions_path is not None:
        frequencies = load_interaction_frequencies(
            interactions_path=args.interactions_path,
            skip_n_last=args.popularity_skip_n_last,
        )
        stats.update(popularity_length_stats(
            codes=codes,
            frequencies=frequencies,
            extreme_fraction=args.popularity_extreme_fraction,
        ))
    sid_lengths = [len(tokens) for tokens in codes.values()]
    length_counts = Counter(sid_lengths)
    stats.update(
        {
            "sid_length_min": int(min(sid_lengths)) if sid_lengths else 0,
            "sid_length_max": int(max(sid_lengths)) if sid_lengths else 0,
            "sid_length_mean": float(np.mean(sid_lengths)) if sid_lengths else 0.0,
            "sid_length_histogram": {str(length): int(count) for length, count in sorted(length_counts.items())},
        }
    )
    if remap_maps is not None:
        stats["reindex_contiguous"] = True
        stats["remap_sizes"] = {name: len(mp) for name, mp in zip(level_names, remap_maps)}
    else:
        stats["reindex_contiguous"] = False

    stats_file = args.stats_file or (args.output_file + ".stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved SID mapping to: {args.output_file}")
    print(f"Saved stats to: {stats_file}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
