"""Evaluation metrics for top-K recommendation.

Primary: MAP@500 (matches the MSD Challenge leaderboard).
Secondary: Recall@K, NDCG@K, catalog coverage, intra-list diversity.

All metrics accept the same shape:
    ranked_by_user: dict[user_id, list[item_id]]  (already sorted, length <= K)
    truth_by_user:  dict[user_id, set[item_id]]
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def average_precision_at_k(ranked: list, truth: set, k: int) -> float:
    """AP@k for a single user. Denominator = min(k, |truth|)."""
    if not truth:
        return 0.0
    score = 0.0
    hits = 0
    for i, item in enumerate(ranked[:k]):
        if item in truth:
            hits += 1
            score += hits / (i + 1)
    return score / min(k, len(truth))


def recall_at_k(ranked: list, truth: set, k: int) -> float:
    if not truth:
        return 0.0
    return sum(1 for it in ranked[:k] if it in truth) / len(truth)


def ndcg_at_k(ranked: list, truth: set, k: int) -> float:
    if not truth:
        return 0.0
    gains = [1.0 if it in truth else 0.0 for it in ranked[:k]]
    if not any(gains):
        return 0.0
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(truth))))
    return dcg / ideal


def map_at_k(ranked_by_user: dict, truth_by_user: dict, k: int = 500) -> float:
    """Mean AP@k across users that have ground truth."""
    aps = [
        average_precision_at_k(ranked_by_user.get(u, []), t, k)
        for u, t in truth_by_user.items()
        if t
    ]
    return float(np.mean(aps)) if aps else 0.0


def mean_recall_at_k(ranked_by_user: dict, truth_by_user: dict, k: int = 500) -> float:
    rs = [recall_at_k(ranked_by_user.get(u, []), t, k) for u, t in truth_by_user.items() if t]
    return float(np.mean(rs)) if rs else 0.0


def mean_ndcg_at_k(ranked_by_user: dict, truth_by_user: dict, k: int = 500) -> float:
    ns = [ndcg_at_k(ranked_by_user.get(u, []), t, k) for u, t in truth_by_user.items() if t]
    return float(np.mean(ns)) if ns else 0.0


# ---------------------------------------------------------------------------
# Diversity / coverage
# ---------------------------------------------------------------------------

def catalog_coverage(ranked_by_user: dict, catalog_size: int, k: int = 500) -> float:
    """Fraction of the catalog recommended to >=1 user in their top-K."""
    seen = set()
    for ranked in ranked_by_user.values():
        seen.update(ranked[:k])
    return len(seen) / catalog_size if catalog_size else 0.0


def intra_list_diversity(
    ranked_by_user: dict,
    item_to_code: dict,
    k: int = 500,
) -> float:
    """Mean pairwise Hamming-style distance between item codes within a user's top-K.

    `item_to_code` maps item_id -> tuple of codes (e.g., semantic-ID levels).
    Distance between two items = fraction of code positions that differ.
    """
    def list_diversity(items):
        codes = [item_to_code.get(it) for it in items[:k]]
        codes = [c for c in codes if c is not None]
        if len(codes) < 2:
            return 0.0
        n_pos = len(codes[0])
        pairs = 0
        total = 0.0
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                diff = sum(1 for a, b in zip(codes[i], codes[j]) if a != b)
                total += diff / n_pos
                pairs += 1
        return total / pairs if pairs else 0.0

    scores = [list_diversity(r) for r in ranked_by_user.values()]
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Bucketed reporting
# ---------------------------------------------------------------------------

def popularity_tiers(item_popularity: pd.Series, tiers: tuple = (0.33, 0.66)) -> dict[str, str]:
    """Bucket items into head/torso/tail by cumulative popularity share.

    Items sorted by popularity desc; the most popular items that account for
    the first `tiers[0]` fraction of total listens are 'head', the next chunk
    up to `tiers[1]` are 'torso', the rest are 'tail'.
    """
    sorted_pop = item_popularity.sort_values(ascending=False)
    total = sorted_pop.sum()
    cum = sorted_pop.cumsum() / total
    out = {}
    for item_id, frac in cum.items():
        if frac <= tiers[0]:
            out[item_id] = "head"
        elif frac <= tiers[1]:
            out[item_id] = "torso"
        else:
            out[item_id] = "tail"
    return out


def metrics_by_bucket(
    ranked_by_user: dict,
    truth_by_user: dict,
    user_bucket: dict[str, str],
    *,
    k: int = 500,
) -> pd.DataFrame:
    """Compute MAP@k / Recall@k / NDCG@k per user bucket. Returns a DataFrame."""
    by_bucket = defaultdict(lambda: {"users": 0, "map": [], "recall": [], "ndcg": []})
    for u, t in truth_by_user.items():
        if not t:
            continue
        bucket = user_bucket.get(u, "unknown")
        r = ranked_by_user.get(u, [])
        by_bucket[bucket]["users"] += 1
        by_bucket[bucket]["map"].append(average_precision_at_k(r, t, k))
        by_bucket[bucket]["recall"].append(recall_at_k(r, t, k))
        by_bucket[bucket]["ndcg"].append(ndcg_at_k(r, t, k))

    rows = []
    for bucket, vals in by_bucket.items():
        rows.append({
            "bucket":      bucket,
            "users":       vals["users"],
            f"MAP@{k}":    float(np.mean(vals["map"])),
            f"Recall@{k}": float(np.mean(vals["recall"])),
            f"NDCG@{k}":   float(np.mean(vals["ndcg"])),
        })
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def tail_focused_map(
    ranked_by_user: dict,
    truth_by_user: dict,
    item_tier: dict[str, str],
    *,
    k: int = 500,
    tier: str = "tail",
) -> float:
    """MAP@k restricted to ground-truth items in a popularity tier.

    For each user, keep only the ground-truth items in `tier`; compute AP@k
    against that filtered truth. Tells us whether the model surfaces long-tail
    songs the user actually played, separately from head-hit performance.
    """
    aps = []
    for u, full_truth in truth_by_user.items():
        tier_truth = {it for it in full_truth if item_tier.get(it) == tier}
        if not tier_truth:
            continue
        aps.append(average_precision_at_k(ranked_by_user.get(u, []), tier_truth, k))
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Ceiling analysis (the framing that worked in assignment 3)
# ---------------------------------------------------------------------------

def retrieval_recall(candidates_by_user: dict, truth_by_user: dict) -> float:
    """Fraction of held-out items that appear ANYWHERE in the candidate pool.

    This is the hard upper bound on MAP@K and Recall@K — no ranker can recover
    an item the retriever didn't surface.
    """
    hits = total = 0
    for u, truth in truth_by_user.items():
        if not truth:
            continue
        cands = candidates_by_user.get(u, set())
        if not isinstance(cands, set):
            cands = set(cands)
        hits += len(truth & cands)
        total += len(truth)
    return hits / total if total else 0.0


def capture_efficiency(achieved_metric: float, ceiling_recall: float) -> float:
    """What fraction of the achievable score did the ranker actually capture?

    achieved / ceiling. Use achieved_metric=MAP@K and ceiling_recall=retrieval
    Recall@K_pool. Below ~30% means the ranker is the bottleneck; above means
    retrieval is.
    """
    if ceiling_recall <= 0:
        return 0.0
    return achieved_metric / ceiling_recall
