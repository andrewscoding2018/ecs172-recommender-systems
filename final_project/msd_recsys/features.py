"""Stage 2 — Feature engineering for the ranker.

For each (user, candidate) pair surfaced by Stage 1, build a row of features.
The ranker uses these to re-rank candidates into the final top-500.

Features grouped per the assignment taxonomy:
  USER-side        - hist_size_log, distinct_artists_log, avg_artist_familiarity
  ITEM-side        - pop_log, artist_familiarity, artist_hotttnesss, decade
  INTERACTION-side - als_score, semantic_score, shared_artists, semantic_match_count
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_SPECS = [
    # name                          group         description
    ("hist_size_log",               "user",        "log1p(# listens in user's history) — engagement proxy"),
    ("distinct_artists_log",        "user",        "log1p(# distinct artists in history) — taste breadth"),
    ("avg_artist_familiarity",      "user",        "mean artist_familiarity across history items — preference for known artists"),
    ("pop_log",                     "item",        "log1p(item popularity in train) — popularity prior"),
    ("artist_familiarity",          "item",        "candidate artist's Echo Nest familiarity score (NaN-safe)"),
    ("artist_hotttnesss",           "item",        "candidate artist's hotttnesss score (NaN-safe)"),
    ("decade",                      "item",        "release decade bucket; -1 for unknown"),
    ("als_score",                   "interaction", "Stage-1 ALS retrieval score for this candidate"),
    ("semantic_score",              "interaction", "Stage-1 semantic-ID match score"),
    ("shared_artists",              "interaction", "1 if candidate's artist is in user's history artists, else 0"),
    ("semantic_match_count",        "interaction", "# code positions where candidate matches user's most-frequent code at that position"),
]
FEATURE_NAMES = [f[0] for f in FEATURE_SPECS]


# ---------------------------------------------------------------------------
# Per-item / per-user feature tables (precompute once, look up per pair)
# ---------------------------------------------------------------------------

@dataclass
class ItemFeatureTable:
    """All item-side feature values indexed by item_id."""
    item_ids: np.ndarray            # (n_items,)
    pop_log: np.ndarray             # (n_items,)
    artist_familiarity: np.ndarray  # (n_items,) NaN-safe
    artist_hotttnesss: np.ndarray
    decade: np.ndarray              # int, -1 if missing
    artist_id: np.ndarray           # for shared_artists feature
    semantic_codes: np.ndarray | None = None  # (n_items, n_code_positions), or None

    @classmethod
    def build(
        cls,
        metadata: pd.DataFrame,
        item_popularity: pd.Series,
        semantic_codes: dict | None = None,
    ) -> "ItemFeatureTable":
        md = metadata.copy()
        if "decade" not in md.columns:
            md["decade"] = (md["year"] // 10 * 10).fillna(-1).astype(int)
        item_ids = md["song_id"].values
        pop = item_popularity.reindex(item_ids).fillna(0).values
        codes = None
        if semantic_codes:
            sample_key = next(iter(semantic_codes))
            n_pos = len(semantic_codes[sample_key])
            codes = np.full((len(item_ids), n_pos), -1, dtype=np.int32)
            for i, iid in enumerate(item_ids):
                c = semantic_codes.get(iid)
                if c is not None:
                    codes[i] = c
        return cls(
            item_ids=item_ids,
            pop_log=np.log1p(pop).astype(np.float32),
            artist_familiarity=md["artist_familiarity"].fillna(md["artist_familiarity"].median()).values.astype(np.float32),
            artist_hotttnesss=md["artist_hotttnesss"].fillna(md["artist_hotttnesss"].median()).values.astype(np.float32),
            decade=md["decade"].values.astype(np.int32),
            artist_id=md["artist_id"].values,
            semantic_codes=codes,
        )


@dataclass
class UserContext:
    """Per-user context computed once per user (not per candidate)."""
    history_size: int
    distinct_artists: int
    avg_artist_familiarity: float
    history_artists: set
    # Most-frequent code per code position in the user's history, for semantic_match_count
    mode_codes: np.ndarray | None  # (n_code_positions,) or None


def compute_user_context(
    user_id: str,
    history: list[str],
    item_table: ItemFeatureTable,
    item_to_ix: dict,
) -> UserContext:
    ixs = [item_to_ix[h] for h in history if h in item_to_ix]
    if not ixs:
        return UserContext(0, 0, 0.0, set(), None)
    fam = item_table.artist_familiarity[ixs]
    artists = set(item_table.artist_id[ixs])
    mode_codes = None
    if item_table.semantic_codes is not None:
        codes = item_table.semantic_codes[ixs]  # (history_size, n_positions)
        n_pos = codes.shape[1]
        mode_codes = np.zeros(n_pos, dtype=np.int32)
        for p in range(n_pos):
            vals, counts = np.unique(codes[:, p], return_counts=True)
            mode_codes[p] = vals[np.argmax(counts)]
    return UserContext(
        history_size=len(ixs),
        distinct_artists=len(artists),
        avg_artist_familiarity=float(np.mean(fam)),
        history_artists=artists,
        mode_codes=mode_codes,
    )


# ---------------------------------------------------------------------------
# Build the feature matrix for the ranker
# ---------------------------------------------------------------------------

def build_feature_rows(
    users: list[str],
    histories: list[list[str]],
    candidate_dicts: list[dict[str, list[float]]],
    item_table: ItemFeatureTable,
    truth_by_user: dict[str, set] | None = None,
):
    """Flatten (user, candidate) pairs into X / y / keys / group_sizes.

    Returns:
        X: (n_pairs, n_features) float32
        y: (n_pairs,) int8 with labels if truth_by_user given, else None
        keys: list of (user_id, item_id) tuples in row order
        group_sizes: list of candidates-per-user (for LightGBM group= arg)
    """
    item_to_ix = {it: i for i, it in enumerate(item_table.item_ids)}
    rows, labels, keys, groups = [], [], [], []

    for u, hist, cands in zip(users, histories, candidate_dicts):
        if not cands:
            groups.append(0)
            continue
        ctx = compute_user_context(u, hist, item_table, item_to_ix)
        truth = truth_by_user.get(u, set()) if truth_by_user is not None else None
        n_added = 0
        for it, (als_s, sid_s) in cands.items():
            ix = item_to_ix.get(it)
            if ix is None:
                continue
            # semantic_match_count: # code positions where candidate matches user's mode code
            sem_match = 0
            if ctx.mode_codes is not None and item_table.semantic_codes is not None:
                sem_match = int(np.sum(item_table.semantic_codes[ix] == ctx.mode_codes))
            shared = 1 if item_table.artist_id[ix] in ctx.history_artists else 0
            rows.append([
                np.log1p(ctx.history_size),                   # hist_size_log
                np.log1p(ctx.distinct_artists),               # distinct_artists_log
                ctx.avg_artist_familiarity,                   # avg_artist_familiarity
                item_table.pop_log[ix],                       # pop_log
                item_table.artist_familiarity[ix],            # artist_familiarity
                item_table.artist_hotttnesss[ix],             # artist_hotttnesss
                float(item_table.decade[ix]),                 # decade
                als_s,                                        # als_score
                sid_s,                                        # semantic_score
                float(shared),                                # shared_artists
                float(sem_match),                             # semantic_match_count
            ])
            if truth is not None:
                labels.append(1 if it in truth else 0)
            keys.append((u, it))
            n_added += 1
        groups.append(n_added)

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8) if truth_by_user is not None else None
    return X, y, keys, groups
