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

from .retrieval import _mb

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
    candidates: pd.DataFrame,
    item_table: ItemFeatureTable,
    truth_by_user: dict[str, set] | None = None,
    verbose: bool = False,
):
    """Build a vectorized feature matrix from the long-format candidate DataFrame.

    Replaces the old per-user, per-candidate Python loop. Now there's one
    Python pass over users (to precompute per-user context arrays), then
    everything else is NumPy/pandas C-level.

    Args:
        users: list[user_id] aligned with candidates.user_idx (0..n_users-1).
        histories: list[list[item_id]] aligned with users.
        candidates: long-format DataFrame from retrieval.build_candidate_pool —
            columns (user_idx, item_ix, als_score, sid_score).
        item_table: precomputed ItemFeatureTable indexed by item_ix.
        truth_by_user: optional {user_id: set[item_id]} for label generation.
        verbose: print memory diagnostics for the output arrays.

    Returns:
        X: (n_pairs, n_features) float32 feature matrix.
        y: (n_pairs,) int8 labels (1 if pair is in truth, else 0), or None.
        keys: pd.DataFrame[user_idx, item_ix] aligned with X rows (for ranker).
        groups: list[int] candidates-per-user count, ordered by user_idx for
            LightGBM's `group` arg (one entry per user, zeros for empty users).
    """
    n_users = len(users)
    item_to_ix = {it: i for i, it in enumerate(item_table.item_ids)}

    # ------------------------------------------------------------------
    # Phase 1: per-user context, vectorized into aligned arrays.
    # One Python pass over users; everything later uses these arrays.
    # ------------------------------------------------------------------
    user_hist_size = np.zeros(n_users, dtype=np.int32)
    user_distinct_artists = np.zeros(n_users, dtype=np.int32)
    user_avg_familiarity = np.zeros(n_users, dtype=np.float32)
    user_history_artists: list[set] = [set() for _ in range(n_users)]
    user_mode_codes = None
    if item_table.semantic_codes is not None:
        n_pos = item_table.semantic_codes.shape[1]
        user_mode_codes = np.zeros((n_users, n_pos), dtype=np.int32)

    for ui, hist in enumerate(histories):
        ixs = [item_to_ix[h] for h in hist if h in item_to_ix]
        if not ixs:
            continue
        user_hist_size[ui] = len(ixs)
        user_avg_familiarity[ui] = item_table.artist_familiarity[ixs].mean()
        artists = set(item_table.artist_id[ixs].tolist())
        user_distinct_artists[ui] = len(artists)
        user_history_artists[ui] = artists
        if user_mode_codes is not None:
            codes = item_table.semantic_codes[ixs]   # (history_size, n_positions)
            for p in range(codes.shape[1]):
                vals, counts = np.unique(codes[:, p], return_counts=True)
                user_mode_codes[ui, p] = vals[np.argmax(counts)]

    # ------------------------------------------------------------------
    # Phase 2: vectorize features over candidate rows.
    # Sort by user_idx so groups come out aligned for LightGBM.
    # ------------------------------------------------------------------
    candidates = candidates.sort_values("user_idx", kind="stable").reset_index(drop=True)
    user_idx = candidates["user_idx"].values
    item_ix = candidates["item_ix"].values
    als_score = candidates["als_score"].values
    sid_score = candidates["sid_score"].values
    n_pairs = len(candidates)

    # User-side features: gather precomputed arrays at user_idx positions.
    hist_size_log = np.log1p(user_hist_size[user_idx]).astype(np.float32)
    distinct_artists_log = np.log1p(user_distinct_artists[user_idx]).astype(np.float32)
    avg_artist_familiarity = user_avg_familiarity[user_idx]

    # Item-side features: gather at item_ix positions.
    pop_log = item_table.pop_log[item_ix]
    artist_familiarity = item_table.artist_familiarity[item_ix]
    artist_hotttnesss = item_table.artist_hotttnesss[item_ix]
    decade = item_table.decade[item_ix].astype(np.float32)

    # Interaction: shared_artists (1 if candidate's artist is in user's history).
    # Set-membership per row — kept as a NumPy loop since sets don't vectorize.
    # Still much faster than the original per-feature Python construction.
    item_artists = item_table.artist_id[item_ix]
    shared_artists = np.zeros(n_pairs, dtype=np.float32)
    for k in range(n_pairs):
        if item_artists[k] in user_history_artists[user_idx[k]]:
            shared_artists[k] = 1.0

    # Interaction: semantic_match_count via fully-vectorized broadcasting.
    sem_match = np.zeros(n_pairs, dtype=np.float32)
    if user_mode_codes is not None and item_table.semantic_codes is not None:
        candidate_codes = item_table.semantic_codes[item_ix]   # (n_pairs, n_pos)
        user_codes_per_pair = user_mode_codes[user_idx]        # (n_pairs, n_pos)
        sem_match = (candidate_codes == user_codes_per_pair).sum(axis=1).astype(np.float32)

    # Stack into final (n_pairs, n_features) matrix.
    X = np.column_stack([
        hist_size_log,
        distinct_artists_log,
        avg_artist_familiarity,
        pop_log,
        artist_familiarity,
        artist_hotttnesss,
        decade,
        als_score.astype(np.float32),
        sid_score.astype(np.float32),
        shared_artists,
        sem_match,
    ]).astype(np.float32)

    # Labels: vectorized hit-check via building (user_id, item_id) tuple lookups.
    y = None
    if truth_by_user is not None:
        users_arr = np.asarray(users, dtype=object)
        pair_user_ids = users_arr[user_idx]
        pair_item_ids = item_table.item_ids[item_ix]
        labels = np.zeros(n_pairs, dtype=np.int8)
        # We loop here because dict-set lookups don't vectorize. Still
        # dramatically faster than the original because no other Python is
        # happening per pair.
        for k in range(n_pairs):
            t = truth_by_user.get(pair_user_ids[k])
            if t and pair_item_ids[k] in t:
                labels[k] = 1
        y = labels

    # Group sizes for LightGBM. groupby gives the count per user_idx; reindex
    # ensures every user_idx 0..n_users-1 has an entry (zero if no candidates).
    groups = (
        candidates.groupby("user_idx", sort=True)
        .size()
        .reindex(range(n_users), fill_value=0)
        .astype(int)
        .tolist()
    )

    # Keys: same shape as X, but cheap (int32 columns, not strings).
    keys = candidates[["user_idx", "item_ix"]].copy()

    if verbose:
        print(f"[features] X {X.shape} ({_mb(X):.1f} MB)  "
              f"y {y.shape if y is not None else None}  "
              f"keys {keys.shape} ({_mb(keys):.1f} MB)  "
              f"pairs={n_pairs:,}")

    return X, y, keys, groups
