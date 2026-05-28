"""Stage 2 — Ranker. LightGBM lambdarank since MAP@500 is rank-sensitive.

Why lambdarank over a regular classifier:
  - MAP@K is a list-wise metric; pairwise/list-wise objectives optimize it more
    directly than independent (user, item) binary classification.
  - LightGBM's lambdarank uses the `group` argument (one int per query = # candidates
    for that user). features.build_feature_rows returns the group sizes for you.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .retrieval import _mb

# lightgbm needs libomp.dylib on macOS; be tolerant of native-library load failures
# so the rest of the library still imports (LightGBM is only needed at ranker-train time).
try:
    import lightgbm as lgb
    _HAS_LGB = True
    _LGB_ERR: Exception | None = None
except Exception as e:
    lgb = None  # type: ignore
    _HAS_LGB = False
    _LGB_ERR = e


def train_lambdarank(
    X: np.ndarray,
    y: np.ndarray,
    group_sizes: list[int],
    *,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    n_estimators: int = 300,
    max_position: int = 500,
    eval_set: tuple | None = None,
    eval_group: list[int] | None = None,
    early_stopping_rounds: int | None = 20,
    random_state: int = 42,
    verbose: int = 50,
) -> Any:
    """Train a LightGBM LGBMRanker with lambdarank objective.

    Args:
        X / y / group_sizes: from features.build_feature_rows.
        eval_set / eval_group: optional (X_val, y_val) and matching group sizes
                                for early stopping.
        max_position: LightGBM's "ndcg_eval_at" position; set to your top-K target.

    Drop zero-sized groups (users with no candidates) before calling.
    """
    if not _HAS_LGB:
        raise ImportError(
            f"lightgbm not available: {_LGB_ERR!r}. "
            "On macOS run `brew install libomp`. On Colab this works out of the box."
        )
    if any(g == 0 for g in group_sizes):
        raise ValueError(
            "Group sizes contain zeros — filter out users with no candidates before "
            "calling train_lambdarank (otherwise LightGBM will misalign rows)."
        )

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="map",
        eval_at=[10, 100, max_position],
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        verbose=-1,
    )
    fit_kwargs: dict[str, Any] = {"group": group_sizes}
    if eval_set is not None and eval_group is not None:
        fit_kwargs["eval_set"] = [eval_set]
        fit_kwargs["eval_group"] = [eval_group]
        if early_stopping_rounds:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(period=verbose),
            ]
    model.fit(X, y, **fit_kwargs)
    return model


def rank_candidates(
    model: Any,
    X: np.ndarray,
    keys: pd.DataFrame,
    *,
    users: list[str],
    item_ids: np.ndarray,
    top_k: int = 500,
    exclude_owned: dict[str, set] | None = None,
    verbose: bool = False,
) -> dict[str, list[str]]:
    """Score with the ranker, group by user, return top-K item_ids per user.

    Now expects DataFrame keys (user_idx, item_ix) from build_feature_rows,
    not the old list of (user_id, item_id) tuples. The int-indexed format is
    ~10x smaller in memory.

    Args:
        model:         trained ranker (lightgbm).
        X:             (n_pairs, n_features) feature matrix.
        keys:          pd.DataFrame[user_idx, item_ix] aligned with X rows.
        users:         list[user_id] indexed by user_idx for the final dict keys.
        item_ids:      np.ndarray mapping item_ix -> song_id (e.g., item_table.item_ids).
        top_k:         number of picks per user.
        exclude_owned: optional {user_id: set[song_id]} of items to skip.
        verbose:       print scoring + memory diagnostics.
    """
    scores = model.predict(X).astype(np.float32)

    # Build a single working DataFrame: (user_idx, item_ix, score). Sort by
    # user then score-desc — groupby on a sorted column is fast.
    df = keys.copy()
    df["score"] = scores
    df = df.sort_values(["user_idx", "score"], ascending=[True, False], kind="stable")

    if verbose:
        print(f"[rank_candidates] working frame: {len(df):,} rows ({_mb(df):.1f} MB)")

    # Per-user top-K with optional owned-set exclusion. The inner loop runs
    # in C (groupby.apply on a sorted frame is fast), but the dict-of-strings
    # output is the materialization the rest of the eval pipeline expects.
    item_ids_arr = np.asarray(item_ids, dtype=object)
    out: dict[str, list[str]] = {}
    for user_idx, grp in df.groupby("user_idx", sort=True):
        user_id = users[int(user_idx)]
        owned = exclude_owned.get(user_id, set()) if exclude_owned else set()
        picks: list[str] = []
        for ix in grp["item_ix"].values:
            song_id = item_ids_arr[ix]
            if song_id in owned:
                continue
            picks.append(song_id)
            if len(picks) == top_k:
                break
        out[user_id] = picks
    return out
