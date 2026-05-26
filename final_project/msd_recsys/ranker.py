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
    keys: list[tuple[str, str]],
    *,
    top_k: int = 500,
    exclude_owned: dict[str, set] | None = None,
) -> dict[str, list[str]]:
    """Score with the ranker, group by user, return top-K item_ids per user.

    Optionally excludes items the user already owns (default behavior in MSD
    where you must not recommend already-listened songs).
    """
    scores = model.predict(X)
    by_user: dict[str, list[tuple[str, float]]] = {}
    for (u, it), s in zip(keys, scores):
        by_user.setdefault(u, []).append((it, float(s)))

    out: dict[str, list[str]] = {}
    for u, lst in by_user.items():
        lst.sort(key=lambda x: -x[1])
        owned = exclude_owned.get(u, set()) if exclude_owned else set()
        picks, seen = [], set()
        for it, _ in lst:
            if it in owned or it in seen:
                continue
            picks.append(it)
            seen.add(it)
            if len(picks) == top_k:
                break
        out[u] = picks
    return out
