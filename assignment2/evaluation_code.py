"""
ECS 172 Assignment 2 — RMSE evaluation helper.

This is the same scorer the TAs will run on your submission. You can use it
locally on a validation split you carve out of train.csv.

Usage
-----
    from evaluation_code import rmse, load_predictions

    y_true = pd.read_csv("my_val_ground_truth.csv")   # user_id, item_id, rating
    y_pred = pd.read_csv("my_val_predictions.csv")    # user_id, item_id, predicted_rating
    print("RMSE:", rmse(y_true, y_pred))

Rules the scorer enforces
-------------------------
  * Predictions must cover every (user_id, item_id) pair in the target file.
  * Extra rows are ignored.
  * Predicted ratings are clipped to [1.0, 5.0] before RMSE is computed
    (so you cannot game the metric with out-of-range values).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

RATING_MIN = 1.0
RATING_MAX = 5.0


def rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Compute clipped RMSE between ground-truth ratings and predictions."""
    required_true = {"user_id", "item_id", "rating"}
    required_pred = {"user_id", "item_id", "predicted_rating"}
    if not required_true.issubset(y_true.columns):
        raise ValueError(f"Ground truth must have columns {required_true}")
    if not required_pred.issubset(y_pred.columns):
        raise ValueError(f"Predictions must have columns {required_pred}")

    merged = y_true.merge(
        y_pred[["user_id", "item_id", "predicted_rating"]],
        on=["user_id", "item_id"],
        how="left",
    )
    missing = merged["predicted_rating"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} target (user, item) pairs have no prediction. "
            "Make sure your submission covers every row in test_pairs.csv."
        )
    yhat = merged["predicted_rating"].clip(RATING_MIN, RATING_MAX).to_numpy()
    y = merged["rating"].to_numpy()
    return math.sqrt(float(np.mean((yhat - y) ** 2)))


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"user_id", "item_id", "predicted_rating"}.issubset(df.columns):
        raise ValueError(
            "Predictions CSV must have columns: user_id, item_id, predicted_rating"
        )
    return df
