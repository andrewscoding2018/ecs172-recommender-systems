"""
ECS 172 Assignment 2 — Global-mean baseline.

This script is the minimum viable submission: predict the training global
mean rating for every (user, item) pair in test_pairs.csv. Use it to verify
that your submission format is correct, then replace the `predict` function
with your own method.

Run:
    python baseline_global_mean.py
produces `predictions_global_mean.csv` in the current directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "train.csv"
TEST_PAIRS_PATH = HERE / "test_pairs.csv"
OUT_PATH = HERE / "predictions_global_mean.csv"


def predict(train: pd.DataFrame, test_pairs: pd.DataFrame) -> pd.DataFrame:
    """Replace this function with your own model."""
    global_mean = float(train["rating"].mean())
    out = test_pairs[["user_id", "item_id"]].copy()
    out["predicted_rating"] = np.clip(global_mean, 1.0, 5.0)
    return out


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test_pairs = pd.read_csv(TEST_PAIRS_PATH)
    preds = predict(train, test_pairs)
    preds.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(preds):,} predictions to {OUT_PATH.name}")


if __name__ == "__main__":
    main()
