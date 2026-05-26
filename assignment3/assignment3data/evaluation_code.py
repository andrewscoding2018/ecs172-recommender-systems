"""
Evaluation for Assignment 3: Hybrid Recommender.

Metrics:
  - Recall@10
  - NDCG@10  (binary relevance, IDCG normalized to the size of the held-out set)

Final leaderboard score = 0.5 * (Recall@10 + NDCG@10).

Also exposes `validate_submission` for students to check format locally.
"""

import math
import pandas as pd

K = 10
RANK_COLS = [f"rank_{i}" for i in range(1, K + 1)]


def validate_submission(submission_csv, test_users_csv, item_metadata_csv):
    """
    Raises AssertionError on the first formatting problem found.
    Safe to call on a partially built submission.
    """
    sub = pd.read_csv(submission_csv)
    test_users = pd.read_csv(test_users_csv)
    items = pd.read_csv(item_metadata_csv)

    expected_cols = ["user_id"] + RANK_COLS
    assert list(sub.columns) == expected_cols, (
        f"Columns must be exactly {expected_cols}, got {list(sub.columns)}"
    )

    sub_users = set(sub["user_id"])
    expected_users = set(test_users["user_id"])
    assert sub_users == expected_users, (
        f"Submission users do not match test_users.csv. "
        f"Missing={len(expected_users - sub_users)}, Extra={len(sub_users - expected_users)}"
    )
    assert len(sub) == len(expected_users), "Each user must appear exactly once."

    valid_items = set(items["item_id"])
    for _, row in sub.iterrows():
        preds = [row[c] for c in RANK_COLS]
        assert len(set(preds)) == K, f"User {row['user_id']}: duplicate items in row."
        for p in preds:
            assert p in valid_items, f"User {row['user_id']}: unknown item_id {p}."

    print(f"Submission OK: {len(sub)} users x {K} items.")


def _ndcg_at_k(ranked_items, relevant_set, k=K):
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
    n_rel = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked_items, relevant_set, k=K):
    if not relevant_set:
        return 0.0
    hits = sum(1 for it in ranked_items[:k] if it in relevant_set)
    return hits / len(relevant_set)


def evaluate(submission_csv, ground_truth_csv, train_csv=None, verbose=True):
    """
    Compute Recall@10 and NDCG@10.

    If `train_csv` is provided, items the user already owns in train are
    silently dropped from the predicted list before scoring (they cannot
    be in the held-out set anyway, so they only waste slots).
    Returns (recall, ndcg, mean_score).
    """
    sub = pd.read_csv(submission_csv)
    gt = pd.read_csv(ground_truth_csv)
    gt_dict = gt.groupby("user_id")["item_id"].apply(set).to_dict()

    seen = {}
    if train_csv is not None:
        train = pd.read_csv(train_csv)
        seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

    recalls, ndcgs = [], []
    for _, row in sub.iterrows():
        uid = row["user_id"]
        preds = [row[c] for c in RANK_COLS]
        if uid in seen:
            preds = [p for p in preds if p not in seen[uid]]
        truth = gt_dict.get(uid, set())
        recalls.append(_recall_at_k(preds, truth, K))
        ndcgs.append(_ndcg_at_k(preds, truth, K))

    recall = sum(recalls) / len(recalls)
    ndcg = sum(ndcgs) / len(ndcgs)
    mean_score = 0.5 * (recall + ndcg)

    if verbose:
        print(f"Users scored: {len(recalls)}")
        print(f"Recall@{K}: {recall:.4f}")
        print(f"NDCG@{K}:   {ndcg:.4f}")
        print(f"Mean:       {mean_score:.4f}")

    return recall, ndcg, mean_score


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python evaluation_code.py submission.csv test_ground_truth.csv [train.csv]")
        sys.exit(1)
    train_csv = sys.argv[3] if len(sys.argv) > 3 else None
    evaluate(sys.argv[1], sys.argv[2], train_csv=train_csv)
