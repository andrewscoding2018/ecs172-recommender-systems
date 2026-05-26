"""
Assignment 3 — Multi-Stage Recommender (skeleton).

End-to-end pipeline wired together so we have a working submission, then iterate.

Stages:
  1. Load data + quick stats
  2. Local validation split that mirrors the leaderboard (last-5 held out per user)
  3. Retrieval (hybrid):
       - Item-CF: item-item cosine on co-ownership
       - Content: TF-IDF over `tags | genres | specs`, user profile = mean of history
       Union top-K from each route into a candidate pool.
  4. Recall@K of the retrieval pool on the validation split
  5. Ranker: engineered features per (user, candidate), HistGradientBoostingClassifier
  6. Recall@10 / NDCG@10 on a held-out slice of validation users
  7. Retrain retrieval on full train.csv, generate + validate submission.csv

Caveat: retrieval gets trained on `train_inner` for ranker training, then retrained
on full `train` for submission. Score distributions should be close enough for the
ranker to generalize; a nested split would tighten this.

Run:
    uv run python pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SEED = 42
DATA_DIR = Path(__file__).parent / "Assignment_3___Multi_stage_Recommendation_export"
SUBMISSION_PATH = Path(__file__).parent / "submission.csv"

TOPK_CF = 150
TOPK_CONTENT = 150
KEEP_LAST_N = 5
MIN_TRAIN_AFTER_HOLDOUT = 3

FEATURE_NAMES = [
    "cf_score", "content_score", "in_cf", "in_content",
    "pop_log", "hist_size_log", "tag_overlap", "release_year",
]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class ItemCF:
    """Item-item cosine on co-ownership, trained on a (user, item) interaction frame."""

    def __init__(self, interactions: pd.DataFrame):
        users = interactions.user_id.unique()
        items = interactions.item_id.unique()
        self.user_to_ix = {u: i for i, u in enumerate(users)}
        self.item_to_ix = {it: i for i, it in enumerate(items)}
        self.ix_to_item = np.asarray(items)

        rows = interactions.user_id.map(self.user_to_ix).values
        cols = interactions.item_id.map(self.item_to_ix).values
        ui = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(users), len(items)),
        )
        ui_col_norm = normalize(ui, norm="l2", axis=0)
        self.item_sim = (ui_col_norm.T @ ui_col_norm).tocsr().astype(np.float32)

    def _histories_to_csr(self, histories: list[list[str]]) -> csr_matrix:
        rows, cols = [], []
        for ui_, hist in enumerate(histories):
            for it in hist:
                ix = self.item_to_ix.get(it)
                if ix is not None:
                    rows.append(ui_)
                    cols.append(ix)
        return csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(histories), len(self.item_to_ix)),
        )

    def topk(self, histories: list[list[str]], top_k: int, batch: int = 1024):
        n = len(histories)
        hist_csr = self._histories_to_csr(histories)
        out_ids = np.empty((n, top_k), dtype=object)
        out_sc = np.zeros((n, top_k), dtype=np.float32)
        for s in range(0, n, batch):
            e = min(s + batch, n)
            scores = (hist_csr[s:e] @ self.item_sim).toarray()
            # exclude items already in history
            hist_dense = hist_csr[s:e].toarray().astype(bool)
            scores[hist_dense] = -np.inf
            k = min(top_k, scores.shape[1])
            part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            for i in range(e - s):
                row = part[i]
                order = np.argsort(-scores[i, row])
                ranked = row[order]
                out_ids[s + i, :k] = self.ix_to_item[ranked]
                out_sc[s + i, :k] = scores[i, ranked]
        return out_ids, out_sc


class ContentTFIDF:
    """TF-IDF over pipe-separated tags/genres/specs. Full catalog -> reaches cold items."""

    def __init__(self, items: pd.DataFrame):
        self.items = items
        items = items.copy()
        items["_text"] = items.apply(self._tokenize, axis=1)
        self.tfidf = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True)
        mat = self.tfidf.fit_transform(items._text)
        self.item_tfidf = normalize(mat, norm="l2", axis=1)
        self.all_items = items.item_id.values
        self.item_to_ix_full = {it: i for i, it in enumerate(self.all_items)}
        self.item_tokens_by_id = dict(
            zip(self.all_items, (set(t.split()) for t in items._text.values))
        )

    @staticmethod
    def _tokenize(row) -> str:
        out = []
        for col in ("tags", "genres", "specs"):
            v = row[col]
            if isinstance(v, str) and v:
                out.extend(t.strip().replace(" ", "_") for t in v.split("|") if t.strip())
        return " ".join(out)

    def topk(self, histories: list[list[str]], top_k: int, batch: int = 512):
        n = len(histories)
        n_items = self.item_tfidf.shape[0]

        # (n_users x n_items) weighting that yields the *mean* TF-IDF over history
        rows, cols, data = [], [], []
        for ui_, hist in enumerate(histories):
            ixs = [self.item_to_ix_full[it] for it in hist if it in self.item_to_ix_full]
            if not ixs:
                continue
            w = 1.0 / len(ixs)
            for ix in ixs:
                rows.append(ui_)
                cols.append(ix)
                data.append(w)
        weight = csr_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(n, n_items),
        )
        profiles = weight @ self.item_tfidf

        tfidf_T = self.item_tfidf.T.tocsr()
        out_ids = np.empty((n, top_k), dtype=object)
        out_sc = np.zeros((n, top_k), dtype=np.float32)
        for s in range(0, n, batch):
            e = min(s + batch, n)
            scores = (profiles[s:e] @ tfidf_T).toarray()
            hist_mask = np.zeros_like(scores, dtype=bool)
            for i, hist in enumerate(histories[s:e]):
                for it in hist:
                    ix = self.item_to_ix_full.get(it)
                    if ix is not None:
                        hist_mask[i, ix] = True
            scores[hist_mask] = -np.inf
            k = min(top_k, scores.shape[1])
            part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            for i in range(e - s):
                row = part[i]
                order = np.argsort(-scores[i, row])
                ranked = row[order]
                out_ids[s + i, :k] = self.all_items[ranked]
                out_sc[s + i, :k] = scores[i, ranked]
        return out_ids, out_sc


def build_candidates(cf_ids, cf_sc, ct_ids, ct_sc) -> list[dict]:
    """For each user, union CF + content top-K into {item_id: [cf_score, content_score]}."""
    all_cands = []
    for i in range(len(cf_ids)):
        d: dict[str, list[float]] = {}
        for j, it in enumerate(cf_ids[i]):
            if it is None:
                continue
            d[it] = [float(cf_sc[i, j]), 0.0]
        for j, it in enumerate(ct_ids[i]):
            if it is None:
                continue
            if it in d:
                d[it][1] = float(ct_sc[i, j])
            else:
                d[it] = [0.0, float(ct_sc[i, j])]
        all_cands.append(d)
    return all_cands


# ---------------------------------------------------------------------------
# Feature engineering for ranker
# ---------------------------------------------------------------------------

def build_feature_rows(
    users, histories, candidate_dicts,
    item_pop_log, item_year, item_to_ix_full, item_tokens_by_id,
    truth_by_user=None,
):
    rows, labels, keys = [], [], []
    for u, hist, cands in zip(users, histories, candidate_dicts):
        if not cands:
            continue
        hist_tokens: set[str] = set()
        for it in hist:
            hist_tokens |= item_tokens_by_id.get(it, set())
        hist_size_log = np.log1p(len(hist))
        truth = truth_by_user.get(u, set()) if truth_by_user is not None else None
        for it, (cf, ct) in cands.items():
            ix = item_to_ix_full[it]
            overlap = len(hist_tokens & item_tokens_by_id.get(it, set()))
            rows.append([
                cf, ct,
                1.0 if cf > 0 else 0.0,
                1.0 if ct > 0 else 0.0,
                item_pop_log[ix],
                hist_size_log,
                float(overlap),
                item_year[ix],
            ])
            if truth is not None:
                labels.append(1 if it in truth else 0)
            keys.append((u, it))
    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8) if truth_by_user is not None else None
    return X, y, keys


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _dcg(rels) -> float:
    rels = np.asarray(rels, dtype=np.float32)
    if rels.size == 0:
        return 0.0
    return float(np.sum(rels / np.log2(np.arange(2, rels.size + 2))))


def eval_top10(scores, keys, truth_by_user) -> tuple[float, float]:
    by_user: dict[str, list[tuple[str, float]]] = {}
    for (u, it), s in zip(keys, scores):
        by_user.setdefault(u, []).append((it, float(s)))
    recalls, ndcgs = [], []
    for u, lst in by_user.items():
        truth = truth_by_user.get(u, set())
        if not truth:
            continue
        lst.sort(key=lambda x: -x[1])
        top10 = [it for it, _ in lst[:10]]
        hits = [1.0 if it in truth else 0.0 for it in top10]
        recalls.append(sum(hits) / len(truth))
        idcg = _dcg([1.0] * min(len(truth), 10))
        ndcgs.append(_dcg(hits) / idcg if idcg > 0 else 0.0)
    return float(np.mean(recalls)), float(np.mean(ndcgs))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)

    print("\n=== 1. Load data ===")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test_users = pd.read_csv(DATA_DIR / "test_users.csv")
    items = pd.read_csv(DATA_DIR / "item_metadata.csv")
    print(f"train         : {train.shape}  ({train.user_id.nunique():,} users, {train.item_id.nunique():,} items)")
    print(f"test_users    : {test_users.shape}")
    print(f"item_metadata : {items.shape}")

    hist_len = train.groupby("user_id").size()
    pop = train.groupby("item_id").size()
    print(f"history length: median={hist_len.median():.0f}, max={hist_len.max()}")
    print(f"catalog coverage in train: {len(pop) / len(items):.1%}")

    print("\n=== 2. Validation split (last-5 per user) ===")
    eligible = hist_len[hist_len >= KEEP_LAST_N + MIN_TRAIN_AFTER_HOLDOUT].index
    train_sorted = train.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    train_sorted["rank_desc"] = train_sorted.groupby("user_id").cumcount(ascending=False)
    is_valid = train_sorted.user_id.isin(eligible) & (train_sorted.rank_desc < KEEP_LAST_N)
    valid = train_sorted[is_valid].drop(columns="rank_desc").reset_index(drop=True)
    train_inner = train_sorted[~is_valid].drop(columns="rank_desc").reset_index(drop=True)
    print(f"train_inner: {len(train_inner):,} rows, {train_inner.user_id.nunique():,} users")
    print(f"valid      : {len(valid):,} rows,    {valid.user_id.nunique():,} users")

    print("\n=== 3. Build retrievers (on train_inner) ===")
    cf = ItemCF(train_inner)
    print(f"item-item sim: {cf.item_sim.shape}, nnz={cf.item_sim.nnz:,}")
    content = ContentTFIDF(items)
    print(f"item TF-IDF  : {content.item_tfidf.shape}, vocab={len(content.tfidf.vocabulary_):,}")

    print("\n=== 4. Retrieve for validation users ===")
    hist_by_user = train_inner.groupby("user_id")["item_id"].apply(list).to_dict()
    valid_users = valid.user_id.unique().tolist()
    valid_histories = [hist_by_user.get(u, []) for u in valid_users]
    valid_truth = valid.groupby("user_id")["item_id"].apply(set).to_dict()

    cf_ids, cf_sc = cf.topk(valid_histories, TOPK_CF)
    ct_ids, ct_sc = content.topk(valid_histories, TOPK_CONTENT)
    valid_cands = build_candidates(cf_ids, cf_sc, ct_ids, ct_sc)

    pool_sizes = [len(c) for c in valid_cands]
    hits = total = 0
    for u, cands in zip(valid_users, valid_cands):
        truth = valid_truth.get(u, set())
        if not truth:
            continue
        hits += len(truth & cands.keys())
        total += len(truth)
    print(f"avg pool size : {np.mean(pool_sizes):.1f}")
    print(f"retrieval Recall@~{int(np.mean(pool_sizes))}: {hits / total:.4f}  "
          f"({hits:,}/{total:,} held-out hits)")

    print("\n=== 5. Build ranker features ===")
    item_pop_log = np.log1p(pop.reindex(content.all_items).fillna(0).values).astype(np.float32)
    item_year = (
        items.set_index("item_id")["release_year"]
        .reindex(content.all_items).fillna(-1).values.astype(np.float32)
    )

    perm = rng.permutation(len(valid_users))
    split = int(0.8 * len(valid_users))
    fit_ix, eval_ix = perm[:split], perm[split:]
    fit_users = [valid_users[i] for i in fit_ix]
    fit_hist = [valid_histories[i] for i in fit_ix]
    fit_cands = [valid_cands[i] for i in fit_ix]
    eval_users = [valid_users[i] for i in eval_ix]
    eval_hist = [valid_histories[i] for i in eval_ix]
    eval_cands = [valid_cands[i] for i in eval_ix]

    X_fit, y_fit, _ = build_feature_rows(
        fit_users, fit_hist, fit_cands,
        item_pop_log, item_year, content.item_to_ix_full, content.item_tokens_by_id,
        truth_by_user=valid_truth,
    )
    X_eval, y_eval, k_eval = build_feature_rows(
        eval_users, eval_hist, eval_cands,
        item_pop_log, item_year, content.item_to_ix_full, content.item_tokens_by_id,
        truth_by_user=valid_truth,
    )
    print(f"ranker fit : X={X_fit.shape}, positives={int(y_fit.sum()):,} ({y_fit.mean():.2%})")
    print(f"ranker eval: X={X_eval.shape}, positives={int(y_eval.sum()):,} ({y_eval.mean():.2%})")

    print("\n=== 6. Train ranker + evaluate ===")
    ranker = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6, random_state=SEED,
    )
    ranker.fit(X_fit, y_fit)

    baseline_scores = X_eval[:, 0] + X_eval[:, 1]
    r_b, n_b = eval_top10(baseline_scores, k_eval, valid_truth)
    proba = ranker.predict_proba(X_eval)[:, 1]
    r_m, n_m = eval_top10(proba, k_eval, valid_truth)
    print(f"sum-of-scores baseline : Recall@10={r_b:.4f}  NDCG@10={n_b:.4f}  mean={(r_b + n_b) / 2:.4f}")
    print(f"HistGB ranker          : Recall@10={r_m:.4f}  NDCG@10={n_m:.4f}  mean={(r_m + n_m) / 2:.4f}")
    print("random reference (data card) ~0.0017")

    print("\n=== 7. Retrain retrieval on full train + generate submission ===")
    cf_full = ItemCF(train)
    pop_full = train.groupby("item_id").size()
    item_pop_log_full = np.log1p(
        pop_full.reindex(content.all_items).fillna(0).values
    ).astype(np.float32)

    hist_full = train.groupby("user_id")["item_id"].apply(list).to_dict()
    test_uids = test_users.user_id.tolist()
    test_hist = [hist_full.get(u, []) for u in test_uids]

    print(f"retrieving for {len(test_uids):,} test users...")
    t_cf_ids, t_cf_sc = cf_full.topk(test_hist, TOPK_CF)
    t_ct_ids, t_ct_sc = content.topk(test_hist, TOPK_CONTENT)
    test_cands = build_candidates(t_cf_ids, t_cf_sc, t_ct_ids, t_ct_sc)

    X_test, _, k_test = build_feature_rows(
        test_uids, test_hist, test_cands,
        item_pop_log_full, item_year, content.item_to_ix_full, content.item_tokens_by_id,
        truth_by_user=None,
    )
    proba_test = ranker.predict_proba(X_test)[:, 1]

    by_user_scores: dict[str, list[tuple[str, float]]] = {}
    for (u, it), s in zip(k_test, proba_test):
        by_user_scores.setdefault(u, []).append((it, float(s)))

    fallback = pop_full.sort_values(ascending=False).index.tolist()
    rows = []
    for u in test_uids:
        owned = set(hist_full.get(u, []))
        lst = by_user_scores.get(u, [])
        lst.sort(key=lambda x: -x[1])
        picks, seen = [], set()
        for it, _ in lst:
            if it in owned or it in seen:
                continue
            picks.append(it)
            seen.add(it)
            if len(picks) == 10:
                break
        if len(picks) < 10:
            for it in fallback:
                if it in owned or it in seen:
                    continue
                picks.append(it)
                seen.add(it)
                if len(picks) == 10:
                    break
        assert len(picks) == 10, f"user {u} only has {len(picks)} picks"
        rows.append([u, *picks])

    sub_cols = ["user_id"] + [f"rank_{i}" for i in range(1, 11)]
    submission = pd.DataFrame(rows, columns=sub_cols)

    # Format validation
    assert len(submission) == len(test_users), "row count mismatch"
    assert set(submission.user_id) == set(test_users.user_id), "user_id mismatch"
    valid_item_ids = set(items.item_id)
    for col in sub_cols[1:]:
        assert submission[col].isin(valid_item_ids).all(), f"{col} has unknown items"
    dupes = submission[sub_cols[1:]].apply(lambda r: len(set(r)) != 10, axis=1)
    assert not dupes.any(), f"{int(dupes.sum())} rows have duplicate item_ids"
    owned_by_user = {u: set(h) for u, h in hist_full.items()}
    flat = submission.melt(id_vars="user_id", value_vars=sub_cols[1:], value_name="item_id")
    overlap = flat.apply(lambda r: r.item_id in owned_by_user.get(r.user_id, set()), axis=1)
    assert not overlap.any(), f"{int(overlap.sum())} predictions are owned by the user"

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"wrote {SUBMISSION_PATH.name}: {submission.shape}  (all format checks passed)")


if __name__ == "__main__":
    main()
