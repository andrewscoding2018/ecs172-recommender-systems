"""
Assignment 3 — multi-stage recommender

two stages

  STAGE 1 (RETRIEVAL): narrow the 32K-item catalog to a few hundred candidates
    per user using...two complementary routes:
      (a) item-cf: item-item cosine on co-ownership
      (b) content TF-IDF: tags|genres|specs vectors

    We then will usion top-K from each route -> hybrid candidate pool.

  STAGE 2 (RANKING): re-rank the candidates using engineered features that
  mix retrieval signals with user/item priors


pipeline in main():
  1. load data and do some quick diagnostics
  2. do a local validation split that mirrors the leaderboard (leave out last-5 per user)
  3. STAGE 1: build retrievers on train_inner, retrieve, and then check out retrieval reccall@K
  4. STAGE 2: build features, train ranker, report recall@10 and ncdg@10
  5. retrain STAGE 1 on full train, then generate our submission.csv

HOW TO RUN:
`uv run pipeline.py`

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import normalize

SEED = 42
DATA_DIR = Path(__file__).parent / "assignment3data"
SUBMISSION_PATH = Path(__file__).parent / "submission.csv"

TOPK_CF = 150
TOPK_CONTENT = 150
KEEP_LAST_N = 5
MIN_TRAIN_AFTER_HOLDOUT = 3


FEATURE_SPECS = [
    ("cf_score",       "Stage-1 item-CF score, high means that = candidate is played alongside with the user's history"),
    ("content_score",  "Stage-1 TF-IDF cosine map to user profile, high = tag/genre overlap with user's history"),
    ("in_cf",          "1 if the CF route surfaced this candidate"),
    ("in_content",     "1 if the content route surfaced this candidate, with in_cf, marks 'both routes agreed' candidates."),
    ("pop_log",        "popular games get reviewed more often"),
    ("hist_size_log",  "this is to measure engagement"),
    ("tag_overlap",    "# tokens shared between candidate's tags/genres/specs and user's history tokens"),
    ("release_year",   "item release year (-1 if missing), to let ranker prefer recent"),
]
FEATURE_NAMES = [name for name, _ in FEATURE_SPECS]


# ============================================================================
# STAGE 1 — RETRIEVAL
# ============================================================================

class ItemCF:
    """Item-item cosine scores
    
    cold catalog items are unreachable from this route, which is why we
    need colaborative filtering
    """

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
        # cosine similarity between item columns using l2 norm
        ui_col_norm = normalize(ui, norm="l2", axis=0)
        # then (U^TU)
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
            # items already in history should be excluded
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
    """TF-IDF over tags / genres / specs.

    this will work on the full 32K-item catalog, so cold items ~ARE~ reachable here
    """

    def __init__(self, items: pd.DataFrame):
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

    # may need to use later
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

        # this users x items matirx will yield the mean TD_IDF over each user's history
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
            # exclude items already in history
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
    """Union STAGE-1 top-K from both routes -> {item_id: [cf_score, content_score]}"""
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


# ============================================================================
# STAGE 2 — RANKING
# ============================================================================

def build_feature_rows(
    users, histories, candidate_dicts,
    item_pop_log, item_year, item_to_ix_full, item_tokens_by_id,
    truth_by_user=None,
):
    """Here we want to flatten (user, candidate) pairs into a feature matrix for STAGE 2 (ranking).

    Features
    - cf_score
    - content_score
    - in_cf
    - in_content,
    - pop_log
    - hist_size_log
    - tag_overlap
    - release_year

    """
    rows, labels, keys = [], [], []
    for u, hist, cands in zip(users, histories, candidate_dicts):
        if not cands:
            continue

        # Precompute the user's token bag once per user (not per candidate).
        hist_tokens: set[str] = set()
        for it in hist:
            hist_tokens |= item_tokens_by_id.get(it, set())
        hist_size_log = float(np.log1p(len(hist)))

        truth = truth_by_user.get(u, set()) if truth_by_user is not None else None
        for it, (cf, ct) in cands.items():
            ix = item_to_ix_full[it]
            overlap = len(hist_tokens & item_tokens_by_id.get(it, set()))
            rows.append([
                cf,                              # cf_score
                ct,                              # content_score
                1.0 if cf > 0 else 0.0,          # in_cf  (route provenance flag)
                1.0 if ct > 0 else 0.0,          # in_content
                item_pop_log[ix],                # pop_log  (popularity prior)
                hist_size_log,                   # hist_size_log  (user engagement)
                float(overlap),                  # tag_overlap  (raw count, ranker can re-scale)
                item_year[ix],                   # release_year (-1 = missing)
            ])
            if truth is not None:
                labels.append(1 if it in truth else 0)
            keys.append((u, it))
    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8) if truth_by_user is not None else None
    return X, y, keys


# ============================================================================
# EVALUATION
# ============================================================================

def _dcg(rels) -> float:
    rels = np.asarray(rels, dtype=np.float32)
    if rels.size == 0:
        return 0.0
    return float(np.sum(rels / np.log2(np.arange(2, rels.size + 2))))


def eval_top10(scores, keys, truth_by_user) -> tuple[float, float]:
    """Group scored (user, item) pairs, take top-10 per user, compute Recall@10 / NDCG@10."""
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


# ============================================================================
# DIAGNOSTICS — print just enough to justify each design choice
# ============================================================================

def diag_data(train: pd.DataFrame, items: pd.DataFrame, hist_len, pop) -> None:
    print("\n[diag] Why hybrid retrieval?")
    cold_share = 1 - len(pop) / len(items)
    print(f"  {cold_share:.0%} of catalog has zero training interactions — item-CF cannot reach them.")
    print(f"  -> content TF-IDF is required to even score these items.")

    print("\n[diag] Popularity skew (why we include pop_log as a feature):")
    top = pop.sort_values(ascending=False).head(5)
    top_titles = items.set_index("item_id").loc[top.index, "title"].tolist()
    for (iid, n), title in zip(top.items(), top_titles):
        print(f"  {iid}  n={n:4d}  {title}")
    print(f"  median pop={pop.median():.0f}  mean={pop.mean():.1f}  max={pop.max()}  "
          f"-> log1p tames the long tail.")


def diag_cf(cf: ItemCF, items: pd.DataFrame, k: int = 5) -> None:
    print("\n[diag] Item-CF sanity check — top co-occurring item pairs:")
    # Pull strongest off-diagonal entries of item_sim
    coo = cf.item_sim.tocoo()
    mask = coo.row != coo.col
    rows, cols, vals = coo.row[mask], coo.col[mask], coo.data[mask]
    order = np.argsort(-vals)[: k * 2]  # 2x because each pair appears twice (symmetric)
    seen = set()
    title_by_id = items.set_index("item_id")["title"].to_dict()
    shown = 0
    for r, c, v in zip(rows[order], cols[order], vals[order]):
        key = tuple(sorted((r, c)))
        if key in seen:
            continue
        seen.add(key)
        a, b = cf.ix_to_item[r], cf.ix_to_item[c]
        print(f"  sim={v:.3f}  {title_by_id.get(a, a)[:35]:35s} <-> {title_by_id.get(b, b)[:35]}")
        shown += 1
        if shown >= k:
            break
    print("  (If these look like obvious siblings/sequels, CF is learning real structure.)")


def diag_content(content: ContentTFIDF, k: int = 5) -> None:
    print("\n[diag] Content TF-IDF — why we need IDF weighting:")
    vocab = content.tfidf.vocabulary_
    inv_vocab = {i: w for w, i in vocab.items()}
    idfs = content.tfidf.idf_
    # most common (low IDF) tokens — these would dominate without IDF
    low = np.argsort(idfs)[:k]
    high = np.argsort(-idfs)[:k]
    print(f"  Lowest-IDF tokens (most common; would swamp similarity if un-weighted):")
    for i in low:
        print(f"    {inv_vocab[i]:30s} idf={idfs[i]:.2f}")
    print(f"  Highest-IDF tokens (rare; most distinctive):")
    for i in high:
        print(f"    {inv_vocab[i]:30s} idf={idfs[i]:.2f}")


def diag_retrieval(users, histories, cands, truth, hist_by_user) -> None:
    print("\n[diag] Hybrid pool composition (per user, averaged):")
    cf_only = ct_only = both = 0
    pool_sizes = []
    for c in cands:
        pool_sizes.append(len(c))
        for cf_s, ct_s in c.values():
            if cf_s > 0 and ct_s > 0:
                both += 1
            elif cf_s > 0:
                cf_only += 1
            else:
                ct_only += 1
    total = cf_only + ct_only + both
    n = len(cands)
    print(f"  avg pool size : {np.mean(pool_sizes):.1f}")
    print(f"  CF only       : {cf_only/n:6.1f} per user ({cf_only/total:.1%} of pool)")
    print(f"  content only  : {ct_only/n:6.1f} per user ({ct_only/total:.1%} of pool)")
    print(f"  both routes   : {both/n:6.1f} per user ({both/total:.1%} of pool)")
    print(f"  -> low 'both' share = the routes complement each other (good).")

    print("\n[diag] Retrieval recall by user history bucket:")
    buckets = [(3, 5), (6, 10), (11, 25), (26, 10_000)]
    rows = []
    for lo, hi in buckets:
        hits = total = users_in_bucket = 0
        for u, c in zip(users, cands):
            h = len(hist_by_user.get(u, []))
            if not (lo <= h <= hi):
                continue
            t = truth.get(u, set())
            if not t:
                continue
            users_in_bucket += 1
            hits += len(t & c.keys())
            total += len(t)
        recall = hits / total if total else 0.0
        rows.append((f"{lo}-{hi if hi < 10_000 else '+'}", users_in_bucket, recall))
    print(f"  {'bucket':<8}{'users':>8}{'recall':>10}")
    for label, n_u, r in rows:
        print(f"  {label:<8}{n_u:>8}{r:>10.4f}")
    print("  -> cold users have lower recall; content route helps but doesn't fully close the gap.")


def diag_features(X: np.ndarray, y: np.ndarray) -> None:
    print("\n[diag] Feature means: positive vs negative class (discriminative power):")
    print(f"  {'feature':<18}{'positive':>12}{'negative':>12}{'pos/neg':>10}")
    pos, neg = X[y == 1], X[y == 0]
    for i, name in enumerate(FEATURE_NAMES):
        p, n = pos[:, i].mean(), neg[:, i].mean()
        ratio = (p / n) if n != 0 else float("inf")
        print(f"  {name:<18}{p:>12.4f}{n:>12.4f}{ratio:>10.2f}x")
    print("  -> ratios far from 1.0 mean the feature separates the classes.")


def diag_ranker(ranker, X_eval, y_eval, rng) -> None:
    print("\n[diag] Permutation importance (on sampled 30K eval rows, scoring=AP):")
    n = min(30_000, len(X_eval))
    idx = rng.choice(len(X_eval), n, replace=False)
    result = permutation_importance(
        ranker, X_eval[idx], y_eval[idx],
        n_repeats=3, random_state=SEED, scoring="average_precision", n_jobs=1,
    )
    order = np.argsort(-result.importances_mean)
    print(f"  {'feature':<18}{'importance':>12}{'std':>10}")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:<18}{result.importances_mean[i]:>12.5f}{result.importances_std[i]:>10.5f}")
    print("  -> drop in AP when this feature is shuffled; higher = ranker relies on it more.")


def diag_results(retrieval_recall, baseline, ranker_score) -> None:
    r_b, _ = baseline
    r_m, _ = ranker_score
    ceiling = retrieval_recall  # upper bound on Recall@10 given the pool
    efficiency = r_m / ceiling if ceiling else 0.0
    lift = r_m / r_b if r_b else float("inf")
    print("\n[diag] Putting it together:")
    print(f"  Retrieval ceiling on Recall@10        : {ceiling:.4f}  (pool contains truth this often)")
    print(f"  Ranker achieved Recall@10             : {r_m:.4f}")
    print(f"  Capture efficiency (achieved/ceiling) : {efficiency:.1%}")
    print(f"  Lift over sum-of-scores baseline      : {lift:.2f}x")
    print(f"  Lift over random (~0.0017)            : {r_m / 0.0017:.1f}x")
    print("  -> retrieval ceiling is the bigger lever right now; raising it lifts everything.")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    rng = np.random.default_rng(SEED)

    # --------------------------------------------------------------------
    # 1. Load + sanity-check the data
    # --------------------------------------------------------------------
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
    diag_data(train, items, hist_len, pop)

    # --------------------------------------------------------------------
    # 2. Validation split — mirror the leaderboard's last-5-per-user holdout
    # --------------------------------------------------------------------
    print("\n=== 2. Validation split (last-5 per user) ===")
    eligible = hist_len[hist_len >= KEEP_LAST_N + MIN_TRAIN_AFTER_HOLDOUT].index
    train_sorted = train.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    train_sorted["rank_desc"] = train_sorted.groupby("user_id").cumcount(ascending=False)
    is_valid = train_sorted.user_id.isin(eligible) & (train_sorted.rank_desc < KEEP_LAST_N)
    valid = train_sorted[is_valid].drop(columns="rank_desc").reset_index(drop=True)
    train_inner = train_sorted[~is_valid].drop(columns="rank_desc").reset_index(drop=True)
    print(f"train_inner: {len(train_inner):,} rows, {train_inner.user_id.nunique():,} users")
    print(f"valid      : {len(valid):,} rows,    {valid.user_id.nunique():,} users "
          f"(only users with >= {KEEP_LAST_N + MIN_TRAIN_AFTER_HOLDOUT} history are eligible)")

    # --------------------------------------------------------------------
    # STAGE 1 — RETRIEVAL
    # --------------------------------------------------------------------
    print("\n=== STAGE 1: build retrievers on train_inner ===")
    cf = ItemCF(train_inner)
    print(f"item-item sim: {cf.item_sim.shape}, nnz={cf.item_sim.nnz:,}")
    diag_cf(cf, items)

    content = ContentTFIDF(items)
    print(f"item TF-IDF  : {content.item_tfidf.shape}, vocab={len(content.tfidf.vocabulary_):,}")
    diag_content(content)

    print("\n=== STAGE 1: retrieve candidates for validation users ===")
    hist_by_user = train_inner.groupby("user_id")["item_id"].apply(list).to_dict()
    valid_users = valid.user_id.unique().tolist()
    valid_histories = [hist_by_user.get(u, []) for u in valid_users]
    valid_truth = valid.groupby("user_id")["item_id"].apply(set).to_dict()

    cf_ids, cf_sc = cf.topk(valid_histories, TOPK_CF)
    ct_ids, ct_sc = content.topk(valid_histories, TOPK_CONTENT)
    valid_cands = build_candidates(cf_ids, cf_sc, ct_ids, ct_sc)

    hits = total = 0
    for u, cands in zip(valid_users, valid_cands):
        truth = valid_truth.get(u, set())
        if not truth:
            continue
        hits += len(truth & cands.keys())
        total += len(truth)
    retrieval_recall = hits / total
    avg_pool = float(np.mean([len(c) for c in valid_cands]))
    print(f"retrieval Recall@~{int(avg_pool)}: {retrieval_recall:.4f}  "
          f"({hits:,}/{total:,} held-out hits)")
    diag_retrieval(valid_users, valid_histories, valid_cands, valid_truth, hist_by_user)

    # --------------------------------------------------------------------
    # STAGE 2 — RANKING
    # --------------------------------------------------------------------
    print("\n=== STAGE 2: build ranker features ===")
    item_pop_log = np.log1p(pop.reindex(content.all_items).fillna(0).values).astype(np.float32)
    item_year = (
        items.set_index("item_id")["release_year"]
        .reindex(content.all_items).fillna(-1).values.astype(np.float32)
    )

    # 80/20 split of valid users -> ranker fit vs. ranker eval
    perm = rng.permutation(len(valid_users))
    split = int(0.8 * len(valid_users))
    fit_ix, eval_ix = perm[:split], perm[split:]
    fit_users  = [valid_users[i]     for i in fit_ix]
    fit_hist   = [valid_histories[i] for i in fit_ix]
    fit_cands  = [valid_cands[i]     for i in fit_ix]
    eval_users = [valid_users[i]     for i in eval_ix]
    eval_hist  = [valid_histories[i] for i in eval_ix]
    eval_cands = [valid_cands[i]     for i in eval_ix]

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
    diag_features(X_fit, y_fit)

    print("\n=== STAGE 2: train ranker + evaluate ===")
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
    print(f"random reference (data card) ~0.0017")
    diag_ranker(ranker, X_eval, y_eval, rng)

    # Recall on the eval slice for ceiling math (different users than the global recall above)
    eval_hits = eval_total = 0
    for u, c in zip(eval_users, eval_cands):
        t = valid_truth.get(u, set())
        if not t:
            continue
        eval_hits += len(t & c.keys())
        eval_total += len(t)
    eval_retrieval_recall = eval_hits / eval_total
    diag_results(eval_retrieval_recall, (r_b, n_b), (r_m, n_m))

    # --------------------------------------------------------------------
    # 3. Retrain STAGE 1 on full train + generate submission
    # --------------------------------------------------------------------
    print("\n=== Retrain STAGE 1 on full train + generate submission ===")
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

    # Popularity fallback for users where retrieval came up < 10 candidates
    fallback = pop_full.sort_values(ascending=False).index.tolist()
    rows = []
    fallback_used = 0
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
            fallback_used += 1
            for it in fallback:
                if it in owned or it in seen:
                    continue
                picks.append(it)
                seen.add(it)
                if len(picks) == 10:
                    break
        assert len(picks) == 10, f"user {u} only has {len(picks)} picks"
        rows.append([u, *picks])
    print(f"users needing popularity fallback: {fallback_used} / {len(test_uids)}")

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
