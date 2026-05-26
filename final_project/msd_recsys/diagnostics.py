"""Diagnostic prints that justify each design choice (not just report a number).

Each function prints a short, focused block. Call from the notebook between
pipeline stages — they're cheap and they keep the report's "why" defensible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def diag_data(interactions: pd.DataFrame, items_catalog: pd.DataFrame | None = None) -> None:
    """Why we need to filter + why a hybrid retriever."""
    n_users = interactions.user_id.nunique()
    n_items = interactions.song_id.nunique()
    print(f"[diag] interactions: {len(interactions):,} rows, {n_users:,} users, {n_items:,} items")
    if items_catalog is not None:
        catalog_size = len(items_catalog)
        cold_share = 1 - n_items / catalog_size
        print(f"[diag] catalog: {catalog_size:,} items, {cold_share:.1%} are cold (no train interactions)")
        print(f"       -> any cold-item recall comes from the content/semantic-ID route.")
    sizes = interactions.groupby("user_id").size()
    pop = interactions.groupby("song_id").size()
    print(f"[diag] user history: median={sizes.median():.0f}, p90={sizes.quantile(0.9):.0f}, max={sizes.max()}")
    print(f"[diag] song popularity: median={pop.median():.0f}, mean={pop.mean():.1f}, max={pop.max()}")
    print(f"       -> long-tailed; aggressive item filtering will reshape the distribution.")


def diag_filter(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Show what filtering kept/dropped — justifies the threshold."""
    nu_b, ni_b = before.user_id.nunique(), before.song_id.nunique()
    nu_a, ni_a = after.user_id.nunique(), after.song_id.nunique()
    print(f"[diag] after filter:")
    print(f"  rows:  {len(before):>12,} -> {len(after):>12,}  ({len(after)/len(before):.1%} kept)")
    print(f"  users: {nu_b:>12,} -> {nu_a:>12,}  ({nu_a/nu_b:.1%} kept)")
    print(f"  items: {ni_b:>12,} -> {ni_a:>12,}  ({ni_a/ni_b:.1%} kept)")
    print(f"       -> tail items dropped; ALS now operates on a denser, more reliable signal.")


def diag_als(als_retriever, item_titles: dict[str, str] | None = None, sample_items: list[str] | None = None, k: int = 5) -> None:
    """ALS sanity check — show top similar items for a few sample songs."""
    print("[diag] ALS similar-item check:")
    try:
        items = sample_items or list(als_retriever.item_to_ix.keys())[:3]
        for it in items:
            ix = als_retriever.item_to_ix.get(it)
            if ix is None:
                continue
            similar_ids, similar_sc = als_retriever.model.similar_items(ix, N=k + 1)
            base_title = (item_titles or {}).get(it, it)
            print(f"  {base_title[:40]}:")
            for sim_ix, sim_sc in zip(similar_ids[1:], similar_sc[1:]):  # skip self
                sim_item = als_retriever.ix_to_item[sim_ix]
                title = (item_titles or {}).get(sim_item, sim_item)
                print(f"    sim={sim_sc:.3f}  {title[:60]}")
    except Exception as e:
        print(f"  (skipped: {e})")
    print("  -> if similar songs are same artist / same era / same genre, ALS is learning real structure.")


def diag_semantic_codes(sid_retriever, item_titles: dict[str, str] | None = None, n_show: int = 10) -> None:
    """Show a few items at each code level — does the clustering look sensible?"""
    print("[diag] Semantic-ID code distribution:")
    if not sid_retriever.item_codes:
        print("  (no codes — call fit first)")
        return
    sample_keys = list(sid_retriever.item_codes.keys())[:n_show]
    for k in sample_keys:
        codes = sid_retriever.item_codes[k]
        title = (item_titles or {}).get(k, k)
        print(f"  {title[:40]:40s} codes={codes}")
    print("  -> items with overlapping codes should share metadata (decade, artist, popularity tier).")


def diag_pool(
    cands_by_user: list[dict[str, list[float]]],
    truth_by_user: dict | None = None,
    user_ids: list[str] | None = None,
) -> None:
    """Pool composition — does the hybrid actually combine two routes?"""
    pool_sizes = [len(c) for c in cands_by_user]
    n = len(cands_by_user)
    als_only = sid_only = both = 0
    for c in cands_by_user:
        for als_s, sid_s in c.values():
            if als_s > 0 and sid_s > 0:
                both += 1
            elif als_s > 0:
                als_only += 1
            else:
                sid_only += 1
    total = als_only + sid_only + both
    print(f"[diag] hybrid pool:")
    print(f"  avg pool size : {np.mean(pool_sizes):.1f}")
    print(f"  ALS only      : {als_only/n:8.1f}/user ({als_only/total:.1%} of pool)")
    print(f"  semantic only : {sid_only/n:8.1f}/user ({sid_only/total:.1%} of pool)")
    print(f"  both routes   : {both/n:8.1f}/user ({both/total:.1%} of pool)")
    print(f"  -> low 'both' = routes are complementary; high 'both' = redundant.")

    if truth_by_user is not None and user_ids is not None:
        hits = total = 0
        for u, c in zip(user_ids, cands_by_user):
            t = truth_by_user.get(u, set())
            if not t:
                continue
            hits += len(t & c.keys())
            total += len(t)
        if total:
            print(f"  retrieval recall@~{int(np.mean(pool_sizes))}: {hits/total:.4f}  ({hits:,}/{total:,} held-out hits)")


def diag_features(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    """Per-feature positive-vs-negative mean and ratio."""
    print("[diag] feature means by class (discriminative power):")
    print(f"  {'feature':<24}{'positive':>14}{'negative':>14}{'ratio':>10}")
    pos, neg = X[y == 1], X[y == 0]
    if pos.shape[0] == 0:
        print("  (no positives — eval truth might be empty)")
        return
    for i, name in enumerate(feature_names):
        p, n = float(pos[:, i].mean()), float(neg[:, i].mean())
        if n == 0:
            ratio_str = "inf" if p > 0 else "n/a"
        else:
            ratio_str = f"{p/n:.2f}x"
        print(f"  {name:<24}{p:>14.4f}{n:>14.4f}{ratio_str:>10}")
    print("  -> ratios far from 1.0 mean the feature separates the classes.")


def diag_permutation_importance(model, X_eval, y_eval, feature_names: list[str], rng=None) -> None:
    """Permutation importance with average_precision scoring (right metric under imbalance).

    For LightGBM rankers, we cast to predict_proba via the raw model.predict.
    Use a sample of eval rows (~30K) to keep it fast.
    """
    from sklearn.inspection import permutation_importance

    if rng is None:
        rng = np.random.default_rng(42)
    n = min(30_000, len(X_eval))
    idx = rng.choice(len(X_eval), n, replace=False)

    # LGBMRanker doesn't expose predict_proba; permutation_importance can accept
    # a custom scorer. Use AP scored against raw scores.
    from sklearn.metrics import average_precision_score
    def scorer(est, X, y):
        return average_precision_score(y, est.predict(X))

    result = permutation_importance(
        model, X_eval[idx], y_eval[idx],
        scoring=scorer, n_repeats=3, random_state=42, n_jobs=1,
    )
    order = np.argsort(-result.importances_mean)
    print("[diag] permutation importance (sampled 30K eval rows, AP scoring):")
    print(f"  {'feature':<24}{'importance':>12}{'std':>10}")
    for i in order:
        print(f"  {feature_names[i]:<24}{result.importances_mean[i]:>12.5f}{result.importances_std[i]:>10.5f}")
    print("  -> drop in AP when feature is shuffled; higher = ranker relies on it more.")


def diag_results(
    *,
    map_score: float,
    retrieval_recall: float,
    baseline_map: float | None = None,
    random_reference: float | None = None,
    k: int = 500,
) -> None:
    """Final interpretation block — ceiling, capture, lift."""
    print("[diag] putting it together:")
    print(f"  Retrieval ceiling (Recall@pool) : {retrieval_recall:.4f}")
    print(f"  Achieved MAP@{k:<3}              : {map_score:.4f}")
    if retrieval_recall > 0:
        print(f"  Capture efficiency              : {map_score/retrieval_recall:.1%}")
    if baseline_map is not None and baseline_map > 0:
        print(f"  Lift over baseline              : {map_score/baseline_map:.2f}x")
    if random_reference is not None and random_reference > 0:
        print(f"  Lift over random                : {map_score/random_reference:.1f}x")
    print(f"  -> if capture efficiency is low, ranker is the bottleneck; if ceiling is low, retrieval is.")
