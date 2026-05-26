"""MSD RecSys — two-stage recommender for the Million Song Dataset.

Stage 1 (retrieval): ALS + semantic-ID content similarity -> hybrid candidate pool.
Stage 2 (ranking):   LightGBM lambdarank over engineered features -> top-500.

Typical usage from a notebook:

    from msd_recsys import data, retrieval, features, ranker, eval, diagnostics
    from msd_recsys.checkpoint import checkpoint

    interactions = data.load_train_triplets(DATA_DIR / "train_triplets.txt")
    filtered     = data.filter_interactions(interactions, min_song=50, min_user=20)
    train_inner, valid = data.holdout_split(filtered, n_per_user=5)
    ui_mat, u_ix, i_ix = data.build_user_item_matrix(train_inner)

    als = checkpoint("als_v1", lambda: retrieval.ALSRetriever(factors=64).fit(ui_mat))
    candidates = als.recommend_batch(ui_mat, top_k=2000)
    ...
"""
from . import data, retrieval, features, ranker, eval, diagnostics, checkpoint  # noqa: F401

__version__ = "0.1.0"
