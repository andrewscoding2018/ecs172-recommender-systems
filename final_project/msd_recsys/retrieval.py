"""Stage 1 — Retrieval.

Two complementary retrievers with a shared interface, union into a hybrid pool:
  - ALSRetriever: implicit-feedback matrix factorization on the user-item matrix.
    Reaches popular and mid-tail items well; can't score items absent from train.
  - SemanticIDRetriever: metadata-derived discrete codes (k-means on numeric
    metadata + categorical buckets). Reaches the full catalog, including cold
    songs ALS can't see.

Both expose recommend_batch(user_indices, top_k) -> (ids, scores) so a third
retriever (e.g., audio-based) can drop in later without changing the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ============================================================================
# ALS retriever (via implicit)
# ============================================================================

class ALSRetriever:
    """Wrapper around implicit.als.AlternatingLeastSquares.

    Fit once on the confidence-weighted user-item matrix; recommend per user.
    Auto-uses GPU if available (controllable via use_gpu).
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 30,
        use_gpu: bool | None = None,
        random_state: int = 42,
    ):
        from implicit.als import AlternatingLeastSquares
        try:
            from implicit.gpu import HAS_CUDA
        except Exception:
            HAS_CUDA = False
        if use_gpu is None:
            use_gpu = bool(HAS_CUDA)
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=use_gpu,
            random_state=random_state,
        )
        self.item_to_ix: dict[str, int] = {}
        self.ix_to_item: np.ndarray = np.empty(0, dtype=object)
        self.user_to_ix: dict[str, int] = {}
        self.ix_to_user: np.ndarray = np.empty(0, dtype=object)

    def fit(self, ui_matrix: csr_matrix, user_to_ix: dict, item_to_ix: dict) -> "ALSRetriever":
        self.model.fit(ui_matrix, show_progress=True)
        self.user_to_ix = user_to_ix
        self.item_to_ix = item_to_ix
        self.ix_to_user = np.asarray(list(user_to_ix.keys()), dtype=object)
        self.ix_to_item = np.asarray(list(item_to_ix.keys()), dtype=object)
        return self

    def recommend_batch(
        self,
        ui_matrix: csr_matrix,
        top_k: int = 2000,
        user_indices: np.ndarray | None = None,
        filter_already_liked: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recommend top_k items for each user.

        Args:
            ui_matrix: full user-item matrix used at fit time (needed to filter
                       already-liked items).
            user_indices: which row indices to score. None = all users.
            top_k: number of items per user.

        Returns (item_ids_per_user, scores_per_user) with shape (n_users, top_k).
        """
        if user_indices is None:
            user_indices = np.arange(ui_matrix.shape[0])
        ids, scores = self.model.recommend(
            userid=user_indices,
            user_items=ui_matrix[user_indices],
            N=top_k,
            filter_already_liked_items=filter_already_liked,
        )
        # ids are column indices into the item space; map back to item_ids
        out_ids = self.ix_to_item[ids]
        return out_ids, scores


# ============================================================================
# Semantic-ID retriever (metadata-based)
# ============================================================================

@dataclass
class SemanticIDConfig:
    """How to derive discrete codes from item metadata.

    The Mei et al. 2025 paper uses RQ-VAE on audio embeddings; we adapt to
    metadata-only by k-means on standardized numeric features + one-hot
    categorical buckets. Each item gets a tuple of `n_levels` cluster IDs,
    so similarity = # matching code positions.
    """
    n_levels: int = 3                # number of independent code books
    codes_per_level: int = 256       # k for each k-means
    numeric_features: tuple = ("artist_familiarity", "artist_hotttnesss", "duration")
    categorical_features: tuple = ("decade", "artist_id")
    random_state: int = 42


class SemanticIDRetriever:
    """Derive multi-level discrete codes per item from metadata, retrieve by code overlap.

    Train: fit k-means per level on a stratified subset of items.
    User profile: bag of codes weighted by listen count in user's history.
    Recommend: items whose codes overlap most with the user's code bag.
    """

    def __init__(self, config: SemanticIDConfig | None = None):
        self.config = config or SemanticIDConfig()
        self.item_codes: dict[str, tuple[int, ...]] = {}
        self.all_items: np.ndarray = np.empty(0, dtype=object)
        # Sparse (n_items x total_codes) one-hot of codes per item
        self.item_code_matrix: csr_matrix | None = None
        self._n_total_codes: int = 0

    def fit(self, metadata: pd.DataFrame) -> "SemanticIDRetriever":
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler

        cfg = self.config
        md = metadata.copy()

        # Bucket year -> decade (handles missing year)
        if "year" in md.columns and "decade" not in md.columns:
            md["decade"] = (md["year"] // 10 * 10).fillna(-1).astype(int)

        # Build feature matrix: scaled numerics + cat-id one-hot indices (via hashing).
        # For simplicity we cluster on the numeric part alone for each level, varying
        # the random seed; categoricals are appended as separate "code positions"
        # (categorical-as-code instead of clustered).
        numeric_cols = [c for c in cfg.numeric_features if c in md.columns]
        numeric = md[numeric_cols].fillna(md[numeric_cols].median()).values.astype(np.float32)
        numeric_scaled = StandardScaler().fit_transform(numeric)

        item_ids = md["song_id"].values if "song_id" in md.columns else md["track_id"].values
        n = len(md)
        codes = np.zeros((n, cfg.n_levels + len(cfg.categorical_features)), dtype=np.int32)

        # Numeric-based codes: k-means at each level with a different seed
        for level in range(cfg.n_levels):
            km = MiniBatchKMeans(
                n_clusters=cfg.codes_per_level,
                random_state=cfg.random_state + level,
                batch_size=4096,
                n_init=3,
            )
            codes[:, level] = km.fit_predict(numeric_scaled)

        # Categorical codes: factorize each requested cat column
        for j, cat in enumerate(cfg.categorical_features):
            if cat in md.columns:
                codes[:, cfg.n_levels + j] = pd.factorize(md[cat])[0]
            else:
                codes[:, cfg.n_levels + j] = -1

        # Build sparse (n_items x total_codes) one-hot. Each code position uses
        # its own slice of column space, offset to avoid collisions.
        offsets = []
        running = 0
        for level in range(cfg.n_levels):
            offsets.append(running)
            running += cfg.codes_per_level
        for j, cat in enumerate(cfg.categorical_features):
            offsets.append(running)
            if cat in md.columns:
                running += md[cat].nunique() + 1
            else:
                running += 1
        self._n_total_codes = running

        rows = np.repeat(np.arange(n), codes.shape[1])
        cols = np.empty(rows.shape, dtype=np.int64)
        for pos in range(codes.shape[1]):
            cols[pos::codes.shape[1]] = codes[:, pos] + offsets[pos]
        data = np.ones(rows.shape, dtype=np.float32)
        self.item_code_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n, running),
        )

        self.all_items = item_ids
        self._item_to_ix = {it: i for i, it in enumerate(item_ids)}
        self.item_codes = {
            item_ids[i]: tuple(int(c) for c in codes[i]) for i in range(n)
        }
        return self

    def recommend_for_histories(
        self,
        histories: list[list[str]],
        top_k: int = 2000,
        already_owned: list[set[str]] | None = None,
        batch: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score every catalog item against each user's code bag; return top_k per user.

        High level: for each user, sum the discrete-code one-hots of items in their
        history to form a "code bag" profile, then score every catalog item by how
        many code positions it shares with that profile. Top-k by score is the
        retrieval output.

        Args:
            histories: one list of item_ids per user (their visible listening history).
            top_k: number of candidates to return per user.
            already_owned: optional per-user set of item_ids to exclude from results
                (e.g., songs the user has already played — we don't want to recommend them).
            batch: number of users scored per inner loop iteration. Bigger = faster
                but more memory for the dense (batch x n_items) score matrix.

        Returns:
            out_ids: (n_users, top_k) object array of item_ids, ranked best->worst.
            out_sc:  (n_users, top_k) float32 array of matching scores.
        """
        # Must have fit() before scoring — that's where item_code_matrix is built.
        assert self.item_code_matrix is not None, "call fit() first"

        n = len(histories)                          # number of users to score
        n_items = self.item_code_matrix.shape[0]    # full catalog size

        # Pre-transpose the item-code matrix once. We'll do `profiles @ item_codes_T`
        # later; having the transpose in CSR form makes that matmul efficient.
        # Shape: (n_total_codes, n_items)
        item_codes_T = self.item_code_matrix.T.tocsr()

        # ------------------------------------------------------------------
        # PHASE 1: build the per-user "code bag" profile matrix.
        #
        # Each user gets a length-n_total_codes vector. Position c counts how many
        # times code c appears across the items in their history. We're building this
        # as a sparse CSR by collecting (row, col, value) triplets and constructing
        # the matrix in one shot at the end — much faster than incrementally building.
        # ------------------------------------------------------------------
        rows, cols, data = [], [], []
        for ui_, hist in enumerate(histories):
            # Map this user's history item_ids -> row indices into item_code_matrix.
            # Drop any items not in our catalog (shouldn't happen if data is clean,
            # but defensive against weird edge cases).
            ixs = [self._item_to_ix[h] for h in hist if h in self._item_to_ix]
            if not ixs:
                continue  # user has zero in-catalog history — leave their profile row empty

            # Sum the one-hot code vectors of all history items.
            # Result is a (1, n_codes) sparse-row sum, materialized as a numpy matrix
            # because scipy returns matrix subclass here.
            sub = self.item_code_matrix[ixs].sum(axis=0)
            sub = np.asarray(sub).ravel()           # flatten matrix -> 1D array

            # Record only the nonzero entries (sparse storage). Building a Python
            # list of triplets is the fast path for sparse matrix construction.
            nz = np.nonzero(sub)[0]
            for c in nz:
                rows.append(ui_)
                cols.append(c)
                data.append(float(sub[c]))

        # Construct the full (n_users, n_total_codes) sparse profile matrix at once.
        profiles = csr_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(n, self._n_total_codes),
        )

        # ------------------------------------------------------------------
        # PHASE 2: score every (user, item) pair, mask owned items, take top-k.
        #
        # We process users in batches because the score matrix (batch x n_items) is
        # dense — for 256 users x ~1M items x 4 bytes that's already ~1 GB. Doing
        # it all at once would OOM on most machines.
        # ------------------------------------------------------------------

        # Pre-allocate output arrays so we write in place instead of appending.
        out_ids = np.empty((n, top_k), dtype=object)
        out_sc = np.zeros((n, top_k), dtype=np.float32)

        for s in range(0, n, batch):
            e = min(s + batch, n)  # batch covers users [s, e)

            # Sparse @ sparse matmul -> sparse result; .toarray() densifies it for
            # the subsequent masking + argpartition (those want dense input).
            # Each entry (i, j) = # code positions that match between user i's
            # profile and item j's code vector.
            scores = (profiles[s:e] @ item_codes_T).toarray()  # shape (batch, n_items)

            # Mask items the user already owns by setting their scores to -inf —
            # argpartition will then never pick them as top-k. We do this per-user
            # inside the batch because each user has a different owned set.
            if already_owned is not None:
                for i, owned in enumerate(already_owned[s:e]):
                    # Map owned item_ids -> column indices; ignore any not in catalog.
                    mask = [self._item_to_ix[o] for o in owned if o in self._item_to_ix]
                    if mask:
                        scores[i, mask] = -np.inf

            # Top-k selection. argpartition(-scores, k-1) gives the indices of the k
            # largest scores per row in O(n_items), unordered. We then sort just
            # those k indices to get a properly ranked top-k list.
            k = min(top_k, n_items)
            part = np.argpartition(-scores, k - 1, axis=1)[:, :k]  # (batch, k) unordered

            # Per-user: sort the k candidates by score (descending), then write the
            # ranked item_ids and scores into the pre-allocated output arrays.
            for i in range(e - s):
                row = part[i]                              # k unordered column indices
                order = np.argsort(-scores[i, row])        # sort those k by score desc
                ranked = row[order]                        # final ranked column indices
                out_ids[s + i, :k] = self.all_items[ranked]
                out_sc[s + i, :k] = scores[i, ranked]

        return out_ids, out_sc


# ============================================================================
# Hybrid candidate pool
# ============================================================================

def build_candidate_pool(
    als_ids: np.ndarray, als_sc: np.ndarray,
    sid_ids: np.ndarray, sid_sc: np.ndarray,
) -> list[dict[str, list[float]]]:
    """Union ALS + semantic-ID top-K per user.

    Returns one dict per user: {item_id: [als_score, semantic_score]}.
    Score is 0.0 if that route didn't surface the candidate (preserves provenance).
    """
    n = len(als_ids)
    out = []
    for i in range(n):
        d: dict[str, list[float]] = {}
        for j in range(als_ids.shape[1]):
            it = als_ids[i, j]
            if it is None or (isinstance(it, float) and np.isnan(it)):
                continue
            d[it] = [float(als_sc[i, j]), 0.0]
        for j in range(sid_ids.shape[1]):
            it = sid_ids[i, j]
            if it is None or (isinstance(it, float) and np.isnan(it)):
                continue
            if it in d:
                d[it][1] = float(sid_sc[i, j])
            else:
                d[it] = [0.0, float(sid_sc[i, j])]
        out.append(d)
    return out
