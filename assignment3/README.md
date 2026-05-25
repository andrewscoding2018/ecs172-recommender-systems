# Assignment 3: Multi-Stage Recommender

You are given a dataset of Steam users and the games they own (with playtime). For each user in the test set, your task is to predict the top-10 games the user will play next, chosen from the full catalog of ~30,000 games — no candidate list is provided this time. Your expected submission is a ranked list of 10 game IDs per user. We evaluate with Recall@10 and NDCG@10 against the held-out games each user actually played.

## Required Pipeline

### Stage 1 — Retrieval

Retrieval narrows the ~30K-item catalog down to a few hundred candidates per user. Pick from the routes below (mix several, or invent your own):

- **Content-based retrieval.** Embed each game from its tags, genres, and specs (TF-IDF, sentence-transformers, etc.). For each user, build a profile vector from games in their history and retrieve nearest neighbors via cosine similarity (or FAISS).
- **Item-CF co-occurrence retrieval.** Build an item–item similarity matrix from co-ownership patterns in `train.csv` (e.g., normalized co-counts, Jaccard, cosine). For each user, score every catalog item as the sum of similarities to games in their history; take the top-N.
- **Matrix-factorization retrieval.** Train ALS or implicit-MF on the user–item interaction matrix; retrieve via user × item dot product.

### Stage 2 — Ranking

Re-rank the candidates per user with a model that uses richer features than your retrieval routes did:

- Build training labels yourself: take each user's last few interactions out of `train.csv` and treat them as positives (label = 1). For negatives (label = 0), sample random games the user hasn't played.
- Features are where the points are. You must engineer at least 5 features. Examples:
  - **User-side:** history size, number of distinct tags played, total playtime, days active.
  - **Item-side:** popularity, average playtime, release year, tag count.
  - **Interaction-side:** each of your retrieval scores; cosine sim between candidate and user history; tag overlap; whether the candidate shares a developer/publisher with any history item.

You may not use the test ground truth for any feature or training signal.

## Files Provided

| File | Description |
|------|-------------|
| `train.csv` | User interaction history: `(user_id, item_id, playtime_minutes, timestamp)` |
| `test_users.csv` | The list of `user_id`s you must produce predictions for |
| `item_metadata.csv` | Per-game metadata: title, tags, genres, specs, developer, publisher, release_year, sentiment |
| `sample_submission.csv` | Example showing the exact output format |
| `evaluation_code.py` | The exact scoring function the autograder runs |
| `data_card.md` | Dataset schema and basic counts |

## Submission Format

A CSV named `submission.csv` with one row per test user and 10 ranked game IDs:

```csv
user_id,rank_1,rank_2,rank_3,rank_4,rank_5,rank_6,rank_7,rank_8,rank_9,rank_10
u_00001,i_04321,i_00088,i_19842,i_02931,i_15007,i_00211,i_07765,i_00102,i_22301,i_18004
u_00002,...
```

Rules:

- Exactly 10 distinct `item_id`s per row, ordered most-to-least relevant.
- Every `user_id` from `test_users.csv` must appear exactly once.
- Every predicted `item_id` must exist in `item_metadata.csv`.
- You may not include items the user already owns in `train.csv`.

See `baseline_format.md` for the full validator.

## Report Requirements

Submit a 3–5 page PDF (`report.pdf`) covering:

- **Architecture diagram.** A simple block diagram of your pipeline: data → retrieval → ranking features → ranker → top-10.
- **Retrieval recall.** Report the Recall@K of your retrieval candidate pool on your local validation set, where K is whatever pool size you feed your ranker (typically K=100–200).
- **Ranking features.** List your features, group them as user / item / interaction, and report a feature-importance table or plot.
- **Cold-user / cold-item handling.** How does your pipeline behave for a user with very limited interaction history (e.g., only 3 observed training items)?

## Grading

| Component | Weight |
|-----------|-------:|
| Leaderboard score (`(Recall@10 + NDCG@10) / 2`) | 25% |
| Two-stage architecture implementation | 20% |
| Report quality | 50% |
| Code reproducibility (one-command run, fixed seeds) | 5% |

For reference only (not grading anchors), random recommendation lands near ~0.0017 on the `(Recall@10 + NDCG@10) / 2` metric.

## Academic Integrity

Discussion of high-level approaches with classmates is fine; sharing code or submission CSVs is not. You may use AI assistants for boilerplate and debugging, but the architectural and feature-engineering decisions must be yours and explained in your report. Submissions are checked for similarity.
