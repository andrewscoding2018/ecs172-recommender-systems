# Data Card

## Overview

- **Task:** Hybrid Recommendation — Top-10 Next-Item Prediction
- **Domain:** Steam video game catalog (PC games)
- **Source:** Subsampled and anonymized from the Steam reviews dataset (Kang & McAuley, UCSD).
  Original files: `steam_reviews.json.gz` (~7.7M reviews, 1.3GB) and `steam_games.json.gz` (~32K game records, 2.7MB) from <https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data>.
- **Feedback type:** A row means the user posted a review for the game on the indicated date. Used as an implicit "the user played this game" signal. `playtime_minutes` is the user's total hours on that title (from the same review record), converted to minutes.

## Counts

|  | Count |
|---|---:|
| Test users (in `test_users.csv`) | 10,000 |
| Catalog items in `item_metadata.csv` | 32,132 |
| ...of which appear in `train.csv` | 8,036 (25%) — 75% of catalog has no training interactions |
| Training interactions (`train.csv`) | 122,366 |
| Mean train history per user | 12.24 |
| Median train history per user | 7 |
| Min / Max train history per user | 3 / 536 |
| Item popularity (median / mean / max) | 4 / 15.2 / 941 |
| Review timestamp range | 2010-10-16 to 2018-01-05 |
| Rows with playtime_minutes = 0 | 677 / 122,366 (0.55%) |
| Median playtime among >0 rows | 522 minutes (~8.7 h) |

## Training Data (`train.csv`)

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Anonymized user identifier (e.g., `u_00000`) |
| item_id | string | Anonymized game identifier (e.g., `i_00000`) |
| playtime_minutes | int | Total playtime by this user on this game (≥ 0; 0 means reviewed but not played) |
| timestamp | int | Unix timestamp (seconds) of the review |

## Test Data

- `test_users.csv` — single column `user_id`, listing every user you must predict for. Ground truth is held out for grading.

## Split Policy

For each user in the dataset:

- We sort their reviews by time.
- The **5 most recent** reviews are pulled out as the hidden ground truth used for grading — you never see these.
- Everything earlier is what you see in `train.csv`.

So every test user has exactly **5** items you are supposed to predict. Your top-10 submission is scored against those 5.

What this means for you:

- Per-user `Recall@K` only takes values in `{0, 0.2, 0.4, 0.6, 0.8, 1.0}` — the denominator is always 5. Don't be surprised by the coarse jumps. Any `K ≥ 5` can in principle reach a perfect 1.0.
- **For your local validation to track the leaderboard, mirror this split.** Hold out each user's **last 5** interactions from `train.csv` and train on the rest. A random split will give you numbers that look fine locally but won't predict your leaderboard score at all.
- Users with very short histories were dropped before the split, so every user has at least 3 interactions in `train.csv` on top of the 5 held-out.

## Item Metadata (`item_metadata.csv`)

| Field | Type | Description | Coverage |
|-------|------|-------------|----------|
| item_id | string | Game identifier | 100% |
| title | string | Game title | 100% |
| tags | string | Pipe-separated crowd-sourced Steam tags (e.g., `Action\|Roguelike\|Indie`) | ~99% |
| genres | string | Pipe-separated official Steam genres (e.g., `Action\|Indie`) | ~80% |
| specs | string | Pipe-separated technical specs (e.g., `Single-player\|Multi-player\|Steam Achievements`) | ~80% |
| developer | string | Developer studio name | High |
| publisher | string | Publisher name | High |
| release_year | int | Year of release parsed from `release_date`; -1 if missing/coming-soon | ~85% |
| sentiment | string | Steam aggregate-review sentiment (e.g., `Mostly Positive`, `Very Positive`, `Mixed`); empty for titles with too few reviews to score | Partial |

## Notes and Known Gotchas

- **Every row in `train.csv` is a positive.** A row means the user played the game. There are no negatives in the data — if your ranker needs negatives, pick random games the user hasn't played.
- **Playtime values are wildly skewed.** Most rows have small playtime, a few are enormous, and ~0.5% are zero (reviewed but not played). If you use playtime as a feature, apply `log1p` first or the long tail will dominate everything.
- **Tags are very imbalanced.** A handful of tags (`Indie`, `Action`, `Adventure`) appear on most games. Out of the box they will swamp any tag-similarity computation — apply IDF weighting or drop the most common ones.
- **A few catalog entries are not games** (utilities, demos, etc.). They are valid predictions but rarely appear in user histories.
- **IDs are anonymized.** Only the `u_NNNNN` / `i_NNNNN` IDs in the provided files are valid — predicting raw Steam appids will fail submission validation. Don't try to look up the original Steam IDs.
- **Most of the catalog is "cold".** 75% of items in `item_metadata.csv` have zero training interactions (see Counts table). A pure CF retrieval can never recommend them. To reach them you need a content-based route.
