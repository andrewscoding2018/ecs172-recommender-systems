# Assignment 1: Content-Based Filtering — Next Item Retrieval
## Task Description
You are given a dataset of user–item interactions from software product reviews.
Each user has a history of items they reviewed, along with review text, ratings,
and timestamps.
For each user in the test set, you are given **5 candidate items**. Your task is to
**rank these 5 items by content relevance to the user** — that is, based on how
well each candidate's content matches the user's preferences as reflected in their
history.
This is a **retrieval/ranking** task using **content-based filtering**.

## Core Concepts
1. **Item Representation:** Use item content from `item_metadata.csv` (title,
description, features, categories, etc.) to build a vector for each item. Methods
like **TF-IDF** or **Bag-of-Words** on text fields are a good starting point.
2. **User Representation:** Build a user profile from their interaction history in
`train.csv`. You can:
- Aggregate the content vectors of items the user has rated
- Use the user's own review texts to build a text-based profile
- Weight by rating, recency, or other signals
3. **Ranking:** For each user's 5 test candidates, score how well each candidate
matches the user's profile, and rank them from most relevant to least relevant.

## Dataset
### Files Provided

| File | Description |
| --- | --- |
| `train.csv` | User interaction history with review text |
| `test.csv` | Test set with 5 candidate items per user (10,000 rows = 2,000 users × 5) |
| `item_metadata.csv` | Content features for all items |
| `sample_submission.csv` | Example submission format |

### Data Schema
**train.csv** — your training data (split this yourself into train/validation as
needed):

| Column | Type | Description |
| --- | --- | --- |
| `user_id` | string | User identifier (e.g., `u_0042`) |
| `item_id` | string | Item identifier (e.g., `i_1234`) |
| `rating` | float | Rating the user gave (1–5) |
| `review_text` | string | The user's review text |
| `timestamp` | int | Unix timestamp of the review (milliseconds) |

**test.csv** — 5 candidate items per user:

| Column | Type | Description |
| --- | --- | --- |
| `user_id` | string | User identifier |
| `item_id` | string | Candidate item identifier |

For each user, there are exactly 5 rows representing 5 candidate items.

**item_metadata.csv** — content features for each item:

| Column | Type | Description |
| --- | --- | --- |
| `item_id` | string | Item identifier |
| `title` | string | Product title |
| `description` | string | Product description |
| `features` | string | Product feature bullet points |
| `categories` | string | Product categories |
| `main_category` | string | Top-level category |
| `store` | string | Developer / seller name |
| `price` | float | Product price |
| `average_rating` | float | Global average rating |

### Dataset Statistics
- **Users:** 2,000
- **Items:** 8,681
- **Training interactions:** 32,206
- **Reviews per user:** 10–48 (mean 16.1)
- **Test:** 2,000 users × 5 candidates = 10,000 rows
## Submission Format
Submit a CSV file named `submission.csv`. For each user, provide a **ranked list of
all 5 candidate items**, ordered from most relevant to least relevant:
```
user_id,rank_1,rank_2,rank_3,rank_4,rank_5
u_0000,i_7284,i_4556,i_6084,i_4817,i_6617
u_0001,i_3339,i_5967,i_6257,i_0734,i_4458
...
```
### Requirements
- **Exactly 2,000 rows** — one row per user in `test.csv`.
- **`user_id`** must match the user IDs in `test.csv`.
- **`rank_1`** through **`rank_5`**: your ranking of the 5 candidate items, from
most relevant (rank_1) to least relevant (rank_5).
- All 5 items must come from that user's candidate set in `test.csv`.
- No duplicate items within a row.
- See `sample_submission.csv` for the exact format.
## Tips
- **Start by exploring the data.** Plot rating distributions, look at review text
examples, check what metadata fields are available.
- **Split your training data** into a train and validation set yourself. A natural
approach: hold out each user's last interaction(s) from `train.csv` for validation.
- **Think about what makes a good user profile.** A user who loves puzzle games
will have reviews mentioning "puzzle", "brain teaser", etc. — their profile should
reflect
