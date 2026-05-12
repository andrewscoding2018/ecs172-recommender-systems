# %% [markdown]
# Approach:
# - explore rating distributions
#      - plot rating distributions
#      - look at review text
#      - check out metadata
#
# - split training into train vs. validation
#     - like hold user's last interactions from `train.csv` for validation
#
# - what makes a good user profile?
#     - user who loves puzzles will have review mentioning puzzle, brain teaser, etc
#

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import vstack, csr_matrix

import numpy as np

# %%
metadata = pd.read_csv("item_metadata.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# %%
metadata

# %%
plt.figure(figsize=(8, 4))
sns.histplot(metadata["average_rating"], kde=True, stat="density", bins=30)
plt.xlabel("average_rating")
plt.ylabel("Density")
plt.title("Distribution of Average Rating")
plt.show()

# %%
interactions_per_user = train.groupby("user_id").count()["rating"]

plt.figure(figsize=(8, 4))
sns.histplot(interactions_per_user, bins=20)
plt.xlabel("Number of reviews per user")
plt.ylabel("Number of users")
plt.title("Distribution of Reviews per User")
plt.show()

# %%
missing = (
    metadata.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .rename("missing_pct")
    .reset_index()
    .rename(columns={"index": "column"})
)

plt.figure(figsize=(8, 4))
sns.barplot(data=missing, x="missing_pct", y="column")
plt.xlabel("Percent Missing")
plt.ylabel("Column")
plt.title("Missing Data by Column")
plt.show()


# %%
text_cols = ["title", "features", "description"]

summary = []
for col in text_cols:
    s = metadata[col].fillna("").astype(str).str.strip()
    word_count = s.str.split().str.len()
    char_count = s.str.len()

    summary.append(
        {
            "column": col,
            "missing_pct": metadata[col].isna().mean() * 100,
            "word_stddev": word_count.std(),
            "char_stddev": char_count.std(),
            "empty_pct": (s == "").mean() * 100,
            "duplicate_pct": s.duplicated().mean() * 100,
            "avg_words": word_count.mean(),
            "median_words": word_count.median(),
            "p90_words": word_count.quantile(0.9),
            "avg_chars": char_count.mean(),
            "median_chars": char_count.median(),
        }
    )

summary_df = pd.DataFrame(summary)
summary_df.T

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, col in zip(axes, text_cols):
    s = metadata[col].fillna("").astype(str).str.strip()
    word_count = s.str.split().str.len()
    sns.histplot(word_count, bins=40, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Word count")

plt.tight_layout()
plt.show()

# %%
for col in text_cols:
    s = metadata[col].fillna("").astype(str).str.strip()
    print(f"\n--- {col} ---")
    print("Missing %:", metadata[col].isna().mean())
    print("Empty %:", (s == "").mean())
    print("Duplicate %:", s.duplicated().mean())
    print("Very short % (<5 words):", (s.str.split().str.len() < 5).mean())

# %%
for col in text_cols:
    s = metadata[col].fillna("").astype(str).str.strip()
    wc = s.str.split().str.len()
    tmp = pd.DataFrame({col: s, "words": wc})

    print(f"\nShortest {col}:")
    print(tmp.sort_values("words").head(5)[[col, "words"]].to_string(index=False))

    print(f"\nLongest {col}:")
    print(
        tmp.sort_values("words", ascending=False)
        .head(5)[[col, "words"]]
        .to_string(index=False)
    )


# %% [markdown]
# ## Merging data

# %%
train_with_product_data = train.merge(
    metadata, left_on="item_id", right_on="item_id", how="left"
)

# %%
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
)

X = vectorizer.fit_transform(train_with_product_data["review_text"].fillna(""))

# %% [markdown]
# ## Approach 1: Item & User Vectors

# %%
text_cols = ["title", "features", "description"]

items = metadata[["item_id"] + text_cols].copy()
items["item_text"] = (
    items[text_cols]
    .fillna("")
    .agg(" ".join, axis=1)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    max_features=20000,
)

item_matrix = vectorizer.fit_transform(items["item_text"])
item_matrix = normalize(item_matrix)

item_to_idx = pd.Series(np.arange(len(items)), index=items["item_id"])

# %%
train_with_product_data = train.merge(
    items[["item_id", "item_text"]], on="item_id", how="left"
)

train_with_product_data["weight"] = (train_with_product_data["rating"] - 3).clip(
    lower=0
)

user_ids = sorted(train_with_product_data["user_id"].unique())
user_profiles = []

for user_id in user_ids:
    user_hist = train_with_product_data[train_with_product_data["user_id"] == user_id]

    item_idxs = item_to_idx.loc[user_hist["item_id"]].to_numpy()
    weights = user_hist["weight"].to_numpy()

    if weights.sum() == 0:
        weights = np.ones(len(weights))

    profile = item_matrix[item_idxs].multiply(weights[:, None]).sum(axis=0)
    profile = csr_matrix(profile / weights.sum())  # optional average instead of raw sum
    user_profiles.append(profile)

user_matrix = vstack(user_profiles, format="csr")
user_matrix = normalize(user_matrix)

user_to_idx = pd.Series(np.arange(len(user_ids)), index=user_ids)

# %%
test_scored = test.copy()

scores = []
for _, row in test_scored.iterrows():
    u_idx = user_to_idx[row["user_id"]]
    i_idx = item_to_idx[row["item_id"]]

    score = user_matrix[u_idx].dot(item_matrix[i_idx].T)[0, 0]
    scores.append(score)

test_scored["score"] = scores


# %%
ranked = (
    test_scored.sort_values(["user_id", "score"], ascending=[True, False])
    .groupby("user_id")["item_id"]
    .apply(list)
    .reset_index()
)

ranked[["rank_1", "rank_2", "rank_3", "rank_4", "rank_5"]] = pd.DataFrame(
    ranked["item_id"].tolist(), index=ranked.index
)

# %%
submission = (
    test_scored.sort_values(["user_id", "score"], ascending=[True, False])
    .groupby("user_id")["item_id"]
    .apply(list)
    .reset_index()
)

submission[["rank_1", "rank_2", "rank_3", "rank_4", "rank_5"]] = pd.DataFrame(
    submission["item_id"].tolist(), index=submission.index
)

submission = submission.drop(columns="item_id")
submission = submission[["user_id", "rank_1", "rank_2", "rank_3", "rank_4", "rank_5"]]

submission.to_csv("submission.csv", index=False)

assert submission.shape == (2000, 6)
assert submission["user_id"].nunique() == 2000
assert submission.columns.tolist() == [
    "user_id",
    "rank_1",
    "rank_2",
    "rank_3",
    "rank_4",
    "rank_5",
]
assert not submission.duplicated("user_id").any()
assert (
    submission[["rank_1", "rank_2", "rank_3", "rank_4", "rank_5"]].notna().all().all()
)


# %% [markdown]
# ## Validation

# %%
# hold out each user's latest interaction.
rng = np.random.default_rng(172)

train_sorted = train.sort_values(["user_id", "timestamp"])
val_holdout = train_sorted.groupby("user_id", as_index=False).tail(1)
val_train = train_sorted.drop(index=val_holdout.index)

all_item_ids = items["item_id"].to_numpy()
user_seen_items = train.groupby("user_id")["item_id"].apply(set).to_dict()

val_candidates = []
for row in val_holdout.itertuples(index=False):
    positive_item = row.item_id
    seen = user_seen_items[row.user_id]
    negative_pool = np.array([item for item in all_item_ids if item not in seen])
    negatives = rng.choice(negative_pool, size=4, replace=False)

    candidates = np.concatenate([[positive_item], negatives])
    rng.shuffle(candidates)

    for item_id in candidates:
        val_candidates.append(
            {
                "user_id": row.user_id,
                "item_id": item_id,
                "relevant": int(item_id == positive_item),
            }
        )

val_candidates = pd.DataFrame(val_candidates)

# %%
# remake user profiles using just the pre-holdout interactions
val_train_with_text = val_train.merge(
    items[["item_id", "item_text"]], on="item_id", how="left"
)
val_train_with_text["weight"] = (val_train_with_text["rating"] - 3).clip(lower=0)

val_user_ids = sorted(val_train_with_text["user_id"].unique())
val_user_profiles = []

for user_id in val_user_ids:
    user_hist = val_train_with_text[val_train_with_text["user_id"] == user_id]
    item_idxs = item_to_idx.loc[user_hist["item_id"]].to_numpy()
    weights = user_hist["weight"].to_numpy()

    if weights.sum() == 0:
        weights = np.ones(len(weights))

    profile = item_matrix[item_idxs].multiply(weights[:, None]).sum(axis=0)
    profile = csr_matrix(profile / weights.sum())
    val_user_profiles.append(profile)

val_user_matrix = normalize(vstack(val_user_profiles, format="csr"))
val_user_to_idx = pd.Series(np.arange(len(val_user_ids)), index=val_user_ids)

# %%
# score each validation candidate with the same cosine-similarity ranking rule
val_scores = []
for row in val_candidates.itertuples(index=False):
    u_idx = val_user_to_idx[row.user_id]
    i_idx = item_to_idx[row.item_id]
    score = val_user_matrix[u_idx].dot(item_matrix[i_idx].T)[0, 0]
    val_scores.append(score)

val_candidates["score"] = val_scores
val_candidates["rank"] = (
    val_candidates.sort_values(["user_id", "score"], ascending=[True, False])
    .groupby("user_id")
    .cumcount()
    + 1
)

positive_ranks = val_candidates[val_candidates["relevant"] == 1]["rank"]
validation_results = pd.Series(
    {
        "users": positive_ranks.size,
        "hit_rate_at_1": (positive_ranks == 1).mean(),
        "hit_rate_at_3": (positive_ranks <= 3).mean(),
        "mrr_at_5": (1 / positive_ranks).mean(),
        "ndcg_at_5": (1 / np.log2(positive_ranks + 1)).mean(),
    }
)

validation_results
