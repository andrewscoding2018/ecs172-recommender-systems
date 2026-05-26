"""Data loading, filtering, splitting, and sparse-matrix construction for MSD.

The MSD challenge ships interactions as tab-separated triplets and track metadata
as a SQLite database. This module wraps the I/O + standard preprocessing so the
notebook stays clean.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_triplets(path: str | Path) -> pd.DataFrame:
    """Load any (user_id, song_id, play_count) triplet file.

    Works for train_triplets.txt, kaggle_visible_evaluation_triplets.txt, and
    year1_{valid,test}_triplets_hidden.txt — same TSV format.
    """
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["user_id", "song_id", "play_count"],
        dtype={"user_id": "string", "song_id": "string", "play_count": np.int32},
    )
    return df


def load_song_id_list(path: str | Path) -> list[str]:
    """Load kaggle_songs.txt — canonical list of song_ids."""
    # File format is usually "1 SOAAADD12A8C13D8C7" — one-indexed line number + song_id.
    # We just want the song_id column.
    with open(path) as f:
        first = f.readline().strip().split()
    if len(first) == 2 and first[0].isdigit():
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["idx", "song_id"])
        return df.song_id.tolist()
    # fallback: one id per line
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_user_list(path: str | Path) -> list[str]:
    """Load kaggle_users.txt — canonical list of user_ids (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_song_to_track(path: str | Path) -> pd.DataFrame:
    """Load taste_profile_song_to_tracks.txt — song_id -> track_id mapping.

    Real format is variable-width:
        song_id<TAB>track_id_1[<TAB>track_id_2[<TAB>...]]
    because one song_id can map to multiple MSD track_ids (different versions
    of the same song). Returns long-format: one row per (song_id, track_id) pair.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            song_id = parts[0]
            for track_id in parts[1:]:
                if track_id:
                    rows.append((song_id, track_id))
    return pd.DataFrame(rows, columns=["song_id", "track_id"]).astype("string")


# Columns we expect from track_metadata.db. Confirm against your DB; the official
# MSD `songs` table has these names. If yours differs, edit METADATA_COLUMNS.
METADATA_COLUMNS = [
    "track_id", "title", "song_id", "release",
    "artist_id", "artist_name",
    "duration", "artist_familiarity", "artist_hotttnesss", "year",
]


def load_track_metadata(db_path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load track metadata from the SQLite DB into a DataFrame."""
    cols = columns or METADATA_COLUMNS
    col_sql = ", ".join(cols)
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(f"SELECT {col_sql} FROM songs", conn)
    # MSD encodes "missing" as 0 for numeric columns
    for c in ("artist_familiarity", "artist_hotttnesss", "year"):
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan
    return df


# ---------------------------------------------------------------------------
# Filtering & splitting
# ---------------------------------------------------------------------------

def filter_interactions(
    df: pd.DataFrame,
    *,
    min_song_listens: int = 50,
    min_user_listens: int = 20,
    max_passes: int = 3,
) -> pd.DataFrame:
    """Drop rare songs and rare users iteratively.

    Filtering one dimension shrinks the other (a song with 60 listens may drop
    below threshold once we remove the users we filtered). `max_passes` iterations
    is almost always enough to reach a fixed point on MSD.
    """
    out = df
    for i in range(max_passes):
        before = len(out)
        song_counts = out.groupby("song_id").size()
        out = out[out.song_id.isin(song_counts[song_counts >= min_song_listens].index)]
        user_counts = out.groupby("user_id").size()
        out = out[out.user_id.isin(user_counts[user_counts >= min_user_listens].index)]
        if len(out) == before:
            break
    return out.reset_index(drop=True)


def holdout_split(
    df: pd.DataFrame,
    *,
    n_per_user: int = 5,
    min_train_after_holdout: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's last `n_per_user` interactions for validation.

    MSD triplets don't have timestamps. We use the row order in `df` as a proxy
    for "later" — fine if your input is ordered by user_id then by the listen
    order encoded in the original file. If you have access to a timestamp,
    sort by it before calling this.

    Users with too-short history (<n_per_user + min_train_after_holdout) keep
    all their data in `train_inner` and contribute no valid rows.
    """
    sizes = df.groupby("user_id").size()
    eligible = sizes[sizes >= n_per_user + min_train_after_holdout].index

    df_sorted = df.copy()
    df_sorted["__rank_desc"] = df_sorted.groupby("user_id").cumcount(ascending=False)
    is_valid = df_sorted.user_id.isin(eligible) & (df_sorted.__rank_desc < n_per_user)

    valid = df_sorted[is_valid].drop(columns="__rank_desc").reset_index(drop=True)
    train_inner = df_sorted[~is_valid].drop(columns="__rank_desc").reset_index(drop=True)
    return train_inner, valid


# ---------------------------------------------------------------------------
# Sparse user-item matrix
# ---------------------------------------------------------------------------

def build_user_item_matrix(
    df: pd.DataFrame,
    *,
    confidence_alpha: float = 40.0,
    use_logged_confidence: bool = True,
) -> tuple[csr_matrix, dict[str, int], dict[str, int]]:
    """Build sparse user x item matrix for ALS.

    For implicit feedback, ALS expects confidence values, not raw counts.
    Standard transform (Hu, Koren, Volinsky 2008): c_ui = 1 + alpha * log(1 + r_ui)
    when use_logged_confidence else c_ui = 1 + alpha * r_ui.

    Returns:
        ui: sparse user x item CSR matrix.
        user_to_ix: dict mapping user_id -> row index.
        item_to_ix: dict mapping song_id -> col index.
    """
    users = df.user_id.unique()
    items = df.song_id.unique()
    user_to_ix = {u: i for i, u in enumerate(users)}
    item_to_ix = {it: i for i, it in enumerate(items)}

    rows = df.user_id.map(user_to_ix).values
    cols = df.song_id.map(item_to_ix).values
    counts = df.play_count.values.astype(np.float32)

    if use_logged_confidence:
        confidence = 1.0 + confidence_alpha * np.log1p(counts)
    else:
        confidence = 1.0 + confidence_alpha * counts

    ui = csr_matrix(
        (confidence.astype(np.float32), (rows, cols)),
        shape=(len(users), len(items)),
    )
    return ui, user_to_ix, item_to_ix


def histories_from_df(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return {user_id: [song_id, ...]} from an interaction frame."""
    return df.groupby("user_id")["song_id"].apply(list).to_dict()
