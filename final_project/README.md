# MSD RecSys — Two-Stage Recommender with Semantic IDs

ECS 172 final project. Two-stage recommender for the Million Song Dataset Challenge.

## Pipeline

```
Stage 0  preprocess: filter rare items/users, hold out last-5 per user
Stage 1  retrieval:  ALS (implicit) + semantic-ID content similarity -> ~2000 candidates/user
Stage 2  ranking:    LightGBM lambdarank over 8+ engineered features -> top-500
Eval     MAP@500, Recall@500, retrieval Recall@K, catalog coverage, intra-list diversity
         All metrics bucketed by song popularity tier (head/torso/tail) and user volume
```

## Layout

```
final_project/
├── pyproject.toml
├── msd_recsys/         # library — import from notebook
│   ├── data.py         # load triplets/metadata, filter, hold-out split, build CSR
│   ├── retrieval.py    # ALSRetriever, SemanticIDRetriever, hybrid candidate pool
│   ├── features.py     # build (user, candidate) feature matrix for ranker
│   ├── ranker.py       # LightGBM lambdarank wrapper
│   ├── eval.py         # MAP@K, Recall@K, NDCG@K, coverage, diversity, bucketed
│   ├── diagnostics.py  # diag_* helpers — justify each design choice with a print
│   └── checkpoint.py   # Drive-aware save/load so Colab sessions don't redo work
└── notebooks/
    └── pipeline.ipynb  # driver — imports msd_recsys, walks through stages
```

## Setup in Google Colab

```python
# Cell 1 — clone + install (once per session)
!git clone https://github.com/YOUR_USER/ecs172-recommender-systems.git
%cd ecs172-recommender-systems
!pip install -q -e ./final_project

# Cell 2 — mount Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Cell 3 — point at your data + checkpoint dirs
import os
DATA_DIR  = '/content/drive/MyDrive/msd_data'         # raw .txt / .db files here
CKPT_DIR  = '/content/drive/MyDrive/msd_checkpoints'  # ALS factors, candidates, models
os.makedirs(CKPT_DIR, exist_ok=True)
```

After that, the notebook drives the library — see `notebooks/pipeline.ipynb`.

## Local dev (outside Colab)

```bash
uv pip install -e final_project
# or with pip:
pip install -e final_project
```

## Data files expected at DATA_DIR

| File | Purpose |
|---|---|
| `train_triplets.txt` | 48M (user, song, play_count) interactions |
| `kaggle_visible_evaluation_triplets.txt` | visible half of eval users |
| `kaggle_songs.txt` | canonical song id list |
| `kaggle_users.txt` | canonical user id list |
| `track_metadata.db` | SQLite, 1M track metadata rows |
| `taste_profile_song_to_tracks.txt` | song_id -> track_id mapping |

See [`millionsongdataset.com`](http://millionsongdataset.com) for download links.
