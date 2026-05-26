"""Drive-aware checkpointing so Colab sessions don't recompute everything.

Wrap any expensive computation with `checkpoint(key, fn)` — first run computes
and saves the result; later runs load from disk.

    als = checkpoint("als_factors_v1", lambda: train_als(ui_matrix))

Pickle-based. Works for numpy arrays, scipy sparse matrices, pandas frames,
sklearn / implicit / lightgbm models, dicts of the above.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

# Override via env var or by passing dir= to checkpoint() directly.
_DEFAULT_DIR = Path(os.environ.get("MSD_CHECKPOINT_DIR", "./checkpoints"))


def set_checkpoint_dir(path: str | os.PathLike) -> None:
    """Set the default checkpoint dir for the session."""
    global _DEFAULT_DIR
    _DEFAULT_DIR = Path(path)
    _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)


def checkpoint(
    key: str,
    fn: Callable[[], T],
    *,
    dir: str | os.PathLike | None = None,
    force: bool = False,
    verbose: bool = True,
) -> T:
    """Load a cached result if it exists, else compute via fn() and save.

    Args:
        key: unique filename stem; final path is `{dir}/{key}.pkl`.
              Bump the version (e.g., "als_v1" -> "als_v2") when inputs change.
        fn: zero-arg callable that produces the result.
        dir: override the default checkpoint directory.
        force: ignore existing cache and recompute.
        verbose: print load/save messages.
    """
    ckpt_dir = Path(dir) if dir is not None else _DEFAULT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{key}.pkl"

    if path.exists() and not force:
        if verbose:
            size_mb = path.stat().st_size / 1e6
            print(f"[ckpt] loading {key}  ({size_mb:.1f} MB)")
        with open(path, "rb") as f:
            return pickle.load(f)

    if verbose:
        print(f"[ckpt] computing {key} ...")
    result = fn()
    with open(path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        size_mb = path.stat().st_size / 1e6
        print(f"[ckpt] saved {key} -> {path}  ({size_mb:.1f} MB)")
    return result


def clear(key: str, *, dir: str | os.PathLike | None = None) -> bool:
    """Delete a single checkpoint. Returns True if a file was removed."""
    ckpt_dir = Path(dir) if dir is not None else _DEFAULT_DIR
    path = ckpt_dir / f"{key}.pkl"
    if path.exists():
        path.unlink()
        return True
    return False


def list_checkpoints(dir: str | os.PathLike | None = None) -> list[tuple[str, float]]:
    """List existing checkpoints as [(name, size_mb), ...] sorted by name."""
    ckpt_dir = Path(dir) if dir is not None else _DEFAULT_DIR
    if not ckpt_dir.exists():
        return []
    out = []
    for p in sorted(ckpt_dir.glob("*.pkl")):
        out.append((p.stem, p.stat().st_size / 1e6))
    return out
