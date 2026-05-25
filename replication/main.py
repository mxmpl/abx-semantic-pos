# ruff: noqa: T201, DOC201, DOC501
"""Verify outputs of ``abx_semantic_pos`` match the original implementation.

The original (``replication/original/abx_data/abxeval_new.py``) takes a per-word
dictionary of pre-pooled embeddings; this library takes time-indexed features
plus word-alignment annotations. We generate one synthetic set of per-word
embeddings, write it into both input formats, then run both implementations
on the same triplets and compare.

The score conventions differ:
  * Original returns *accuracy* in percent (1 if ``cos(X,A) > cos(X,B)``).
  * fastabx returns *error rate* in [0, 1] (ties counted as 0.5).
With random features ties are negligible, so ``orig_pct/100 + new_err ~= 1``.
"""

import importlib.util
import multiprocessing
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import polars as pl
import torch

from abx_semantic_pos import abx_pos, abx_semantic, read_triplets

# The original uses ``multiprocessing.Pool``. macOS defaults to "spawn", which
# re-imports the worker target by qualified name; ``parallel_abx`` lives in a
# module we loaded via ``spec_from_file_location`` and is not importable that
# way. ``fork`` inherits the parent's memory, so the module is already there.
multiprocessing.set_start_method("fork", force=True)

REPO = Path(__file__).resolve().parent
ORIG_DIR = REPO / "original" / "abx_data"

# Tasks: (library task name, original task-file prefix).
TASKS = (("semantic", "syn"), ("pos", "pos"))
SPLITS = ("dev", "test")

# Comparison config — keep small so the original (Python loop over all
# pronunciation pairs) finishes in a reasonable time.
N_PER_WORD = 4
DIM = 32
FREQUENCY = 50
SEED = 1234
THRESHOLD = 10  # max_size_group / thresh — set >= N_PER_WORD to disable subsampling.
TOL = 1e-4  # tolerance on absolute score difference.


def _load_original_module() -> ModuleType:
    """Import ``abxeval_new`` from ``replication/original/abx_data`` by file path."""
    spec = importlib.util.spec_from_file_location("_abxeval_original", ORIG_DIR / "abxeval_new.py")
    if spec is None or spec.loader is None:
        msg = f"Could not load original module from {ORIG_DIR}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_abxeval_original"] = module
    spec.loader.exec_module(module)
    return module


def _all_words(triplets: pl.DataFrame) -> set[str]:
    return set(triplets["a"].to_list()) | set(triplets["b"].to_list()) | set(triplets["x"].to_list())


def _gen_features(words: set[str], *, n: int, dim: int, seed: int) -> dict[str, np.ndarray]:
    """Generate ``n`` random feature vectors per word."""
    rng = np.random.default_rng(seed)
    return {w: rng.standard_normal((n, dim)).astype(np.float32) for w in sorted(words)}


def _write_for_original(features: dict[str, np.ndarray], out_dir: Path) -> Path:
    """Write a single ``0.npz`` mapping word -> ``(N, D)`` array."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "0.npz"
    np.savez(path, **features)  # ty: ignore[invalid-argument-type]
    return path


def _write_for_new(
    features: dict[str, np.ndarray],
    feat_dir: Path,
    words_path: Path,
    *,
    frequency: int,
) -> None:
    """Write a single stacked ``file_0.pt`` plus a space-separated word-alignment file.

    Each pronunciation occupies one frame at ``frequency`` Hz, so its onset/offset
    pair selects exactly one row of the stacked tensor.
    """
    feat_dir.mkdir(parents=True, exist_ok=True)
    step = 1.0 / frequency
    vectors: list[np.ndarray] = []
    lines: list[str] = []
    for word, batch in features.items():
        for vec in batch:
            t = len(vectors)
            onset = t * step
            offset = (t + 1) * step
            lines.append(f"file_0 {onset:.6f} {offset:.6f} {word}")
            vectors.append(vec)
    torch.save(torch.from_numpy(np.stack(vectors)), feat_dir / "file_0.pt")
    words_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_one(
    task: Literal["semantic", "pos"],
    fname: str,
    split: Literal["dev", "test"],
    tmp_root: Path,
    original: ModuleType,
) -> tuple[float, float]:
    """Generate features, run both implementations, return ``(orig_err, new_err)``."""
    print(f"[{task}/{split}] generating features...", flush=True)
    triplets = read_triplets(task, split)
    words = _all_words(triplets)
    features = _gen_features(words, n=N_PER_WORD, dim=DIM, seed=SEED)

    orig_dir = tmp_root / f"orig_{task}_{split}"
    new_feat_dir = tmp_root / f"new_{task}_{split}"
    new_words_path = tmp_root / f"words_{task}_{split}.txt"

    npz_path = _write_for_original(features, orig_dir)
    _write_for_new(features, new_feat_dir, new_words_path, frequency=FREQUENCY)

    print(f"[{task}/{split}] running original...", flush=True)
    npz = original.load_feature(str(npz_path))
    task_txt = ORIG_DIR / f"{fname}_{split}.txt"
    orig_pct = float(original.evaluate_similarity_task(str(task_txt), npz, 0, task))
    orig_err = 1.0 - orig_pct / 100.0

    print(f"[{task}/{split}] running new library...", flush=True)
    fn = abx_semantic if task == "semantic" else abx_pos
    new_err = float(
        fn(
            new_feat_dir,
            new_words_path,
            split=split,
            frequency=FREQUENCY,
            threshold=THRESHOLD,
            seed=SEED,
        )
    )
    return orig_err, new_err


def main() -> int:
    """Entry-point."""
    original = _load_original_module()
    print(
        f"Config: n_per_word={N_PER_WORD}, dim={DIM}, frequency={FREQUENCY}, threshold={THRESHOLD}, seed={SEED}",
        flush=True,
    )

    rows: list[tuple[str, str, float, float, float]] = []
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="abx-replication-") as tmp:
        tmp_root = Path(tmp)
        print(f"Working in {tmp_root}", flush=True)
        for task, fname in TASKS:
            for split in SPLITS:
                orig_err, new_err = _run_one(task, fname, split, tmp_root, original)
                diff = abs(new_err - orig_err)
                rows.append((task, split, orig_err, new_err, diff))
                if diff > TOL:
                    failures.append(f"{task}/{split}: differs by {diff:.2e}")

    header = f"{'task':<10}{'split':<6}{'orig_err':>12}{'new_err':>12}{'|Δ|':>12}"
    print()
    print(header)
    print("-" * len(header))
    for task, split, oe, ne, d in rows:
        print(f"{task:<10}{split:<6}{oe:>12.6f}{ne:>12.6f}{d:>12.2e}")

    print()
    if failures:
        print("MISMATCH:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"OK: all scores match original within {TOL:.0e}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
