# ruff: noqa: T201, DOC201, DOC501
"""Verify outputs of ``abx_semantic_pos`` match the original implementation.

Layout:
  * ``provide_features`` returns one vector per ``words_df`` row, in row order.
    Default is random; swap to plug in real features.
  * ``run_original`` builds the original's standard input (a per-word dict
    written to npz and reloaded with ``original.load_feature``, the way
    ``preextracted_lm_tasks.get_features`` writes its output) and calls
    ``evaluate_similarity_task``.
  * ``run_new`` writes the new library's standard input (per-file ``.pt``
    tensors at ``frequency`` Hz) and calls ``abx_semantic`` / ``abx_pos``.

The annotation source is the union of all 7 LibriSpeech subset files (the
list that ``download_words`` ships with). That covers every triplet word
except two (``equate`` in ``pos/dev``, ``underline`` in ``semantic/dev``)
that simply have no librispeech pronunciation — ``write_filtered_task``
drops those handful of rows to match the new lib's silent inner-join drop.

Score conventions:
  * Original: *accuracy* in percent (1 if ``cos(X,A) > cos(X,B)``).
  * fastabx: *error rate* in [0, 1] (ties counted as 0.5).
With random features ties are negligible, so ``orig_pct/100 + new_err ~= 1``.
"""

import importlib.util
import io
import multiprocessing
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import polars as pl
import torch
from fastabx.dataset import item_frontiers

from abx_semantic_pos import abx_pos, abx_semantic, read_triplets, read_words

# Original uses ``multiprocessing.Pool``. macOS defaults to "spawn", which
# re-imports the worker target by qualified name; ``parallel_abx`` lives in a
# module we loaded via ``spec_from_file_location`` and is not importable that
# way. ``fork`` inherits the parent's memory, so the module is already there.
multiprocessing.set_start_method("fork", force=True)

DIM = 32
FREQUENCY = 50
SEED = 1234
THRESHOLD = 10
TOL = 5e-4

FeatureProvider = Callable[[pl.DataFrame], np.ndarray]


def load_all_words(
    path_words: str | Path,
    *,
    frequency: int = 50,
    subsets: tuple[str, ...] = ("train-clean-100", "train-clean-360", "train-other-500"),
) -> pl.DataFrame:
    """Concatenate all 7 LibriSpeech subset annotations, keep only rows in ``vocab``."""
    vocab = set()
    for task in ("semantic", "pos"):
        for split in ("dev", "test"):
            df = read_triplets(task, split)
            vocab |= set(df["a"].unique()) | set(df["b"].unique()) | set(df["x"].unique())

    words = pl.concat([read_words(Path(path_words) / s) for s in subsets]).filter(pl.col("word").is_in(vocab))
    start, end, _, _ = item_frontiers(frequency, "onset", "offset")
    return words.with_columns(start, end).filter(pl.col("end") > pl.col("start"))


def _load_original_module(name: str, script: str | Path) -> ModuleType:
    """Import ``abxeval_new`` from ``replication/original/abx_data`` by file path."""
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        msg = f"Could not load original module from {script}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def random_features(words: pl.DataFrame, *, dim: int = 32, seed: int = 1234) -> np.ndarray:
    """Default feature provider: deterministic Gaussian vectors, one per ``words`` row."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((len(words), dim)).astype(np.float32)


# TODO: real feature provider — must return one vector per ``words`` row in row
# order. Wire to ``preextracted_lm_tasks.get_features`` or equivalent.


# --- Original implementation: build inputs and run ---------------------------


def build_original_features(words: pl.DataFrame, vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Build the ``dict[word] -> (N, D)`` batch dict the original consumes.

    Mirrors ``preextracted_lm_tasks.get_features``: walk pronunciations in
    source order, append per-word, ``concatenate``. We stack instead of
    concatenate since each vector is already 1-D.
    """
    grouped = words.with_row_index().group_by("word", maintain_order=True).agg(pl.col("index"))
    return {word: np.stack([vectors[i] for i in idxs]) for word, idxs in grouped.iter_rows()}


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an ``.npz`` archive compatible with ``np.load``.

    ``np.savez(file=path, **arrays)`` would collide on the literal word
    ``"file"`` in the vocab (it appears in dev-clean, train-clean-100, …).
    We bypass that by writing the zip ourselves.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", buf.getvalue())


def write_filtered_task(triplets: pl.DataFrame, vocab: set[str], path: Path) -> int:
    """Write a task file (``a x b1 b2 ...``) keeping only triplets fully covered by ``vocab``.

    Even loading all 7 librispeech subsets, a handful of triplet words have
    no pronunciation at all (``equate`` in ``pos/dev``, ``underline`` in
    ``semantic/dev``). The new library silently drops those triplets via its
    inner-join on ``word``; the original would ``KeyError`` on the first
    missing key. This function drops the same rows so both sides evaluate
    the same set.
    """
    grouped = (
        triplets.filter(pl.col("a").is_in(vocab) & pl.col("b").is_in(vocab) & pl.col("x").is_in(vocab))
        .group_by(["a", "x"], maintain_order=True)
        .agg(pl.col("b"))
    )
    lines = [f"{a} {x} {' '.join(bs)}" for a, x, bs in grouped.iter_rows()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def run_original(
    original: ModuleType,
    task: Literal["semantic", "pos"],
    split: Literal["dev", "test"],
    npz_path: Path,
    vocab: set[str],
    tmp_root: Path,
) -> float:
    """Run the original's standard ``evaluate_similarity_task`` end-to-end."""
    fname = "syn" if task == "semantic" else "pos"
    triplets = read_triplets(task, split)
    task_txt = tmp_root / f"{fname}_{split}.txt"
    kept = write_filtered_task(triplets, vocab, task_txt)
    print(f"[orig {task}/{split}] {kept} (a,x) lines after dropping librispeech-missing words", flush=True)
    features = original.load_feature(str(npz_path))
    pct = float(original.evaluate_similarity_task(str(task_txt), features, 0, task))
    return 1.0 - pct / 100.0


# --- New library: build inputs and run ---------------------------------------


def write_new_features(words: pl.DataFrame, vectors: np.ndarray, feat_dir: Path) -> None:
    """Write per-file ``.pt`` tensors filling each annotation's window with its vector.

    ``mean``-pooling over a window then returns the row's vector, so the new
    library sees the exact same per-pronunciation embedding as the original.
    """
    feat_dir.mkdir(parents=True, exist_ok=True)
    indexed = words.with_row_index()
    for (file_id,), group in indexed.group_by(["file"], maintain_order=True):
        n_frames = int(group["end"].max())  # type: ignore[arg-type]
        tensor = np.zeros((n_frames, DIM), dtype=np.float32)
        for start, end, idx in zip(group["start"], group["end"], group["index"], strict=True):
            tensor[int(start) : int(end)] = vectors[int(idx)]
        torch.save(torch.from_numpy(tensor), feat_dir / f"{file_id}.pt")


def write_words_file(words: pl.DataFrame, path: Path) -> None:
    """Write the concatenated annotation DataFrame back as a space-separated text file.

    Round-trips through the same format ``read_words`` expects (no header,
    ``file onset offset word``). This is just a re-serialization of real
    librispeech annotation rows — no synthesis.
    """
    words.select("file", "onset", "offset", "word").write_csv(path, include_header=False, separator=" ")


def run_new(
    task: Literal["semantic", "pos"],
    split: Literal["dev", "test"],
    feat_dir: Path,
    words_path: Path,
) -> float:
    """Run the new library's standard entry point."""
    print(f"[new  {task}/{split}] running...", flush=True)
    fn = abx_semantic if task == "semantic" else abx_pos
    return float(fn(feat_dir, words_path, split=split, frequency=FREQUENCY, threshold=THRESHOLD, seed=SEED))


# --- Driver -----------------------------------------------------------------


def main(provide_features: FeatureProvider = random_features) -> int:
    """Entry-point. Swap ``provide_features`` to plug in real features."""
    print(f"Config: dim={DIM}, frequency={FREQUENCY}, threshold={THRESHOLD}, seed={SEED}", flush=True)

    print("Loading annotations across 7 LibriSpeech subsets...", flush=True)
    words = load_all_words()
    print(f"  {len(words)} pronunciations across {words['word'].n_unique()} unique words", flush=True)

    vectors = provide_features(words)
    if vectors.shape != (len(words), DIM):
        msg = f"provide_features must return shape ({len(words)}, {DIM}), got {vectors.shape}"
        raise ValueError(msg)

    original = _load_original_module()

    rows: list[tuple[str, str, float, float, float]] = []
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="abx-replication-") as tmp:
        tmp_root = Path(tmp)
        feat_dir = tmp_root / "features"
        npz_path = tmp_root / "features.npz"
        words_path = tmp_root / "words.txt"

        print(f"Working in {tmp_root}; writing npz + .pt features + concatenated words...", flush=True)
        save_npz(npz_path, build_original_features(words, vectors))
        write_new_features(words, vectors, feat_dir)
        write_words_file(words, words_path)
        npz_vocab = set(words["word"].unique().to_list())

        for task in ("semantic", "pos"):
            for split in ("dev", "test"):
                orig_err = run_original(original, task, split, npz_path, npz_vocab, tmp_root)  # type: ignore[arg-type]
                new_err = run_new(task, split, feat_dir, words_path)  # type: ignore[arg-type]
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
