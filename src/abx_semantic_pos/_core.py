from importlib import resources
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from fastabx import Dataset, Score, Subsampler, Task
from fastabx.dataset import InMemoryAccessor, find_all_files, load_data_from_item
from fastabx.pooling import pooling

__all__ = ["abx_pos", "abx_semantic", "read_triplets", "read_words"]


def read_words(source: str | Path) -> pl.DataFrame:
    """Read word annotations from a space-separated file into a DataFrame.

    Args:
        source: Path to the space-separated file with columns file, onset, offset, word.

    Returns:
        DataFrame with columns file (str), onset (Decimal), offset (Decimal), word (str).

    """
    schema = {"file": pl.String, "onset": pl.String, "offset": pl.String, "word": pl.String}
    df = pl.read_csv(source, has_header=False, separator=" ", schema=schema)
    return df.with_columns(
        df["onset"].str.to_decimal(inference_length=len(df)),
        df["offset"].str.to_decimal(inference_length=len(df)),
    )


def read_triplets(task: Literal["semantic", "pos"], split: Literal["dev", "test"]) -> pl.DataFrame:
    """Load ABX triplets from the bundled asset file for the given task and split.

    Args:
        task: Evaluation type, either ``"pos"`` (part-of-speech) or ``"semantic"``.
        split: Dataset split to load, either ``"dev"`` or ``"test"``.

    Returns:
        DataFrame of ABX triplets with columns a, b (exploded), and x.

    """
    return pl.read_ndjson(str(resources.files(__package__) / f"assets/{task}-{split}.jsonl.zst")).explode("b")


def _build_cells_and_labels(
    *,
    triplets: pl.DataFrame,
    words: pl.DataFrame,
    subsampler: Subsampler,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build subsampled cells and filtered labels DataFrames from triplets and word annotations.

    Returns:
        The subsampled cells and corresponding word labels.

    """
    idx = words.lazy().with_row_index().group_by("word", maintain_order=True).agg("index")
    cells = (
        triplets.lazy()
        .join(idx, left_on="a", right_on="word", maintain_order="left")
        .rename({"index": "index_a"})
        .join(idx, left_on="b", right_on="word", maintain_order="left")
        .rename({"index": "index_b"})
        .join(idx, left_on="x", right_on="word", maintain_order="left")
        .rename({"index": "index_x"})
    )
    cells = subsampler(cells, with_across=False).collect()
    idx_used = (
        pl.concat(
            [
                cells.lazy().select(pl.col(col).explode()).rename({col: "index"})
                for col in ["index_a", "index_b", "index_x"]
            ],
        )
        .unique()
        .sort("index")
        .collect()
    )
    words = (
        words.with_row_index(name="orig_index")
        .filter(pl.col("orig_index").is_in(idx_used["index"].implode()))
        .with_row_index()
    )
    mapping = {row["orig_index"]: row["index"] for row in words[["index", "orig_index"]].to_dicts()}
    cells = (
        cells.with_columns(
            pl.col(col).list.eval(pl.element().replace_strict(mapping)) for col in ["index_a", "index_b", "index_x"]
        )
        .with_columns(header=pl.concat_str(pl.lit("A="), "a", pl.lit(" B="), "b", pl.lit(" X="), "x"))
        .with_columns(description="header")
    )
    return cells, words.drop("orig_index", "index")


def _abx_with_triplets(
    triplets: pl.DataFrame,
    path_features: str | Path,
    path_words: str | Path,
    *,
    frequency: int,
    threshold: int,
    seed: int,
) -> float:
    """Run ABX evaluation on preloaded triplets and return the angular score.

    Returns:
        The collapsed ABX error-rate.

    """
    words = read_words(path_words)
    subsampler = Subsampler(max_size_group=threshold, max_x_across=None, seed=seed)
    cells, labels = _build_cells_and_labels(triplets=triplets, words=words, subsampler=subsampler)
    paths = find_all_files(path_features, ".pt")
    indices, data = load_data_from_item(paths, labels, frequency, torch.load, "file", "onset", "offset")
    dataset = Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))
    task = Task(pooling(dataset, "mean"), on="word", cells=cells)
    task.is_symmetric = False
    return Score(task, "angular").collapse(levels=[])


def abx_pos(
    path_features: str | Path,
    path_words: str | Path,
    *,
    split: Literal["dev", "test"] = "test",
    frequency: int = 50,
    threshold: int = 10,
    seed: int = 0,
) -> float:
    """Compute the ABX part-of-speech score for the given features.

    Args:
        path_features: Path to the directory containing feature ``.pt`` files.
        path_words: Path to the word annotations file.
        split: Dataset split to evaluate on, either ``"dev"`` or ``"test"``.
        frequency: Feature frequency used when loading data.
        threshold: Maximum group size for subsampling.
        seed: Random seed for subsampling.

    Returns:
        ABX score (angular distance, lower is better).

    """
    triplets = read_triplets("pos", split)
    return _abx_with_triplets(triplets, path_features, path_words, frequency=frequency, threshold=threshold, seed=seed)


def abx_semantic(
    path_features: str | Path,
    path_words: str | Path,
    *,
    split: Literal["dev", "test"] = "test",
    frequency: int = 50,
    threshold: int = 10,
    seed: int = 0,
) -> float:
    """Compute the ABX semantic score for the given features.

    Args:
        path_features: Path to the directory containing feature ``.pt`` files.
        path_words: Path to the word annotations file.
        split: Dataset split to evaluate on, either ``"dev"`` or ``"test"``.
        frequency: Feature frequency used when loading data.
        threshold: Maximum group size for subsampling.
        seed: Random seed for subsampling.

    Returns:
        ABX score (angular distance, lower is better).

    """
    triplets = read_triplets("semantic", split)
    return _abx_with_triplets(triplets, path_features, path_words, frequency=frequency, threshold=threshold, seed=seed)
