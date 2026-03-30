from importlib import resources
from pathlib import Path
from typing import Literal

import polars as pl
from fastabx import Dataset, Score, Subsampler, Task
from fastabx.pooling import pooling


def read_words(source: str | Path) -> pl.DataFrame:
    schema = {"file": pl.String, "onset": pl.String, "offset": pl.String, "word": pl.String}
    df = pl.read_csv(source, has_header=False, separator=" ", schema=schema)
    return df.with_columns(
        df["onset"].str.to_decimal(inference_length=len(df)),
        df["offset"].str.to_decimal(inference_length=len(df)),
    )


def read_triplets(task: Literal["semantic", "pos"], split: Literal["dev", "test"]) -> pl.DataFrame:
    return pl.read_ndjson(str(resources.files(__package__) / f"assets/{task}-{split}.jsonl.zst")).explode("b")


def build_cells(triplets: pl.DataFrame, words: pl.DataFrame, *, threshold: int, seed: int) -> pl.DataFrame:
    idx = words.lazy().with_row_index().group_by("word", maintain_order=True).agg("index")
    cells = (
        triplets.lazy()
        .join(idx, left_on="a", right_on="word")
        .rename({"index": "index_a"})
        .join(idx, left_on="b", right_on="word")
        .rename({"index": "index_b"})
        .join(idx, left_on="x", right_on="word")
        .rename({"index": "index_x"})
    )
    return Subsampler(max_size_group=threshold, max_x_across=None, seed=seed)(cells, with_across=False).collect()


def build_dataset(path_features: str | Path, cells: pl.DataFrame, words: pl.DataFrame, *, frequency: float) -> Dataset:
    idx_used = (
        pl.concat(
            [
                cells.lazy().select(pl.col(col).explode()).rename({col: "index"})
                for col in ["index_a", "index_b", "index_x"]
            ]
        )
        .unique()
        .sort("index")
        .collect()
    )


def abx_with_predefined_triplets(
    triplets: pl.DataFrame,
    path_features: str | Path,
    path_words: str | Path,
    *,
    frequency: float,
    threshold: int,
    seed: int,
) -> float:
    words = read_words(path_words)
    cells = build_cells(triplets, words, threshold=threshold, seed=seed)
    dataset = build_dataset(path_features, cells, words, frequency=frequency)
    task = Task(pooling(dataset, "mean"), on="label", cells=cells)
    task.is_symmetric = False
    return Score(task, "angular").collapse()


def abx_pos(
    path_features: str | Path,
    path_words: str | Path,
    *,
    split: Literal["dev", "test"] = "test",
    frequency: float = 50,
    threshold: int = 10,
    seed: int = 0,
) -> float:
    return abx_with_predefined_triplets(
        read_triplets("pos", split),
        path_features,
        path_words,
        frequency=frequency,
        threshold=threshold,
        seed=seed,
    )


def abx_semantic(
    path_features: str | Path,
    path_words: str | Path,
    *,
    split: Literal["dev", "test"] = "test",
    frequency: float = 50,
    threshold: int = 10,
    seed: int = 0,
) -> float:
    return abx_with_predefined_triplets(
        read_triplets("semantic", split),
        path_features,
        path_words,
        frequency=frequency,
        threshold=threshold,
        seed=seed,
    )
