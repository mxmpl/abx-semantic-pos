from importlib import resources
from pathlib import Path
from typing import Literal

import polars as pl
from fastabx import Dataset, Score, Task
from fastabx.distance import DistanceName
from fastabx.pooling import pooling


def read_words(source: str | Path) -> pl.DataFrame:
    schema = {"file": pl.String, "onset": pl.String, "offset": pl.String, "word": pl.String}
    df = pl.read_csv(source, has_header=False, separator=" ", schema=schema)
    return df.with_columns(
        df["onset"].str.to_decimal(inference_length=len(df)),
        df["offset"].str.to_decimal(inference_length=len(df)),
    )


def read_triplets(task: Literal["semantic", "pos"], split: Literal["dev", "test"]) -> pl.DataFrame:
    return pl.read_ndjson(str(resources.files(__package__) / f"assets/{task}-{split}.jsonl.zst"))


def simple_abx_with_pooling(dataset: Dataset, *, distance_name: DistanceName = "angular") -> Score:
    return Score(Task(pooling(dataset, "mean"), on="index"), distance_name)


def abx_pos(
    root: str | Path,
    words: str | Path,
    *,
    frequency: float = 50,
    split: Literal["dev", "test"] = "test",
) -> float:
    dataset = ...
    distance_name = ...
    score = simple_abx_with_pooling(dataset, distance_name=distance_name)
    return score.collapse()


def abx_semantic(
    root: str | Path,
    words: str | Path,
    *,
    frequency: float = 50,
    split: Literal["dev", "test"] = "test",
) -> float:
    dataset = ...
    distance_name = ...
    score = simple_abx_with_pooling(dataset, distance_name=distance_name)
    return score.collapse()
