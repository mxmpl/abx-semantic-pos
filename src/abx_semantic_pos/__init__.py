from typing import Literal
from pathlib import Path

import polars as pl
from fastabx import Dataset, Score, Task
from fastabx.distance import DistanceName
from fastabx.pooling import pooling
from importlib import resources

__all__ = ["abx_part_of_speech", "abx_semantic"]


def read_words(source: str | Path) -> pl.DataFrame:
    schema = {"file": pl.String, "onset": pl.String, "offset": pl.String, "word": pl.String}
    df = pl.read_csv(source, has_header=False, separator=" ", schema=schema)
    return df.with_columns(
        df["onset"].str.to_decimal(inference_length=len(df)),
        df["offset"].str.to_decimal(inference_length=len(df)),
    )


def read_triplets(task: Literal["semantic", "pos"], split: Literal["dev", "test"]):
    return pl.read_ndjson(str(resources.files(__package__) / f"assets/{task}-{split}.jsonl.zst"))


def simple_abx_with_pooling(dataset: Dataset, *, on: str, distance_name: DistanceName = "angular") -> Score:
    return Score(Task(pooling(dataset, "mean"), on=on), distance_name)


def abx_pos(
    root: str | Path,
    words: str | Path,
    *,
    frequency: float = 50,
    split: Literal["dev", "test"] = "test",
) -> float:
    dataset = ...
    distance_name = ...
    score = simple_abx_with_pooling(dataset, on="pos_idx", distance_name=distance_name)
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
    score = simple_abx_with_pooling(dataset, on="synonym_idx", distance_name=distance_name)
    return score.collapse()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ABX semantic or part-of-speech")
    parser.add_argument("task", type=str, choices=["semantic", "pos"], help="Task, either 'pos' or 'semantic'.")
    parser.add_argument("root", type=Path, help="Path to the root directory of the features or the units file")
    parser.add_argument("words", type=Path, help="Path to directory with word annotations")
    parser.add_argument("--frequency", type=float, default=50, help="Feature frequency")
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="test", help="Which split to consider")
    args = parser.parse_args()

    match args.task:
        case "semantic":
            score = abx_semantic(args.root, args.words, frequency=args.frequency, split=args.split)
        case "pos":
            score = abx_pos(args.root, args.words, frequency=args.frequency, split=args.split)
        case _:
            parser.error("Invalid task")
    print(score)
