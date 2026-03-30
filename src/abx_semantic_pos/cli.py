import argparse
from collections.abc import Callable
from pathlib import Path

from .core import abx_pos, abx_semantic


def abx_cli(abx_fn: Callable[..., float], description: str) -> float:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("root", type=Path, help="Path to the root directory of the features or the units file")
    parser.add_argument("words", type=Path, help="Path to directory with word annotations")
    parser.add_argument("--frequency", type=float, default=50, help="Feature frequency")
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="test", help="Which split to consider")
    args = parser.parse_args()
    score = abx_fn(args.root, args.words, frequency=args.frequency, split=args.split)
    print(score)
    return score


def pos_cli() -> float:
    return abx_cli(abx_pos, "ABX syntactic (part-of-speech)")


def semantic_cli() -> float:
    return abx_cli(abx_semantic, "ABX semantic")
