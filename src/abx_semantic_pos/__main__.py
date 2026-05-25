"""CLI entry-point."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from ._core import abx_pos, abx_semantic, download_words


def _add_args(parser: ArgumentParser) -> None:
    """Add common positional and optional arguments to a subcommand parser."""
    parser.add_argument("root", type=Path, help="Path to the root directory of features")
    parser.add_argument("words", type=Path, help="Path to directory with word annotations")
    parser.add_argument("--frequency", type=int, default=50, help="Feature frequency")
    parser.add_argument("--threshold", type=int, default=10, help="Maximum group size for subsampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling")
    parser.add_argument("--split", choices=["dev", "test"], default="test", help="Which split to evaluate")


def main() -> None:
    """Entry point for the ABX evaluation CLI."""
    parser = ArgumentParser(description="ABX syntactic or semantic evaluation", allow_abbrev=False)
    sub = parser.add_subparsers(dest="task", required=True)
    _add_args(sub.add_parser("pos", help="Syntactic (part-of-speech)", formatter_class=ArgumentDefaultsHelpFormatter))
    _add_args(sub.add_parser("semantic", help="Semantic", formatter_class=ArgumentDefaultsHelpFormatter))
    sub.add_parser(
        "download",
        help="Download word annotations from GitHub",
        formatter_class=ArgumentDefaultsHelpFormatter,
    ).add_argument("output", type=Path, help="Directory where the files will be saved")

    args = parser.parse_args()
    if args.task == "download":
        download_words(args.output)
    else:
        score = (abx_pos if args.task == "pos" else abx_semantic)(
            args.root,
            args.words,
            frequency=args.frequency,
            threshold=args.threshold,
            seed=args.seed,
            split=args.split,
        )
        print(f"{score}:.2%")  # noqa: T201


if __name__ == "__main__":
    main()
