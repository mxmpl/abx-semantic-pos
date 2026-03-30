from pathlib import Path

from .core import abx_pos, abx_semantic

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
    print(score)  # noqa: T201
