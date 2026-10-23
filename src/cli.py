import argparse
from pathlib import Path

from src.experiment import run_benchmark
from src.report import (
    show_class_report,
    show_confusion_matrix,
    show_errors,
    show_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Drug review NLP benchmark"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run baseline benchmark",
    )
    run_parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Show model comparison",
    )
    report_parser.add_argument(
        "--run-dir",
        default=None,
    )
    report_parser.add_argument(
        "--model",
        default=None,
        help="Show per-class metrics for a specific model",
    )
    report_parser.add_argument(
        "--errors",
        action="store_true",
        help="Show incorrectly classified reviews",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_benchmark(args.config)

    elif args.command == "report":
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            latest = Path("artifacts/latest")
            if not latest.exists():
                raise FileNotFoundError(
                    "No latest run found. Run benchmark first."
                )
            run_dir = Path(
                latest.read_text(encoding="utf-8").strip()
            )

        show_report(run_dir)

        if args.model:
            show_class_report(run_dir, args.model)
            show_confusion_matrix(run_dir, args.model)
            if args.errors:
                show_errors(run_dir, args.model)


if __name__ == "__main__":
    main()