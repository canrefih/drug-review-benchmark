from pathlib import Path

import pandas as pd


def show_report(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    comparison_path = run_dir / "model_comparison.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Model comparison file not found: {comparison_path}"
        )

    df = pd.read_csv(comparison_path)
    columns = [
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    print("\n=== MODEL COMPARISON ===")
    print(
        df[columns]
        .sort_values("macro_f1", ascending=False)
        .round(4)
        .to_string(index=False)
    )


def show_class_report(run_dir: str | Path, model_name: str) -> None:
    run_dir = Path(run_dir)

    report_path = run_dir / f"{model_name}_per_class_metrics.csv"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Class report not found: {report_path}"
        )

    df = pd.read_csv(report_path)

    print(f"\n=== CLASS REPORT: {model_name} ===")
    print(df.round(4).to_string(index=False))


def show_confusion_matrix(
    run_dir: str | Path,
    model_name: str,
) -> None:
    path = Path(run_dir) / f"{model_name}_confusion_matrix.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Confusion matrix not found: {path}"
        )

    df = pd.read_csv(path, index_col=0)

    print(f"\n=== CONFUSION MATRIX: {model_name} ===")
    print("Rows = Actual | Columns = Predicted")
    print(df.to_string())


def show_errors(
    run_dir: str | Path,
    model_name: str,
    limit: int = 10,
) -> None:
    path = Path(run_dir) / f"{model_name}_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}"
        )

    df = pd.read_csv(path)

    errors = df[df["actual"] != df["predicted"]].copy()

    print(f"\n=== ERROR ANALYSIS: {model_name} ===")
    print(f"Total predictions: {len(df)}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Error rate: {len(errors) / len(df):.2%}")

    if errors.empty:
        print("No classification errors found.")
        return

    print("\nMost common misclassifications:")
    print(
        errors.groupby(["actual", "predicted"])
        .size()
        .sort_values(ascending=False)
        .head(limit)
        .to_string()
    )

    print("\nExample errors:")
    for _, row in errors.head(limit).iterrows():
        print(f"\nActual: {row['actual']}")
        print(f"Predicted: {row['predicted']}")
        print(f"Review: {str(row['text'])[:250]}")
        print("-" * 60)