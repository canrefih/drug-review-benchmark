from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tsv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put your TSV file in data/raw/ or update configs/default.yaml."
        )
    return pd.read_csv(path, sep="\t", on_bad_lines="warn")


def prepare_frame(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    conditions: list[str] | None = None,
) -> pd.DataFrame:
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame[[text_column, label_column]].copy()
    result = result.dropna(subset=[text_column, label_column])
    result[text_column] = result[text_column].astype(str).str.strip()
    result[label_column] = result[label_column].astype(str).str.strip()
    result = result[(result[text_column] != "") & (result[label_column] != "")]
    if conditions:
        result = result[result[label_column].isin(conditions)]
    # Duplicate review texts can cross train/test boundaries; exact duplicate rows are removed
    # within each file. Cross-file overlap is reported by the experiment runner.
    result = result.drop_duplicates(subset=[text_column, label_column]).reset_index(drop=True)
    if result[label_column].nunique() < 2:
        raise ValueError("At least two target classes are required after filtering.")
    return result


def make_holdout(frame: pd.DataFrame, label_column: str, test_size: float, seed: int):
    try:
        return train_test_split(
            frame, test_size=test_size, random_state=seed, stratify=frame[label_column]
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a stratified holdout. Ensure every selected class has enough examples."
        ) from exc



def normalize_review_text(series: pd.Series) -> pd.Series:
    """Normalize review text for train/test overlap checks."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def make_clean_unseen_test(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Remove ambiguous test texts and texts already present in training data.

    Returns the cleaned test frame and counts useful for experiment metadata.
    """
    train = train_frame.copy()
    test = test_frame.copy()

    train["_review_key"] = normalize_review_text(train[text_column])
    test["_review_key"] = normalize_review_text(test[text_column])

    # A review with multiple labels is ambiguous; exclude it from evaluation.
    test_label_counts = test.groupby("_review_key")[label_column].nunique()
    ambiguous_keys = set(test_label_counts[test_label_counts > 1].index)
    ambiguous_mask = test["_review_key"].isin(ambiguous_keys)
    ambiguous_rows = int(ambiguous_mask.sum())
    test = test.loc[~ambiguous_mask].copy()

    train_keys = set(train["_review_key"])
    overlap_mask = test["_review_key"].isin(train_keys)
    overlap_rows = int(overlap_mask.sum())
    clean_test = test.loc[~overlap_mask].copy()

    stats = {
        "test_rows_before_cleaning": len(test_frame),
        "ambiguous_test_rows_removed": ambiguous_rows,
        "overlapping_test_rows_removed": overlap_rows,
        "clean_test_rows": len(clean_test),
    }

    return clean_test.drop(columns=["_review_key"]), stats
