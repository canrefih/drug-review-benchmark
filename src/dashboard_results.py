
import json

import pandas as pd


def load_result_summary(metrics_path):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if isinstance(metrics.get("models"), list):
        return pd.DataFrame(metrics["models"])

    required = {"model", "accuracy", "macro_f1", "weighted_f1"}

    if required.issubset(metrics):
        return pd.DataFrame([{
            "model": metrics["model"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }])

    return pd.DataFrame()


def is_comparable_run(metrics, metadata):
    """Check whether a run uses the verified clean-test dataset."""
    expected_fingerprint = (
        "a52ea065995c4e050dc82f97343db055423ae346b1610065c69b260a92a96fa0"
    )

    fingerprint = (
        metrics.get("clean_test_fingerprint")
        or metadata.get("clean_test_fingerprint")
    )

    return fingerprint == expected_fingerprint


import hashlib


def fingerprint_test_data(frame):
    """Compute a deterministic fingerprint of test texts and labels."""
    required_columns = {"review", "condition"}

    if not required_columns.issubset(frame.columns):
        raise ValueError("Missing required test columns")

    digest = hashlib.sha256()

    rows = frame[["review", "condition"]].fillna("").astype(str)

    for review, condition in rows.itertuples(index=False, name=None):
        record = json.dumps(
            [review, condition],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()
