import json
import pytest

from src.dashboard_results import (
    load_result_summary,
    is_comparable_run,
    fingerprint_test_data,
)


def test_classic_ml_results(tmp_path):
    metrics_path = tmp_path / "metrics.json"

    metrics_path.write_text(
        json.dumps({
            "models": [
                {
                    "model": "tfidf_linear_svm",
                    "accuracy": 0.96,
                    "macro_f1": 0.94,
                    "weighted_f1": 0.96,
                }
            ]
        }),
        encoding="utf-8",
    )

    summary = load_result_summary(metrics_path)

    assert len(summary) == 1
    assert summary.iloc[0]["model"] == "tfidf_linear_svm"


def test_transformer_results(tmp_path):
    metrics_path = tmp_path / "metrics.json"

    metrics_path.write_text(
        json.dumps({
            "model": "distilbert-base-uncased",
            "accuracy": 0.95,
            "macro_f1": 0.93,
            "weighted_f1": 0.95,
        }),
        encoding="utf-8",
    )

    summary = load_result_summary(metrics_path)

    assert len(summary) == 1
    assert summary.iloc[0]["model"] == "distilbert-base-uncased"


def test_invalid_results(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")

    summary = load_result_summary(metrics_path)

    assert summary.empty




CLEAN_FINGERPRINT = (
    "a52ea065995c4e050dc82f97343db055423ae346b1610065c69b260a92a96fa0"
)


def test_clean_run_is_comparable():
    metrics = {"clean_test_fingerprint": CLEAN_FINGERPRINT}
    assert is_comparable_run(metrics, {})


def test_run_without_fingerprint_is_not_comparable():
    assert not is_comparable_run({}, {})


def test_different_fingerprint_is_not_comparable():
    metrics = {"clean_test_fingerprint": "different"}
    assert not is_comparable_run(metrics, {})


import pandas as pd


def test_fingerprint_is_deterministic():
    frame = pd.DataFrame({
        "review": ["Helpful medication", "No improvement"],
        "condition": ["Depression", "Diabetes, Type 2"],
    })

    assert fingerprint_test_data(frame) == fingerprint_test_data(frame)


def test_fingerprint_changes_when_content_changes():
    first = pd.DataFrame({
        "review": ["Helpful medication"],
        "condition": ["Depression"],
    })

    second = pd.DataFrame({
        "review": ["Different review"],
        "condition": ["Depression"],
    })

    assert fingerprint_test_data(first) != fingerprint_test_data(second)


def test_fingerprint_requires_columns():
    frame = pd.DataFrame({"review": ["Example"]})

    with pytest.raises(ValueError):
        fingerprint_test_data(frame)
