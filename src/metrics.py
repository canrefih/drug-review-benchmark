from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def evaluate(model, x_test, y_test, labels: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "n_test": len(y_test),
    }
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    per_class = pd.DataFrame(report).T.reset_index(names="class")
    cm = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels
    )
    cm.index.name = "actual"
    cm.columns.name = "predicted"

    prediction_df = pd.DataFrame({"text": list(x_test), "actual": list(y_test), "predicted": list(predictions)})
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(x_test)
            model_classes = list(model.classes_)
            if len(model_classes) == len(labels) and set(model_classes) == set(labels):
                y_bin = label_binarize(y_test, classes=model_classes)
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr")
                )
                for idx, cls in enumerate(model_classes):
                    prediction_df[f"probability::{cls}"] = probabilities[:, idx]
        except (ValueError, AttributeError):
            metrics["roc_auc_ovr_macro"] = None
    else:
        metrics["roc_auc_ovr_macro"] = None

    return metrics, per_class, cm, prediction_df


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
