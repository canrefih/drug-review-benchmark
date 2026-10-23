from __future__ import annotations

import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from joblib import dump
from sklearn.model_selection import train_test_split

from src.dashboard_results import fingerprint_test_data
from src.data import (
    load_tsv,
    make_clean_unseen_test,
    make_holdout,
    prepare_frame,
    sha256_file,
)
from src.metrics import evaluate, write_json
from src.models import build_models


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_benchmark(config_path: str | Path) -> Path:
    config = load_config(config_path)
    data_cfg, exp_cfg = config["data"], config["experiment"]
    seed = int(exp_cfg["seed"])
    np.random.seed(seed)

    train_path = Path(data_cfg["train_path"])
    test_path = Path(data_cfg["test_path"])
    train_raw = load_tsv(train_path)
    train = prepare_frame(train_raw, data_cfg["text_column"], data_cfg["label_column"], data_cfg.get("conditions"))
    has_labeled_test = test_path.exists()
    if has_labeled_test:
        test_raw = load_tsv(test_path)
        if data_cfg["label_column"] in test_raw.columns and data_cfg["text_column"] in test_raw.columns:
            test = prepare_frame(
                test_raw,
                data_cfg["text_column"],
                data_cfg["label_column"],
                data_cfg.get("conditions"),
            )

            test, clean_test_stats = make_clean_unseen_test(
                train_frame=train,
                test_frame=test,
                text_column=data_cfg["text_column"],
                label_column=data_cfg["label_column"],
            )

            print("Clean unseen test statistics:")
            print(clean_test_stats)

            if test.empty or test[data_cfg["label_column"]].nunique() < 2:
                raise ValueError(
                    "Clean unseen test set has too few rows or fewer than two classes."
                )

            fit_frame, eval_frame = train, test
            evaluation_source = "provided_labeled_test_clean_unseen"
        else:
            fit_frame, eval_frame = make_holdout(train, data_cfg["label_column"], float(exp_cfg["test_size"]), seed)
            evaluation_source = "stratified_holdout_test_file_unusable"
    else:
        fit_frame, eval_frame = make_holdout(train, data_cfg["label_column"], float(exp_cfg["test_size"]), seed)
        evaluation_source = "stratified_holdout"

    text_col, label_col = data_cfg["text_column"], data_cfg["label_column"]
    x_train, y_train = fit_frame[text_col].tolist(), fit_frame[label_col].tolist()
    x_eval, y_eval = eval_frame[text_col].tolist(), eval_frame[label_col].tolist()
    labels = sorted(set(y_train) | set(y_eval))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_dir = Path(exp_cfg.get("output_dir", "artifacts/runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(config)
    summary_rows = []
    best_name, best_macro_f1 = None, -1.0
    full_models = {}

    for model_name in models:
        # Evaluate a single full-data fit and save that model for inference.
        model = build_models(config)[model_name]
        model.fit(x_train, y_train)
        metrics, per_class, cm, pred_df = evaluate(model, x_eval, y_eval, labels)
        metrics.update({"model": model_name, "train_fraction": 1.0})
        summary_rows.append(metrics)
        per_class.to_csv(run_dir / f"{model_name}_per_class_metrics.csv", index=False)
        cm.to_csv(run_dir / f"{model_name}_confusion_matrix.csv")
        pred_df.to_csv(run_dir / f"{model_name}_predictions.csv", index=False)
        dump(model, run_dir / f"{model_name}.joblib")
        full_models[model_name] = model
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1, best_name = metrics["macro_f1"], model_name

        # Learning curve uses nested, stratified subsets of the fit split and the same evaluation set.
        for fraction in exp_cfg.get("train_fractions", [0.1, 0.25, 0.5, 0.75, 1.0]):
            fraction = float(fraction)
            if not 0 < fraction <= 1:
                continue
            if fraction >= 0.999:
                subset_x, subset_y = x_train, y_train
            else:
                try:
                    _, subset_idx = train_test_split(
                        np.arange(len(y_train)),
                        test_size=fraction,
                        random_state=seed,
                        stratify=y_train,
                    )
                    subset_x = [x_train[i] for i in subset_idx]
                    subset_y = [y_train[i] for i in subset_idx]
                except ValueError:
                    # Small fractions may not contain all classes; skip rather than distort comparison.
                    print(f"Skipping {model_name} at train fraction {fraction}: insufficient class examples.")
                    continue
            curve_model = build_models(config)[model_name]
            curve_model.fit(subset_x, subset_y)
            curve_metrics, _, _, _ = evaluate(curve_model, x_eval, y_eval, labels)
            summary_rows.append({**curve_metrics, "model": model_name, "train_fraction": fraction})

    summary = pd.DataFrame(summary_rows).drop_duplicates(subset=["model", "train_fraction"], keep="last")
    summary.to_csv(run_dir / "learning_curve.csv", index=False)
    summary[summary["train_fraction"] >= 0.999].sort_values("macro_f1", ascending=False).to_csv(
        run_dir / "model_comparison.csv", index=False
    )
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "evaluation_source": evaluation_source,
        "train_rows_after_filtering": len(fit_frame),
        "evaluation_rows_after_filtering": len(eval_frame),
        "classes": labels,
        "best_model_by_macro_f1": best_name,
        "dataset_hashes": {"train": sha256_file(train_path), "test": sha256_file(test_path) if test_path.exists() else None},
        "python_version": sys.version,
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "clean_test_stats": clean_test_stats if has_labeled_test else None,
        "clean_test_fingerprint": (
            fingerprint_test_data(
                eval_frame.rename(columns={
                    data_cfg["text_column"]: "review",
                    data_cfg["label_column"]: "condition",
                })
            )
            if has_labeled_test
            else None
        ),
    }
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "metrics.json", {
        "best_model_by_macro_f1": best_name,
        "models": summary[summary["train_fraction"] >= 0.999].to_dict(orient="records"),
    })
    latest = Path(exp_cfg.get("output_dir", "artifacts/runs")).parent / "latest"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(run_dir.resolve()), encoding="utf-8")
    print(f"Benchmark complete. Results: {run_dir}")
    print(f"Best model by macro-F1: {best_name} ({best_macro_f1:.4f})")
    return run_dir
