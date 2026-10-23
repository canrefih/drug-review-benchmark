from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from src.dashboard_results import load_result_summary, is_comparable_run

st.set_page_config(
    page_title="Drug Review NLP Benchmark",
    page_icon="📊",
    layout="wide",
)

st.title("Drug Review NLP Benchmark")
st.caption(
    "Reproducible model comparisons · "
    "Dataset condition classification · Research use only"
)

runs_root = Path("artifacts/runs")
runs = (
    sorted(
        [
            p for p in runs_root.iterdir()
            if p.is_dir()
            and (p / "metrics.json").exists()
        ],
        reverse=True,
    )
    if runs_root.exists()
    else []
)

if not runs:
    st.info(
        "No benchmark results found. Run "
        "`python -m src.cli run --config configs/default.yaml` "
        "from the project root."
    )
    st.markdown(
        "This dashboard displays actual experiment results. "
        "No synthetic scores are shown."
    )
    st.stop()

run_names = [p.name for p in runs]
selected_name = st.sidebar.selectbox("Experiment run", run_names)
run_dir = runs_root / selected_name

metrics_path = run_dir / "metrics.json"
metadata_path = run_dir / "run_metadata.json"

if not metrics_path.exists():
    st.error("metrics.json was not found in this experiment directory.")
    st.stop()

metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
metadata = (
    json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata_path.exists()
    else {}
)

# Transformer runs store evaluation metadata in metrics.json.
if not metadata and "evaluation_source" in metrics_json:
    metadata = {
        "evaluation_source": metrics_json["evaluation_source"],
        "evaluation_rows_after_filtering": metrics_json.get(
            "evaluation_rows", "N/A"
        ),
        "classes": metrics_json.get("classes", []),
        "clean_test_stats": metrics_json.get("clean_test_stats"),
    }

summary = load_result_summary(metrics_path)

if summary.empty:
    st.warning("No model comparison results found for this experiment.")
    st.stop()

st.subheader("Experiment overview")

c1, c2 = st.columns(2)

c1.metric("Number of classes", len(metadata.get("classes", [])))
c2.metric(
    "Best model",
    metrics_json.get("best_model_by_macro_f1", "N/A"),
)

st.subheader("Model comparison")

st.download_button(
    label="Download model comparison CSV",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_name}_model_comparison.csv",
    mime="text/csv",
)

if {"model", "accuracy", "macro_f1"}.issubset(summary.columns):
    ranked_models = summary.sort_values(
        "macro_f1", ascending=False
    ).reset_index(drop=True)

    best_model = ranked_models.iloc[0]
    worst_model = ranked_models.iloc[-1]

    macro_f1_gap = (
        best_model["macro_f1"] - worst_model["macro_f1"]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Top-performing model",
        best_model["model"],
    )

    c2.metric(
        "Best Macro-F1",
        f"{best_model['macro_f1']:.4f}",
    )

    c3.metric(
        "Macro-F1 performance gap",
        f"{macro_f1_gap:.4f}",
    )

    leaderboard = ranked_models[
        ["model", "accuracy", "macro_f1"]
    ].copy()

    leaderboard.insert(
        0,
        "rank",
        range(1, len(leaderboard) + 1),
    )

    st.dataframe(
        leaderboard,
        width="stretch",
        hide_index=True,
    )

display_cols = [
    c for c in [
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_recall",
        "roc_auc_ovr_macro",
    ]
    if c in summary.columns
]

st.dataframe(
    summary[display_cols].sort_values(
        "macro_f1", ascending=False
    ),
    width="stretch",
    hide_index=True,
)

if {"macro_f1", "weighted_f1", "model"}.issubset(summary.columns):
    st.bar_chart(
        summary.set_index("model")[["macro_f1", "weighted_f1"]]
    )

curve_path = run_dir / "learning_curve.csv"

if curve_path.exists():
    curve = pd.read_csv(curve_path)
    st.subheader("Learning curves")

    if (
        not curve.empty
        and {"train_fraction", "macro_f1", "model"}.issubset(curve.columns)
    ):
        curve = curve.sort_values("train_fraction")
        st.line_chart(
            curve.pivot(
                index="train_fraction",
                columns="model",
                values="macro_f1",
            )
        )
        st.caption(
            "The X-axis represents the fraction of training data used. "
            "The  dataset remains unchanged."
        )

st.subheader("Per-class analysis")

model_files = sorted(run_dir.glob("*_per_class_metrics.csv"))

if model_files:
    model_choice = st.selectbox(
        "Model for per-class metrics",
        [
            p.name.replace("_per_class_metrics.csv", "")
            for p in model_files
        ],
    )

    class_report = pd.read_csv(
        run_dir / f"{model_choice}_per_class_metrics.csv"
    )

    st.dataframe(
        class_report,
        width="stretch",
        hide_index=True,
    )

cm_files = sorted(run_dir.glob("*_confusion_matrix.csv"))

if cm_files:
    cm_choice = st.selectbox(
        "Model for confusion matrix",
        [
            p.name.replace("_confusion_matrix.csv", "")
            for p in cm_files
        ],
    )

    cm = pd.read_csv(
        run_dir / f"{cm_choice}_confusion_matrix.csv",
        index_col=0,
    )

    st.dataframe(cm, width="stretch")

    normalize_cm = st.checkbox(
        "Normalize confusion matrix by actual class",
        value=False,
    )

    if normalize_cm:
        row_totals = cm.sum(axis=1).replace(0, 1)
        plot_cm = cm.div(row_totals, axis=0) * 100
    else:
        plot_cm = cm

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(
        plot_cm,
        annot=True,
        fmt=".1f" if normalize_cm else "d",
        cmap="Blues",
        linewidths=0.5,
        vmin=0,
        vmax=100 if normalize_cm else None,
        cbar_kws={
            "label": "Percentage (%)" if normalize_cm else "Number of reviews"
        },
        ax=ax,
    )

    ax.set_title(
        f"Confusion Matrix — {cm_choice}"
        + (" (Row-normalized %)" if normalize_cm else "")
    )
    ax.set_xlabel("Predicted condition")
    ax.set_ylabel("Actual condition")

    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

st.subheader("Experiment metadata")

with st.expander("Configuration, dataset hashes, and environment"):
    st.json(metadata)

st.subheader("Test dataset cleaning")

clean_stats = metadata.get("clean_test_stats")

if clean_stats:
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Original test samples",
        clean_stats.get("test_rows_before_cleaning", "N/A"),
    )

    c2.metric(
        "Overlapping samples removed",
        clean_stats.get("overlapping_test_rows_removed", "N/A"),
    )

    c3.metric(
        "Clean test samples",
        clean_stats.get("clean_test_rows", "N/A"),
    )

    st.caption(
        "Overlapping and ambiguous review texts are excluded "
        "to reduce evaluation leakage."
    )
else:
    st.info("Cleaning statistics are unavailable for this experiment.")

st.subheader("Prediction error analysis")

prediction_files = sorted(run_dir.glob("*_predictions.csv"))

if prediction_files:
    error_model = st.selectbox(
        "Model for error analysis",
        [
            p.name.replace("_predictions.csv", "")
            for p in prediction_files
        ],
    )

    predictions = pd.read_csv(
        run_dir / f"{error_model}_predictions.csv"
    )

    required_columns = {"text", "actual", "predicted"}

    if required_columns.issubset(predictions.columns):
        errors = predictions.loc[
            predictions["actual"] != predictions["predicted"]
        ].copy()

        if not errors.empty:
            st.subheader("Most common misclassifications")

            error_pairs = (
                errors.groupby(["actual", "predicted"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )

            error_pairs["confusion"] = (
                error_pairs["actual"]
                + " → "
                + error_pairs["predicted"]
            )

            st.bar_chart(
                error_pairs.head(10).set_index("confusion")["count"],
                horizontal=True,
            )
        else:
            st.success("No incorrect predictions found.")

        total = len(predictions)
        error_count = len(errors)
        error_rate = error_count / total if total else 0.0

        c1, c2 = st.columns(2)
        c1.metric("Incorrect predictions", error_count)
        c2.metric("Error rate", f"{error_rate:.2%}")

        st.subheader("Explore incorrect predictions")

        available_classes = sorted(errors["actual"].dropna().unique())

        selected_class = st.selectbox(
            "Filter by actual condition",
            ["All classes", *available_classes],
        )

        filtered_errors = (
            errors
            if selected_class == "All classes"
            else errors[errors["actual"] == selected_class]
        )

        st.caption(f"Showing {len(filtered_errors)} incorrect predictions")

        st.download_button(
            label="Download filtered prediction errors CSV",
            data=filtered_errors[
                ["actual", "predicted", "text"]
            ].to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{selected_name}_{error_model}_errors.csv",
            mime="text/csv",
        )

        st.dataframe(
            filtered_errors[["actual", "predicted", "text"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("Prediction file has missing required columns.")
else:
    st.info("No prediction files found for this experiment.")

st.subheader("Cross-experiment leaderboard")

st.caption(
    "Comparisons are restricted to experiments with verified "
    "clean-test statistics. Exact test-set fingerprinting "
    "will be added to future runs."
)

all_results = []

for experiment_dir in runs:
    experiment_metrics_path = experiment_dir / "metrics.json"

    try:
        experiment_metrics = json.loads(
            experiment_metrics_path.read_text(encoding="utf-8")
        )

        if isinstance(experiment_metrics.get("models"), list):
            experiment_models = experiment_metrics["models"]

        elif all(
            key in experiment_metrics
            for key in ("model", "accuracy", "macro_f1", "weighted_f1")
        ):
            experiment_models = [experiment_metrics]

        else:
            continue

        experiment_metadata_path = experiment_dir / "run_metadata.json"

        experiment_metadata = (
            json.loads(experiment_metadata_path.read_text(encoding="utf-8"))
            if experiment_metadata_path.exists()
            else {}
        )

        if not is_comparable_run(experiment_metrics, experiment_metadata):
            continue

        for model_result in experiment_models:
            all_results.append({
                "experiment": experiment_dir.name,
                "model": model_result["model"],
                "accuracy": model_result["accuracy"],
                "macro_f1": model_result["macro_f1"],
                "weighted_f1": model_result["weighted_f1"],
                "evaluation_source": (
                    experiment_metrics.get("evaluation_source")
                    or (
                        json.loads(
                            (experiment_dir / "run_metadata.json").read_text(
                                encoding="utf-8"
                            )
                        ).get("evaluation_source", "unknown")
                        if (experiment_dir / "run_metadata.json").exists()
                        else "unknown"
                    )
                ),
            })

    except (OSError, ValueError, KeyError, TypeError):
        continue

if all_results:
    cross_experiment = pd.DataFrame(all_results)

    st.caption(
        "Only results evaluated on the same clean unseen test "
        "protocol are included in the fair comparison."
    )

    comparable_results = cross_experiment.copy()

    if comparable_results.empty:
        st.info("No comparable benchmark results available.")
    else:
        comparable_results = comparable_results.sort_values(
            "macro_f1", ascending=False
        ).reset_index(drop=True)

        comparable_results.insert(
            0,
            "rank",
            range(1, len(comparable_results) + 1),
        )

        st.dataframe(
            comparable_results,
            width="stretch",
            hide_index=True,
        )

        st.subheader("Cross-model performance")

        chart_data = comparable_results[
            ["model", "macro_f1", "accuracy", "weighted_f1"]
        ].copy()

        chart_data = chart_data.set_index("model")

        st.bar_chart(
            chart_data,
            horizontal=True,
        )

        if len(comparable_results) >= 2:
            best = comparable_results.iloc[0]
            runner_up = comparable_results.iloc[1]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Macro-F1 difference",
                f"{best['macro_f1'] - runner_up['macro_f1']:+.4f}",
            )

            c2.metric(
                "Accuracy difference",
                f"{(best['accuracy'] - runner_up['accuracy']) * 100:+.2f} pp",
            )

            c3.metric(
                "Weighted-F1 difference",
                f"{best['weighted_f1'] - runner_up['weighted_f1']:+.4f}",
            )
else:
    st.info("No completed experiment results available.")