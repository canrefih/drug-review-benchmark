
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data import prepare_frame, make_clean_unseen_test
from src.dashboard_results import fingerprint_test_data


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    actual = eval_pred.label_ids

    return {
        "accuracy": accuracy_score(actual, predictions),
        "macro_f1": f1_score(
            actual, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            actual, predictions, average="weighted", zero_division=0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate DistilBERT on a clean test dataset"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    experiment_config = config["experiment"]

    text_column = data_config["text_column"]
    label_column = data_config["label_column"]
    conditions = data_config["conditions"]
    seed = int(experiment_config["seed"])

    train = prepare_frame(
        pd.read_csv(data_config["train_path"], sep="\t"),
        text_column=text_column,
        label_column=label_column,
        conditions=conditions,
    )

    test = prepare_frame(
        pd.read_csv(data_config["test_path"], sep="\t"),
        text_column=text_column,
        label_column=label_column,
        conditions=conditions,
    )

    clean_test, clean_stats = make_clean_unseen_test(
        train_frame=train,
        test_frame=test,
        text_column=text_column,
        label_column=label_column,
    )

    if clean_test.empty:
        raise ValueError("The clean test dataset is empty.")

    print("Clean test statistics:", clean_stats)

    labels = sorted(train[label_column].unique())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    train_df, validation_df = train_test_split(
        train,
        test_size=0.1,
        random_state=seed,
        stratify=train[label_column],
    )

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_dataset(frame):
        dataset_frame = pd.DataFrame({
            "text": frame[text_column].tolist(),
            "label": frame[label_column].map(label2id).tolist(),
        })

        dataset = Dataset.from_pandas(
            dataset_frame,
            preserve_index=False,
        )

        return dataset.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=256,
            ),
            batched=True,
            remove_columns=["text"],
        )

    train_dataset = create_dataset(train_df)
    validation_dataset = create_dataset(validation_df)
    test_dataset = create_dataset(clean_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-distilbert"
    )

    output_dir = (
        Path(experiment_config.get("output_dir", "artifacts/runs"))
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    predicted_ids = np.argmax(predictions.predictions, axis=-1)
    actual_ids = clean_test[label_column].map(label2id).to_numpy()

    results = {
        "model": model_name,
        "accuracy": float(accuracy_score(actual_ids, predicted_ids)),
        "macro_f1": float(
            f1_score(
                actual_ids,
                predicted_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual_ids,
                predicted_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        "classes": labels,
        "evaluation_source": "provided_labeled_test_clean_unseen",
        "evaluation_rows": len(clean_test),
        "clean_test_stats": clean_stats,
        "clean_test_fingerprint": fingerprint_test_data(
            clean_test[["review", "condition"]]
        ),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame({
        "text": clean_test[text_column].tolist(),
        "actual": clean_test[label_column].tolist(),
        "predicted": [id2label[int(index)] for index in predicted_ids],
    }).to_csv(
        output_dir / "distilbert_predictions.csv",
        index=False,
    )

    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    print(json.dumps(results, indent=2))
    print(f"DistilBERT results saved to: {output_dir}")


if __name__ == "__main__":
    main()
