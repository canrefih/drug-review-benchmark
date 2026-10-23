import json
from pathlib import Path

import pandas as pd
import yaml

from src.experiment import run_benchmark


def test_benchmark_uses_clean_unseen_test(tmp_path):
    train = pd.DataFrame({
        "review": [
            "Shared review",
            "Train depression",
            "Train birth control",
        ],
        "condition": [
            "Depression",
            "Depression",
            "Birth Control",
        ],
    })

    test = pd.DataFrame({
        "review": [
            "SHARED REVIEW",
            "Unseen depression",
            "Unseen birth control",
        ],
        "condition": [
            "Depression",
            "Depression",
            "Birth Control",
        ],
    })

    train_path = tmp_path / "train.tsv"
    test_path = tmp_path / "test.tsv"

    train.to_csv(train_path, sep="\t", index=False)
    test.to_csv(test_path, sep="\t", index=False)

    config = {
        "data": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "text_column": "review",
            "label_column": "condition",
            "conditions": ["Depression", "Birth Control"],
        },
        "experiment": {
            "seed": 42,
            "test_size": 0.2,
            "train_fractions": [],
            "max_features": 100,
            "ngram_range": [1, 1],
            "min_df": 1,
            "output_dir": str(tmp_path / "runs"),
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = Path(run_benchmark(config_path))

    metadata = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["evaluation_source"] == "provided_labeled_test_clean_unseen"
    assert metadata["evaluation_rows_after_filtering"] == 2
    assert metadata["clean_test_stats"]["overlapping_test_rows_removed"] == 1
    assert metadata["clean_test_stats"]["clean_test_rows"] == 2