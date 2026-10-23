
import json

import pandas as pd
import yaml

from src.data import prepare_frame, make_clean_unseen_test
from src.dashboard_results import fingerprint_test_data


def main():
    with open("configs/default.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data = config["data"]
    text_column = data["text_column"]
    label_column = data["label_column"]
    conditions = data["conditions"]

    train_raw = pd.read_csv(data["train_path"], sep="\t")
    test_raw = pd.read_csv(data["test_path"], sep="\t")

    train = prepare_frame(
        train_raw, text_column, label_column, conditions
    )
    test = prepare_frame(
        test_raw, text_column, label_column, conditions
    )

    clean_test, stats = make_clean_unseen_test(
        train, test, text_column, label_column
    )

    fingerprint = fingerprint_test_data(
        clean_test.rename(columns={
            text_column: "review",
            label_column: "condition",
        })
    )

    result = {
        "dataset": "clean_unseen_test",
        "rows": len(clean_test),
        "sha256": fingerprint,
        "clean_test_stats": stats,
    }

    print(json.dumps(result, indent=2))

    with open("clean_test_fingerprint.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
