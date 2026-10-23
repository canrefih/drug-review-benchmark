
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_text(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def profile_file(path, text_column, label_column):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {path}")

    df = pd.read_csv(path, sep="\t", low_memory=False)

    result = {
        "file": path.name,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(df),
        "columns": list(df.columns),
        "column_types": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_values": {
            str(k): int(v) for k, v in df.isna().sum().items()
        },
        "unique_values": {
            str(k): int(df[k].nunique(dropna=True)) for k in df.columns
        },
    }

    if label_column in df.columns:
        result["label_distribution"] = {
            str(k): int(v)
            for k, v in df[label_column].value_counts(
                dropna=False
            ).items()
        }

    if text_column in df.columns:
        text = df[text_column].fillna("").astype(str)
        nonempty = text[text.str.strip().ne("")]
        hashes = text.map(sha256_text)

        result["text_stats"] = {
            "nonempty_texts": int(text.str.strip().ne("").sum()),
            "unique_nonempty_texts": int(
                nonempty.map(sha256_text).nunique()
            ),
            "duplicate_text_rows": int(
                text.str.strip().ne("").sum()
                - nonempty.map(sha256_text).nunique()
            ),
        }
        result["_text_hashes"] = set(hashes[text.str.strip().ne("")])

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset metadata without printing review text."
    )
    parser.add_argument(
        "--train", default="data/raw/drugsComTrain_raw.tsv"
    )
    parser.add_argument(
        "--test", default="data/raw/drugsComTest_raw.tsv"
    )
    parser.add_argument("--text-column", default="review")
    parser.add_argument("--label-column", default="condition")
    parser.add_argument("--output", default="data_summary.json")
    args = parser.parse_args()

    train = profile_file(
        args.train, args.text_column, args.label_column
    )
    test = profile_file(
        args.test, args.text_column, args.label_column
    )

    train_hashes = train.pop("_text_hashes")
    test_hashes = test.pop("_text_hashes")

    summary = {
        "train": train,
        "test": test,
        "exact_review_text_overlap_count": len(
            train_hashes & test_hashes
        ),
        "note": (
            "Only metadata and label counts are saved. "
            "Review text and individual text hashes are not saved."
        ),
    }

    output = Path(args.output)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOzet dosyasi kaydedildi: {output.resolve()}")


if __name__ == "__main__":
    main()
