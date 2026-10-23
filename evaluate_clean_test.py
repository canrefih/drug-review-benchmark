
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

CONDITIONS = [
    "Birth Control",
    "Depression",
    "High Blood Pressure",
    "Diabetes, Type 2",
]

RUN_DIR = Path("artifacts/runs/20261021T092000Z-4fedaf")
OUTPUT_DIR = RUN_DIR / "clean_test_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare(path):
    df = pd.read_csv(path, sep="\t")
    df = df[["review", "condition"]].dropna().copy()
    df["review"] = df["review"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()
    df = df[
        (df["review"] != "")
        & (df["condition"].isin(CONDITIONS))
    ].copy()

    # Karşılaştırma için normalize edilmiş metin anahtarı
    df["review_key"] = df["review"].str.lower()

    # Aynı metin ve aynı etiket tekrarlarını kaldır
    return df.drop_duplicates(subset=["review_key", "condition"])


train = prepare("data/raw/drugsComTrain_raw.tsv")
test = prepare("data/raw/drugsComTest_raw.tsv")

# Bir metin birden fazla etikete sahipse değerlendirmeden çıkar.
test_label_counts = test.groupby("review_key")["condition"].nunique()
ambiguous_test_keys = set(test_label_counts[test_label_counts > 1].index)
test = test[~test["review_key"].isin(ambiguous_test_keys)].copy()

# Eğitimde görülen metinleri testten çıkar.
train_keys = set(train["review_key"])
before_overlap_filter = len(test)
test = test[~test["review_key"].isin(train_keys)].copy()

if test.empty:
    raise ValueError("Örtüşme temizliğinden sonra test kümesi boş kaldı.")

X_test = test["review"].tolist()
y_test = test["condition"].tolist()

print("=== TEMİZ TEST DEĞERLENDİRMESİ ===")
print(f"Testte kalan örnek: {len(test):,}")
print(f"Çıkarılan örtüşen test satırı: {before_overlap_filter - len(test):,}")
print("\nSınıf dağılımı:")
print(test["condition"].value_counts().to_string())

summary = []

for model_path in sorted(RUN_DIR.glob("*.joblib")):
    model_name = model_path.stem
    model = load(model_path)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, labels=CONDITIONS,
        average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, labels=CONDITIONS,
        average="weighted", zero_division=0
    )

    summary.append({
        "model": model_name,
        "evaluation": "clean_unseen_texts",
        "n_test": len(test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    report = classification_report(
        y_test,
        predictions,
        labels=CONDITIONS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / f"{model_name}_classification_report.csv"
    )

    cm = confusion_matrix(y_test, predictions, labels=CONDITIONS)
    pd.DataFrame(
        cm,
        index=CONDITIONS,
        columns=CONDITIONS,
    ).to_csv(OUTPUT_DIR / f"{model_name}_confusion_matrix.csv")

    pd.DataFrame({
        "review": X_test,
        "actual": y_test,
        "predicted": predictions,
    }).to_csv(
        OUTPUT_DIR / f"{model_name}_predictions.csv",
        index=False,
    )

results = pd.DataFrame(summary).sort_values("macro_f1", ascending=False)
results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\n=== MODEL KARŞILAŞTIRMASI ===")
print(results.to_string(index=False))
print(f"\nRaporlar kaydedildi: {OUTPUT_DIR}")
