
import pandas as pd

conditions = [
    "Birth Control",
    "Depression",
    "High Blood Pressure",
    "Diabetes, Type 2",
]

def prepare(path):
    df = pd.read_csv(path, sep="\t")
    df = df[["review", "condition"]].dropna().copy()
    df = df[df["condition"].isin(conditions)].copy()
    df["review_key"] = df["review"].astype(str).str.strip().str.lower()
    df = df[df["review_key"] != ""]
    return df.drop_duplicates(subset=["review_key", "condition"])

train = prepare("data/raw/drugsComTrain_raw.tsv")
test = prepare("data/raw/drugsComTest_raw.tsv")

train_label_counts = train.groupby("review_key")["condition"].nunique()
test_label_counts = test.groupby("review_key")["condition"].nunique()

train_conflicts = int((train_label_counts > 1).sum())
test_conflicts = int((test_label_counts > 1).sum())

train_keys = set(train["review_key"])
clean_test = test[~test["review_key"].isin(train_keys)]

print("=== VERİ BAĞIMSIZLIĞI RAPORU ===")
print(f"Eğitimde benzersiz metin: {train['review_key'].nunique():,}")
print(f"Testte benzersiz metin: {test['review_key'].nunique():,}")
print(f"Eğitimde birden fazla etiketi olan metin: {train_conflicts:,}")
print(f"Testte birden fazla etiketi olan metin: {test_conflicts:,}")
print(f"Örtüşme nedeniyle çıkarılacak test satırı: {len(test) - len(clean_test):,}")
print(f"Temiz test satırı: {len(clean_test):,}")
print("\nTemiz testteki sınıf dağılımı:")
print(clean_test["condition"].value_counts())
