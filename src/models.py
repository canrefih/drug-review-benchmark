from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_models(config: dict) -> dict[str, Pipeline]:
    exp = config["experiment"]
    vectorizer_args = {
        "max_features": int(exp.get("max_features", 100000)),
        "ngram_range": tuple(exp.get("ngram_range", [1, 2])),
        "min_df": int(exp.get("min_df", 2)),
        "strip_accents": "unicode",
        "sublinear_tf": True,
    }
    return {
        "tfidf_logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(**vectorizer_args)),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=int(exp["seed"]))),
        ]),
        "tfidf_multinomial_nb": Pipeline([
            ("tfidf", TfidfVectorizer(**vectorizer_args)),
            ("classifier", MultinomialNB()),
        ]),
        "tfidf_linear_svm": Pipeline([
            ("tfidf", TfidfVectorizer(**vectorizer_args)),
            ("classifier", LinearSVC(class_weight="balanced", random_state=int(exp["seed"]))),
        ]),
    }
