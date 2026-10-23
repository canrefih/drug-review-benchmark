from src.models import build_models


def test_models_are_pipeline_based():
    config = {"experiment": {"seed": 42, "max_features": 100, "ngram_range": [1, 2], "min_df": 1}}
    models = build_models(config)
    assert {"tfidf_logistic_regression", "tfidf_multinomial_nb", "tfidf_linear_svm"} == set(models)
    for model in models.values():
        assert hasattr(model, "fit") and hasattr(model, "predict")
