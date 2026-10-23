import pandas as pd
import pytest

from src.data import make_clean_unseen_test, normalize_review_text, prepare_frame


def test_prepare_frame_filters_conditions_and_drops_missing():
    frame = pd.DataFrame({
        "review": ["good result", "bad result", None, "other"],
        "condition": ["Depression", "Diabetes, Type 2", "Depression", "Other"],
    })
    got = prepare_frame(frame, "review", "condition", ["Depression", "Diabetes, Type 2"])
    assert len(got) == 2
    assert set(got["condition"]) == {"Depression", "Diabetes, Type 2"}


def test_prepare_frame_requires_columns():
    with pytest.raises(ValueError):
        prepare_frame(pd.DataFrame({"review": ["text"]}), "review", "condition")


def test_make_clean_unseen_test_removes_overlap_and_ambiguous_texts():
    train = pd.DataFrame({
        "review": ["Seen review", "Another seen review"],
        "condition": ["Depression", "Birth Control"],
    })
    test = pd.DataFrame({
        "review": [
            "SEEN REVIEW",
            "New review",
            "Ambiguous review",
            "Ambiguous review",
        ],
        "condition": [
            "Depression",
            "Birth Control",
            "Depression",
            "Diabetes, Type 2",
        ],
    })

    clean, stats = make_clean_unseen_test(
        train, test, "review", "condition"
    )

    assert clean["review"].tolist() == ["New review"]
    assert stats["ambiguous_test_rows_removed"] == 2
    assert stats["overlapping_test_rows_removed"] == 1
    assert stats["clean_test_rows"] == 1


def test_clean_test_has_no_train_overlap():
    train = pd.DataFrame({
        "review": ["Great medicine", "Bad side effects"],
        "condition": ["Depression", "Birth Control"],
    })

    test = pd.DataFrame({
        "review": [
            " GREAT   MEDICINE ",
            "Completely new review",
            "Another unseen review",
        ],
        "condition": [
            "Depression",
            "Birth Control",
            "Depression",
        ],
    })

    clean, stats = make_clean_unseen_test(
        train, test, "review", "condition"
    )

    train_keys = set(normalize_review_text(train["review"]))
    clean_keys = set(normalize_review_text(clean["review"]))

    assert train_keys.isdisjoint(clean_keys)
    assert len(clean) == 2
    assert stats["overlapping_test_rows_removed"] == 1