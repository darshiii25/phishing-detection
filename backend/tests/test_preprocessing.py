"""Unit tests for the preprocessing module using synthetic data."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    load_dataset,
    sanitize_dataset,
    create_stratified_splits,
    preprocess_pipeline,
)


@pytest.fixture
def synthetic_csv_file(tmp_path: Path) -> Path:
    """Creates a temporary synthetic CSV dataset with known anomalies."""
    data = {
        "url": [
            "http://example-phish.com/login",      # 1
            "http://safe-bank.com/home",           # 0
            "https://malicious-site.org/verify",   # 1
            "https://legit-news.com/article",      # 0
            "http://duplicate.com/page",           # 1 (duplicate 1)
            "http://duplicate.com/page",           # 1 (duplicate 2)
            "https://missing-label.com",           # NaN label
            None,                                  # Missing URL
            "http://invalid-label.com",            # label = 99
            "http://another-safe.com",             # 0
        ],
        "label": [1, 0, 1, 0, 1, 1, np.nan, 0, 99, 0],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "synthetic_dataset.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_load_dataset_success(synthetic_csv_file: Path):
    """Verifies successful loading of valid CSV."""
    df = load_dataset(synthetic_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "url" in df.columns
    assert "label" in df.columns


def test_load_dataset_missing_file():
    """Verifies FileNotFoundError on non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file_path_12345.csv")


def test_load_dataset_missing_columns(tmp_path: Path):
    """Verifies ValueError when expected columns are missing."""
    bad_df = pd.DataFrame({"wrong_col1": [1], "wrong_col2": [2]})
    file_path = tmp_path / "bad_columns.csv"
    bad_df.to_csv(file_path, index=False)

    with pytest.raises(ValueError) as exc:
        load_dataset(file_path, url_column="url", label_column="label")
    assert "missing expected column" in str(exc.value).lower()


def test_sanitize_dataset_handling(synthetic_csv_file: Path):
    """Verifies dropping of missing rows, invalid labels, and duplicates."""
    df = load_dataset(synthetic_csv_file)
    clean_df, report = sanitize_dataset(df, drop_duplicates=True)

    # Breakdown of anomalies:
    # 1 row missing label (NaN) -> dropped
    # 1 row missing URL (None) -> dropped
    # 1 row invalid label (99) -> dropped
    # 1 duplicate row -> dropped
    # Retained valid rows: 6
    assert len(clean_df) == 6
    assert report.missing_labels == 1
    assert report.missing_urls == 1
    assert report.invalid_label_rows == 1
    assert report.duplicate_urls == 1
    assert report.rows_after_cleaning == 6

    # Labels must be binary {0, 1}
    assert set(clean_df["label"].unique()).issubset({0, 1})


def test_create_stratified_splits_proportions():
    """Verifies stratified split proportions and class distribution maintenance."""
    # Create synthetic balanced dataset of 100 samples (50 legit, 50 phish)
    urls = [f"http://example{i}.com" for i in range(100)]
    labels = [0] * 50 + [1] * 50
    df = pd.DataFrame({"url": urls, "label": labels})

    splits = create_stratified_splits(
        df,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        random_seed=42,
    )

    # Check partition sizes
    assert len(splits.X_train) == 80
    assert len(splits.y_train) == 80
    assert len(splits.X_val) == 10
    assert len(splits.y_val) == 10
    assert len(splits.X_test) == 10
    assert len(splits.y_test) == 10

    # Stratification check (class balance preserved)
    assert np.isclose(splits.y_train.mean(), 0.5)
    assert np.isclose(splits.y_val.mean(), 0.5)
    assert np.isclose(splits.y_test.mean(), 0.5)


def test_imbalance_preserved_across_splits():
    """Verifies natural class imbalance (~63% legit, ~37% phish) is preserved across splits."""
    # Create synthetic imbalanced dataset of 100 samples (63 legit, 37 phish)
    urls = [f"http://sample{i}.org" for i in range(100)]
    labels = [0] * 63 + [1] * 37
    df = pd.DataFrame({"url": urls, "label": labels})

    splits = create_stratified_splits(
        df,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        random_seed=42,
    )

    # Class 0 ratio should be approx 0.63 in all splits
    train_c0_ratio = (splits.y_train == 0).mean()
    val_c0_ratio = (splits.y_val == 0).mean()
    test_c0_ratio = (splits.y_test == 0).mean()

    assert abs(train_c0_ratio - 0.63) < 0.05
    assert abs(val_c0_ratio - 0.63) < 0.10
    assert abs(test_c0_ratio - 0.63) < 0.10


def test_no_overlap_between_splits():
    """Verifies that train, validation, and test splits are strictly disjoint."""
    urls = [f"http://unique-url-{i}.com" for i in range(100)]
    labels = [0] * 60 + [1] * 40
    df = pd.DataFrame({"url": urls, "label": labels})

    splits = create_stratified_splits(
        df,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        random_seed=42,
    )

    set_train = set(splits.X_train)
    set_val = set(splits.X_val)
    set_test = set(splits.X_test)

    # Assert mutual disjointness
    assert len(set_train.intersection(set_val)) == 0
    assert len(set_train.intersection(set_test)) == 0
    assert len(set_val.intersection(set_test)) == 0
    assert len(set_train) + len(set_val) + len(set_test) == 100


def test_reproducibility_with_seed():
    """Verifies that identical random_seed yields identical splits."""
    urls = [f"http://repro-{i}.net" for i in range(100)]
    labels = [0] * 60 + [1] * 40
    df = pd.DataFrame({"url": urls, "label": labels})

    splits_a = create_stratified_splits(df, random_seed=123)
    splits_b = create_stratified_splits(df, random_seed=123)

    pd.testing.assert_series_equal(splits_a.X_train, splits_b.X_train)
    np.testing.assert_array_equal(splits_a.y_train, splits_b.y_train)
    pd.testing.assert_series_equal(splits_a.X_val, splits_b.X_val)
    np.testing.assert_array_equal(splits_a.y_val, splits_b.y_val)
    pd.testing.assert_series_equal(splits_a.X_test, splits_b.X_test)
    np.testing.assert_array_equal(splits_a.y_test, splits_b.y_test)


def test_class_count_reporting_and_summary(capsys):
    """Verifies correct calculation and formatted printing of class distribution."""
    urls = [f"http://report-{i}.org" for i in range(100)]
    labels = [0] * 70 + [1] * 30
    df = pd.DataFrame({"url": urls, "label": labels})

    splits = create_stratified_splits(df, random_seed=42)

    dist_train = splits.get_distribution("train")
    assert dist_train.total == 80
    assert dist_train.legitimate_count == 56
    assert np.isclose(dist_train.legitimate_pct, 70.0)
    assert dist_train.phishing_count == 24
    assert np.isclose(dist_train.phishing_pct, 30.0)

    # Test summary printing
    splits.print_summary()
    captured = capsys.readouterr()
    assert "PHISHING URL DETECTION" in captured.out
    assert "Cleaned Dataset" in captured.out
    assert "Train Split" in captured.out


def test_create_stratified_splits_invalid_sum():
    """Verifies ValueError when ratios do not sum to 1.0."""
    df = pd.DataFrame({"url": ["http://a.com", "http://b.com"], "label": [0, 1]})
    with pytest.raises(ValueError):
        create_stratified_splits(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)


def test_preprocess_pipeline_end_to_end(synthetic_csv_file: Path):
    """Verifies end-to-end preprocessing pipeline execution."""
    splits = preprocess_pipeline(
        synthetic_csv_file,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        drop_duplicates=True,
        random_seed=42,
        verbose=False,
    )
    total_split_rows = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
    assert total_split_rows == 6
    assert isinstance(splits.y_train, np.ndarray)
