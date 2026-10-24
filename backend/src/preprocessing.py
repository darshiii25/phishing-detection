"""Preprocessing module for Phishing URL Detection.

Handles:
- Loading CSV dataset.
- Validation of schema and binary label integrity.
- Safe cleaning of missing values and duplicate URLs.
- Descriptive statistical reporting.
- Stratified train/validation/test partitioning.
- Strictly offline processing (never queries or fetches URLs over the network).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup module logger
logger = logging.getLogger(__name__)


@dataclass
class DatasetReport:
    """Statistical summary of dataset health and class distribution."""
    total_raw_rows: int
    missing_urls: int
    missing_labels: int
    invalid_label_rows: int
    duplicate_urls: int
    rows_after_cleaning: int
    class_0_count: int
    class_1_count: int
    class_0_ratio: float
    class_1_ratio: float

    def __str__(self) -> str:
        return (
            f"--- Dataset Health Report ---\n"
            f"Raw rows:                {self.total_raw_rows}\n"
            f"Missing URLs:            {self.missing_urls}\n"
            f"Missing labels:          {self.missing_labels}\n"
            f"Invalid labels:          {self.invalid_label_rows}\n"
            f"Duplicate URLs removed:  {self.duplicate_urls}\n"
            f"Cleaned rows:            {self.rows_after_cleaning}\n"
            f"Legitimate (0):          {self.class_0_count} ({self.class_0_ratio:.2%})\n"
            f"Phishing (1):            {self.class_1_count} ({self.class_1_ratio:.2%})\n"
            f"-----------------------------"
        )


@dataclass
class SplitDistribution:
    """Class count and percentage distribution for a specific dataset partition."""
    total: int
    legitimate_count: int
    legitimate_pct: float
    phishing_count: int
    phishing_pct: float

    def __str__(self) -> str:
        return (
            f"Total: {self.total:>6} | "
            f"Legitimate (0): {self.legitimate_count:>6} ({self.legitimate_pct:>6.2f}%) | "
            f"Phishing (1): {self.phishing_count:>6} ({self.phishing_pct:>6.2f}%)"
        )


@dataclass
class DataSplits:
    """Container holding stratified train, validation, and test partitions.

    Attributes:
        X_train: pd.Series of training URL strings.
        y_train: np.ndarray of training labels (0 or 1, int32).
        X_val:   pd.Series of validation URL strings.
        y_val:   np.ndarray of validation labels (0 or 1, int32).
        X_test:  pd.Series of test URL strings (held out for final evaluation only).
        y_test:  np.ndarray of test labels (0 or 1, int32).
        report:  DatasetReport containing data sanitization metrics.
    """
    X_train: pd.Series
    y_train: np.ndarray
    X_val: pd.Series
    y_val: np.ndarray
    X_test: pd.Series
    y_test: np.ndarray
    report: DatasetReport

    @property
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Returns sample counts for train, val, and test splits."""
        return {
            "train": (len(self.X_train),),
            "val": (len(self.X_val),),
            "test": (len(self.X_test),),
        }

    def get_distribution(self, split_name: str) -> SplitDistribution:
        """Calculates class counts and percentages for a given split name.

        Args:
            split_name: One of 'cleaned', 'train', 'val', or 'test'.

        Returns:
            SplitDistribution containing counts and percentages.
        """
        split_name = split_name.lower().strip()
        if split_name in ("cleaned", "all", "total"):
            y = np.concatenate([self.y_train, self.y_val, self.y_test])
        elif split_name == "train":
            y = self.y_train
        elif split_name in ("val", "validation"):
            y = self.y_val
        elif split_name == "test":
            y = self.y_test
        else:
            raise ValueError(f"Unknown split name '{split_name}'. Expected 'cleaned', 'train', 'val', or 'test'.")

        total = len(y)
        c0 = int((y == 0).sum())
        c1 = int((y == 1).sum())
        p0 = (c0 / total * 100.0) if total > 0 else 0.0
        p1 = (c1 / total * 100.0) if total > 0 else 0.0
        return SplitDistribution(
            total=total,
            legitimate_count=c0,
            legitimate_pct=p0,
            phishing_count=c1,
            phishing_pct=p1,
        )

    def print_summary(self) -> None:
        """Prints a formatted report of URL counts and percentages across all splits."""
        dist_all = self.get_distribution("cleaned")
        dist_train = self.get_distribution("train")
        dist_val = self.get_distribution("val")
        dist_test = self.get_distribution("test")

        print("=" * 80)
        print("          PHISHING URL DETECTION - DATASET & SPLIT CLASS DISTRIBUTION")
        print("=" * 80)
        print(f" Cleaned Dataset : {dist_all}")
        print(f" Train Split     : {dist_train}")
        print(f" Validation Split: {dist_val}")
        print(f" Test Split      : {dist_test}")
        print("=" * 80)


def load_dataset(
    file_path: Union[str, Path],
    url_column: str = "url",
    label_column: str = "label",
) -> pd.DataFrame:
    """Loads CSV dataset and verifies required columns exist.

    Args:
        file_path: Absolute or relative Path to CSV dataset.
        url_column: Name of column containing URL strings.
        label_column: Name of column containing target labels.

    Returns:
        pd.DataFrame containing the raw loaded data.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    df = pd.read_csv(path)

    missing_cols = [c for c in [url_column, label_column] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV file is missing expected column(s): {missing_cols}. "
            f"Found columns: {df.columns.tolist()}"
        )

    return df


def sanitize_dataset(
    df: pd.DataFrame,
    url_column: str = "url",
    label_column: str = "label",
    drop_duplicates: bool = True,
    strip_protocol: bool = False,
) -> Tuple[pd.DataFrame, DatasetReport]:
    """Cleans dataset and generates diagnostic report.

    Policy:
    1. Rows missing either URL or label are dropped and logged.
    2. Labels are coerced to integer {0, 1}; non-binary labels are dropped.
    3. Empty or whitespace-only URLs are dropped.
    4. Optional duplicate URL removal retains the first occurrence.
    5. No external network requests are ever made.

    Args:
        df: Raw DataFrame.
        url_column: Column name for URLs.
        label_column: Column name for labels.
        drop_duplicates: Whether to drop duplicate URLs.
        strip_protocol: Whether to strip http://, https://, and www. prefixes.

    Returns:
        Tuple of (cleaned_df, report).
    """
    total_raw = len(df)
    missing_urls = int(df[url_column].isna().sum())
    missing_labels = int(df[label_column].isna().sum())

    # Drop rows with nulls in required columns
    clean_df = df.dropna(subset=[url_column, label_column]).copy()

    # Ensure URLs are strings and strip whitespace
    clean_df[url_column] = clean_df[url_column].astype(str).str.strip()
    clean_df = clean_df[clean_df[url_column].str.len() > 0]

    # Optional prefix stripping (as noted in paper Section 4.1)
    if strip_protocol:
        clean_df[url_column] = (
            clean_df[url_column]
            .str.replace(r"^https?://", "", regex=True)
            .str.replace(r"^www\.", "", regex=True)
        )

    # Validate binary label values (must be 0 or 1)
    # Convert numeric strings or floats (e.g., 0.0, 1.0) to numeric
    clean_df[label_column] = pd.to_numeric(clean_df[label_column], errors="coerce")
    valid_labels_mask = clean_df[label_column].isin([0, 1, 0.0, 1.0])
    invalid_labels = int((~valid_labels_mask).sum())
    clean_df = clean_df[valid_labels_mask].copy()
    clean_df[label_column] = clean_df[label_column].astype(int)

    # Handle duplicates
    dup_count = 0
    if drop_duplicates:
        dup_count = int(clean_df[url_column].duplicated().sum())
        clean_df = clean_df.drop_duplicates(subset=[url_column], keep="first").copy()

    clean_df = clean_df.reset_index(drop=True)

    # Class counts
    class_counts = clean_df[label_column].value_counts()
    c0 = int(class_counts.get(0, 0))
    c1 = int(class_counts.get(1, 0))
    n_clean = len(clean_df)
    r0 = c0 / n_clean if n_clean > 0 else 0.0
    r1 = c1 / n_clean if n_clean > 0 else 0.0

    report = DatasetReport(
        total_raw_rows=total_raw,
        missing_urls=missing_urls,
        missing_labels=missing_labels,
        invalid_label_rows=invalid_labels,
        duplicate_urls=dup_count,
        rows_after_cleaning=n_clean,
        class_0_count=c0,
        class_1_count=c1,
        class_0_ratio=r0,
        class_1_ratio=r1,
    )

    logger.info(f"Dataset sanitized: {n_clean}/{total_raw} rows retained.")
    return clean_df, report


def create_stratified_splits(
    df: pd.DataFrame,
    url_column: str = "url",
    label_column: str = "label",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
    report: Optional[DatasetReport] = None,
) -> DataSplits:
    """Partitions dataset into stratified Train, Validation, and Test sets.

    Maintains class ratio identical across all three splits to avoid bias.

    Args:
        df: Cleaned DataFrame.
        url_column: Column name containing URLs.
        label_column: Column name containing binary labels.
        train_ratio: Fraction of dataset allocated to training (default: 0.80).
        val_ratio: Fraction allocated to validation (default: 0.10).
        test_ratio: Fraction allocated to test (default: 0.10).
        random_seed: Fixed seed for reproducibility.
        report: Optional DatasetReport to bundle into DataSplits.

    Returns:
        DataSplits dataclass holding X, y for train, val, and test partitions.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0. Given: {train_ratio} + {val_ratio} + {test_ratio} = {total}"
        )

    X = df[url_column]
    y = df[label_column].to_numpy(dtype=np.int32)

    # Guard: check if each class has >= 2 members for stratified splitting
    class_counts = np.bincount(y) if len(y) > 0 else []
    can_stratify_first = len(class_counts) >= 2 and np.min(class_counts) >= 2
    stratify_first = y if can_stratify_first else None

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=stratify_first,
    )

    # Second split: Partition Temp into Val and Test
    val_share_of_temp = val_ratio / temp_ratio
    temp_class_counts = np.bincount(y_temp) if len(y_temp) > 0 else []
    can_stratify_second = len(temp_class_counts) >= 2 and np.min(temp_class_counts) >= 2
    stratify_second = y_temp if can_stratify_second else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_share_of_temp),
        random_state=random_seed,
        stratify=stratify_second,
    )

    # Reset Series indices for downstream compatibility
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    if report is None:
        c0 = int((y == 0).sum())
        c1 = int((y == 1).sum())
        n = len(df)
        report = DatasetReport(
            total_raw_rows=n,
            missing_urls=0,
            missing_labels=0,
            invalid_label_rows=0,
            duplicate_urls=0,
            rows_after_cleaning=n,
            class_0_count=c0,
            class_1_count=c1,
            class_0_ratio=c0 / n if n > 0 else 0.0,
            class_1_ratio=c1 / n if n > 0 else 0.0,
        )

    return DataSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        report=report,
    )


def preprocess_pipeline(
    file_path: Union[str, Path],
    url_column: str = "url",
    label_column: str = "label",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    drop_duplicates: bool = True,
    strip_protocol: bool = False,
    random_seed: int = 42,
    verbose: bool = True,
) -> DataSplits:
    """Convenience end-to-end pipeline loading, sanitizing, and splitting data.

    Args:
        file_path: Path to Dataset.csv.
        url_column: URL column name.
        label_column: Label column name.
        train_ratio: Training proportion (default: 0.80).
        val_ratio: Validation proportion (default: 0.10).
        test_ratio: Testing proportion (default: 0.10).
        drop_duplicates: Remove duplicate URLs.
        strip_protocol: Remove URL protocol prefix.
        random_seed: Reproducibility seed.
        verbose: If True, prints formatted class distribution report across all splits.

    Returns:
        DataSplits ready for encoding and downstream modeling.
    """
    raw_df = load_dataset(file_path, url_column, label_column)
    clean_df, report = sanitize_dataset(
        raw_df,
        url_column=url_column,
        label_column=label_column,
        drop_duplicates=drop_duplicates,
        strip_protocol=strip_protocol,
    )
    splits = create_stratified_splits(
        clean_df,
        url_column=url_column,
        label_column=label_column,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        report=report,
    )
    if verbose:
        splits.print_summary()
    return splits
