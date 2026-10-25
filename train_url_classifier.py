
from pathlib import Path
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import joblib


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATASET_PATH = Path(__file__).resolve().parent / "Dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "url_classifier.joblib"


# ============================================================
# 2. DATA INGESTION AND VALIDATION
# ============================================================

print("=" * 65)
print("URL CLASSIFICATION - RANDOM FOREST ML PIPELINE")
print("=" * 65)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place Dataset.csv in the same directory as this script."
    )

df = pd.read_csv(DATASET_PATH)

# Normalize column names so URL, url, Label, and label are handled.
df.columns = df.columns.astype(str).str.strip().str.lower()

required_columns = {"url", "label"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df[["url", "label"]].copy()

# Remove rows with missing URLs or labels.
initial_rows = len(df)

df = df.dropna(subset=["url", "label"]).copy()
df["url"] = df["url"].astype(str).str.strip()
df = df[df["url"] != ""].copy()

# Reject empty datasets and datasets with only one class.
if df.empty:
    raise ValueError("No valid records remain after cleaning.")

if df["label"].nunique() < 2:
    raise ValueError("The dataset must contain at least two classes.")

# Check for inconsistent labels assigned to the same URL.
conflicting_urls = (
    df.groupby("url")["label"].nunique()
    .loc[lambda counts: counts > 1]
)

if not conflicting_urls.empty:
    raise ValueError(
        f"Found {len(conflicting_urls)} URLs with conflicting labels. "
        "Resolve these label conflicts before training."
    )

print(f"\nDataset loaded: {DATASET_PATH.name}")
print(f"Original records: {initial_rows:,}")
print(f"Records after basic cleaning: {len(df):,}")
print(f"Columns used: {df.columns.tolist()}")

# Global shuffle for reproducible data ordering.
df = df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# 3. LABEL DISTRIBUTION AND DUPLICATE DIAGNOSTICS
# ============================================================

print("\n" + "-" * 65)
print("DATASET DIAGNOSTICS")
print("-" * 65)

print("\nClass distribution:")
print(df["label"].value_counts().to_string())

print("\nClass proportions:")
print(
    (df["label"].value_counts(normalize=True) * 100)
    .round(2)
    .astype(str)
    .add("%")
    .to_string()
)

duplicate_count = int(df.duplicated(subset=["url"]).sum())

print(f"\nRepeated URL records: {duplicate_count:,}")

if duplicate_count:
    warnings.warn(
        "Repeated URLs exist. A standard stratified split may place "
        "the same URL in both train and test sets, inflating metrics. "
        "Review duplicate URLs before treating test performance as "
        "evidence of generalization."
    )


# ============================================================
# 4. STRATIFIED TRAIN-TEST SPLIT
# ============================================================

X = df["url"]
y = df["label"]

# Stratification requires enough examples of each class.
class_counts = y.value_counts()

if class_counts.min() < 2:
    raise ValueError(
        "Every class needs at least two records for a stratified "
        "train-test split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n" + "-" * 65)
print("TRAIN-TEST SPLIT")
print("-" * 65)

print(f"Training records: {len(X_train):,}")
print(f"Testing records:  {len(X_test):,}")

print("\nTraining class proportions (%):")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTesting class proportions (%):")
print((y_test.value_counts(normalize=True) * 100).round(2))


# ============================================================
# 5. BUILD THE TF-IDF + RANDOM FOREST PIPELINE
# ============================================================

# Character n-grams capture URL structure, subdomains,
# suspicious character combinations, and common patterns.
#
# The vectorizer is fitted ONLY on the training data.
# Pipeline ensures this separation is maintained.

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=10000,
                lowercase=True,
                sublinear_tf=True,
                dtype="float32",
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_features="sqrt",
                class_weight="balanced",
            ),
        ),
    ]
)


# ============================================================
# 6. TRAIN THE MODEL
# ============================================================

print("\n" + "-" * 65)
print("MODEL TRAINING")
print("-" * 65)

print("Fitting character-level TF-IDF and Random Forest...")

model.fit(X_train, y_train)

print("Training completed successfully.")


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

print("\n" + "-" * 65)
print("MODEL EVALUATION")
print("-" * 65)

y_pred = model.predict(X_test)

labels = model.named_steps["classifier"].classes_

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )
)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=labels)

print(
    pd.DataFrame(
        cm,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels],
    ).to_string()
)


# ============================================================
# 8. PRECISION, RECALL, AND F1 SUMMARY
# ============================================================

# These aggregate metrics use macro averaging, giving each
# class equal weight regardless of its frequency.

print("\nSummary Metrics:")
print(f"Macro Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Macro Recall:    {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Macro F1-score:  {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")


# ============================================================
# 9. CONFUSION MATRIX VISUALIZATION
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[str(label) for label in labels],
)

display.plot(values_format="d")
plt.title("URL Classification - Random Forest")
plt.tight_layout()
plt.show()


# ============================================================
# 10. SAVE THE TRAINED MODEL
# ============================================================

joblib.dump(model, MODEL_PATH)

print("\n" + "-" * 65)
print("PIPELINE COMPLETED")
print("-" * 65)
print(f"Trained model saved to: {MODEL_PATH.name}")
print("The saved pipeline includes TF-IDF and the classifier.")
