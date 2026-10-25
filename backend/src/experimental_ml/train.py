"""Train an isolated char-TF-IDF/Random Forest model on backend/data/Dataset.csv."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Dataset.csv"
MODEL = ROOT / "models" / "url_classifier.joblib"
METRICS = ROOT / "outputs" / "random_forest_metrics.json"

def main():
    df = pd.read_csv(DATA, usecols=["url", "label"]).dropna().copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[(df.url != "") & df.label.isin([0, 1])].drop_duplicates("url")
    x_train, x_rest, y_train, y_rest = train_test_split(df.url, df.label.astype(int), test_size=.2, random_state=42, stratify=df.label.astype(int))
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=.5, random_state=42, stratify=y_rest)
    model = Pipeline([("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000, lowercase=True, sublinear_tf=True, dtype="float32")), ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_features="sqrt", class_weight="balanced"))])
    model.fit(x_train, y_train)
    # Validation metrics are recorded separately; test is held out until this final evaluation.
    def score(x, y):
        pred = model.predict(x); p = model.predict_proba(x)[:, list(model.named_steps['classifier'].classes_).index(0)]
        return {"accuracy": accuracy_score(y, pred), "phishing_precision": precision_score(1-y, 1-pred, zero_division=0), "phishing_recall": recall_score(1-y, 1-pred, zero_division=0), "phishing_f1": f1_score(1-y, 1-pred, zero_division=0), "phishing_roc_auc": roc_auc_score(1-y, p), "phishing_positive_confusion_matrix_tn_fp_fn_tp": confusion_matrix(1-y, 1-pred, labels=[0,1]).ravel().tolist()}
    results = {"label_mapping": {"0":"phishing", "1":"legitimate"}, "split_seed":42, "train_rows":len(x_train), "validation":score(x_val,y_val), "held_out_test":score(x_test,y_test)}
    MODEL.parent.mkdir(parents=True, exist_ok=True); METRICS.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    METRICS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2)); print(f"Model saved: {MODEL}")

if __name__ == "__main__": main()
