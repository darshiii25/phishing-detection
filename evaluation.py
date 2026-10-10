"""Evaluation and Performance Metrics Module for Phishing URL Detection.

Positive Class Definition:
- Label 1 = Phishing URL (Target positive class)
- Label 0 = Legitimate URL (Negative class)

Classification Outcomes:
- True Positive  (TP): Phishing URL correctly identified as Phishing.
- False Positive (FP): Legitimate URL incorrectly flagged as Phishing (false alarm).
- True Negative  (TN): Legitimate URL correctly identified as Legitimate.
- False Negative (FN): Phishing URL incorrectly classified as Legitimate (missed security threat).

Formulations (Albahadili et al., 2024, Section 4.2):
- Accuracy    = (TP + TN) / (TP + TN + FP + FN)           [Eq. 34]
- Recall/Sens = TP / (TP + FN)                            [Eq. 35]
- Precision   = TP / (TP + FP)                            [Eq. 36]
- F1-Score    = 2 * (Precision * Recall) / (Precision + Recall)
- ROC-AUC     = Area Under Receiver Operating Characteristic Curve
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


@dataclass
class EvaluationMetrics:
    """Standardized performance metrics container."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    tp: int
    fp: int
    tn: int
    fn: int
    total_samples: int

    @property
    def confusion_matrix(self) -> np.ndarray:
        """Returns 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=np.int64)

    def to_dict(self) -> Dict[str, Union[float, int, None]]:
        """Converts metrics to a structured dictionary."""
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "roc_auc": float(self.roc_auc) if self.roc_auc is not None else None,
            "true_positives": int(self.tp),
            "false_positives": int(self.fp),
            "true_negatives": int(self.tn),
            "false_negatives": int(self.fn),
            "total_samples": int(self.total_samples),
        }

    def print_report(self, dataset_name: str = "Test Set") -> None:
        """Prints a comprehensive tabular evaluation report."""
        print("=" * 70)
        print(f"        MODEL EVALUATION REPORT - {dataset_name.upper()}")
        print("=" * 70)
        print(f" Total Evaluated Samples: {self.total_samples:,}")
        print("-" * 70)
        print(f" Accuracy                : {self.accuracy:.4f} ({self.accuracy * 100:.2f}%)")
        print(f" Precision               : {self.precision:.4f} ({self.precision * 100:.2f}%)")
        print(f" Recall / Sensitivity    : {self.recall:.4f} ({self.recall * 100:.2f}%)")
        print(f" F1-Score                : {self.f1_score:.4f}")
        auc_str = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A (single class or no probs)"
        print(f" ROC-AUC                 : {auc_str}")
        print("-" * 70)
        print(" Confusion Matrix Breakdown:")
        print(f"   True Negatives  (TN)  : {self.tn:>6} (Legitimate correctly cleared)")
        print(f"   False Positives (FP)  : {self.fp:>6} (Legitimate falsely flagged as Phish)")
        print(f"   False Negatives (FN)  : {self.fn:>6} (Phishing missed - Dangerous)")
        print(f"   True Positives  (TP)  : {self.tp:>6} (Phishing correctly caught)")
        print("=" * 70)


def compute_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_probs: Optional[Union[np.ndarray, list]] = None,
) -> EvaluationMetrics:
    """Calculates all key evaluation metrics for binary classification.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_probs: Optional predicted continuous probabilities for the positive class (1).

    Returns:
        EvaluationMetrics dataclass instance.
    """
    yt = np.asarray(y_true, dtype=np.int32).flatten()
    yp = np.asarray(y_pred, dtype=np.int32).flatten()

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true ({len(yt)}) vs y_pred ({len(yp)})")

    # Confusion matrix: [[TN, FP], [FN, TP]]
    # Ensure 2x2 shape even if one class is missing in a small batch
    cm = metrics.confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = len(yt)
    acc = float((tp + tn) / total) if total > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * (prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

    # ROC-AUC calculation (only valid if both classes are present in y_true and probs provided)
    roc_auc = None
    if y_probs is not None:
        ypr = np.asarray(y_probs, dtype=np.float32).flatten()
        if len(np.unique(yt)) > 1:
            try:
                roc_auc = float(metrics.roc_auc_score(yt, ypr))
            except Exception:
                roc_auc = None

    return EvaluationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        roc_auc=roc_auc,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        total_samples=total,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Renders and optionally saves a clean visualization of the confusion matrix.

    Args:
        cm: 2x2 confusion matrix array [[TN, FP], [FN, TP]].
        save_path: Optional file path to save the generated plot image.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=f"{cm[i, j]:,}",
                va="center",
                ha="center",
                size="large",
                fontweight="bold",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate (0)", "Phishing (1)"])
    ax.set_yticklabels(["Legitimate (0)", "Phishing (1)"])
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label")
    ax.set_title(title, pad=15)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(path), dpi=150)
    plt.close(fig)
