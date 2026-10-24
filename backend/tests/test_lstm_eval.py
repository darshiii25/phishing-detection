"""Unit tests for LSTM classifier and evaluation metrics module."""

from pathlib import Path
import numpy as np
import pytest

from src.lstm import (
    prepare_lstm_input,
    build_lstm_classifier,
    predict_probabilities,
    predict_classes,
)
from src.evaluation import compute_metrics, plot_confusion_matrix


# ---------------------------------------------------------
# 1. LSTM Tests
# ---------------------------------------------------------

def test_prepare_lstm_input_steps_1():
    """Verifies sequence transformation (N, k) -> (N, k, 1)."""
    feats = np.random.randn(8, 12).astype(np.float32)
    seq = prepare_lstm_input(feats, sequence_mode="steps_1")
    assert seq.shape == (8, 12, 1)
    assert seq.dtype == np.float32


def test_prepare_lstm_input_step_k():
    """Verifies sequence transformation (N, k) -> (N, 1, k)."""
    feats = np.random.randn(8, 12).astype(np.float32)
    seq = prepare_lstm_input(feats, sequence_mode="step_k")
    assert seq.shape == (8, 1, 12)


def test_prepare_lstm_input_invalid_mode():
    """Verifies ValueError on invalid mode."""
    with pytest.raises(ValueError):
        prepare_lstm_input(np.zeros((5, 5)), sequence_mode="unsupported_mode")


def test_build_lstm_classifier_forward():
    """Verifies Bi-LSTM building, prediction shapes, and output ranges."""
    batch_size = 4
    timesteps = 10
    feature_dim = 1

    model = build_lstm_classifier(
        input_shape=(timesteps, feature_dim),
        lstm_units=[16, 8],
        bidirectional=True,
        dense_units=8,
    )

    X_dummy = np.random.randn(batch_size, timesteps, feature_dim).astype(np.float32)

    probs = predict_probabilities(model, X_dummy)
    assert probs.shape == (batch_size,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    preds = predict_classes(model, X_dummy, threshold=0.5)
    assert preds.shape == (batch_size,)
    assert set(np.unique(preds)).issubset({0, 1})


# ---------------------------------------------------------
# 2. Evaluation Module Tests
# ---------------------------------------------------------

def test_compute_metrics_perfect_predictions():
    """Verifies 100% scores on identical true and predicted labels."""
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_pred = np.array([0, 0, 1, 1], dtype=np.int32)
    y_probs = np.array([0.05, 0.10, 0.90, 0.95], dtype=np.float32)

    res = compute_metrics(y_true, y_pred, y_probs=y_probs)

    assert res.accuracy == 1.0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1_score == 1.0
    assert res.roc_auc == 1.0
    assert res.tp == 2
    assert res.tn == 2
    assert res.fp == 0
    assert res.fn == 0


def test_compute_metrics_known_breakdown():
    """Verifies calculation with 1 of each outcome (TN=1, FP=1, FN=1, TP=1)."""
    # 0 -> 0: TN (1)
    # 0 -> 1: FP (1)
    # 1 -> 0: FN (1)
    # 1 -> 1: TP (1)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    res = compute_metrics(y_true, y_pred)

    assert res.tn == 1
    assert res.fp == 1
    assert res.fn == 1
    assert res.tp == 1
    assert res.accuracy == 0.5
    assert res.precision == 0.5
    assert res.recall == 0.5
    assert res.f1_score == 0.5


def test_compute_metrics_mismatch_raises():
    """Verifies ValueError when y_true and y_pred lengths differ."""
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0, 1, 1])


def test_plot_confusion_matrix(tmp_path: Path):
    """Verifies rendering and saving confusion matrix plot to image."""
    cm = np.array([[50, 5], [3, 42]])
    out_file = tmp_path / "cm_test.png"

    plot_confusion_matrix(cm, save_path=out_file, title="Test Confusion Matrix")
    assert out_file.is_file()
    assert out_file.stat().st_size > 0
