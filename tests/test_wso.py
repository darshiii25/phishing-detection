"""Unit tests for CNN, MLP evaluator, and White Shark Optimization (WSO).

Uses lightweight synthetic data to run swiftly without downloading external models
or training large networks.
"""

from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from src.cnn import build_cnn_models, extract_features
from src.mlp import build_mlp_classifier, MLPFitnessEvaluator
from src.wso import WhiteSharkOptimizer, WSOResult


# ---------------------------------------------------------
# 1. CNN Tests (Shapes and Feature Extraction)
# ---------------------------------------------------------

def test_cnn_model_shapes():
    """Verifies CNN classifier and feature-extractor output tensor dimensions."""
    batch_size = 4
    max_url_len = 30
    vocab_size = 50
    embedding_dim = 16
    feature_dim = 24

    clf_model, feat_extractor = build_cnn_models(
        vocab_size=vocab_size,
        max_url_len=max_url_len,
        embedding_dim=embedding_dim,
        conv_filters=[32, 16],
        kernel_sizes=[5, 3],
        pool_sizes=[2, 2],
        feature_dense_dim=feature_dim,
    )

    dummy_input = np.random.randint(0, vocab_size, size=(batch_size, max_url_len), dtype=np.int32)

    # Feature extractor must output shape (batch_size, feature_dim)
    features = feat_extractor(dummy_input)
    assert features.shape == (batch_size, feature_dim)

    # Classifier must output shape (batch_size, 1)
    predictions = clf_model(dummy_input)
    assert predictions.shape == (batch_size, 1)


def test_cnn_extract_features_helper():
    """Verifies extract_features helper function returns float32 numpy array."""
    batch_size = 6
    max_url_len = 20
    feature_dim = 16

    _, feat_extractor = build_cnn_models(
        vocab_size=40,
        max_url_len=max_url_len,
        embedding_dim=8,
        conv_filters=[16],
        kernel_sizes=[3],
        pool_sizes=[2],
        feature_dense_dim=feature_dim,
    )

    X_dummy = np.random.randint(0, 40, size=(batch_size, max_url_len), dtype=np.int32)
    feats = extract_features(feat_extractor, X_dummy, batch_size=2)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (batch_size, feature_dim)
    assert feats.dtype == np.float32


# ---------------------------------------------------------
# 2. MLP Fitness Evaluator Tests
# ---------------------------------------------------------

def test_build_mlp_classifier():
    """Verifies MLP architecture compiles and runs on synthetic features."""
    input_dim = 8
    mlp = build_mlp_classifier(input_dim=input_dim, hidden_units=[16, 8])

    dummy_X = np.random.randn(5, input_dim).astype(np.float32)
    out = mlp(dummy_X)
    assert out.shape == (5, 1)


def test_mlp_evaluator_cost_and_caching():
    """Verifies MLPFitnessEvaluator computes Cost (Eq. 25) and caches results."""
    n_train, n_val, d = 40, 20, 10
    rng = np.random.default_rng(42)

    X_train = rng.standard_normal((n_train, d)).astype(np.float32)
    y_train = rng.integers(0, 2, size=n_train).astype(np.float32)
    X_val = rng.standard_normal((n_val, d)).astype(np.float32)
    y_val = rng.integers(0, 2, size=n_val).astype(np.float32)

    evaluator = MLPFitnessEvaluator(
        X_train_features=X_train,
        y_train=y_train,
        X_val_features=X_val,
        y_val=y_val,
        hidden_units=[16],
        alpha=0.99,
        beta=0.01,
        epochs=1,
        batch_size=32,
    )

    mask = np.zeros(d, dtype=np.int32)
    mask[0] = 1
    mask[3] = 1  # 2 features selected

    cost1 = evaluator.evaluate(mask)
    assert 0.0 <= cost1 <= 1.0
    assert evaluator.evaluation_count == 1
    assert evaluator.cache_hits == 0

    # Second evaluation of the exact same mask must hit cache
    cost2 = evaluator.evaluate(mask)
    assert cost1 == cost2
    assert evaluator.evaluation_count == 1
    assert evaluator.cache_hits == 1


def test_mlp_evaluator_empty_mask_handling():
    """Verifies empty mask (all zeros) returns penalty cost 1.0 without error."""
    evaluator = MLPFitnessEvaluator(
        X_train_features=np.zeros((10, 5)),
        y_train=np.zeros(10),
        X_val_features=np.zeros((10, 5)),
        y_val=np.zeros(10),
    )
    empty_mask = np.zeros(5, dtype=np.int32)
    cost = evaluator.evaluate(empty_mask)
    assert cost == 1.0


# ---------------------------------------------------------
# 3. White Shark Optimization (WSO) Tests
# ---------------------------------------------------------

def test_wso_binarization_and_empty_prevention():
    """Verifies WSO binarize creates valid binary mask and prevents all-zeros."""
    wso = WhiteSharkOptimizer(num_features=8)
    all_negative = np.full(8, -100.0)
    mask = wso._binarize(all_negative)

    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.sum() >= 1  # Guard activated at least 1 feature


def test_wso_optimization_loop():
    """Verifies WSO optimization loop, mask dimensions, and history tracking."""
    d = 12

    # Simple synthetic objective function: target selecting exactly features [1, 4]
    def synthetic_objective(mask: np.ndarray) -> float:
        target = np.zeros(d, dtype=np.int32)
        target[1] = 1
        target[4] = 1
        error = float(np.sum(np.abs(mask - target))) / d
        return error

    wso = WhiteSharkOptimizer(
        num_features=d,
        population_size=6,
        max_iterations=3,
        random_seed=42,
    )

    result = wso.optimize(synthetic_objective, verbose=False)

    assert isinstance(result, WSOResult)
    assert len(result.best_mask) == d
    assert set(np.unique(result.best_mask)).issubset({0, 1})
    assert result.num_selected == len(result.selected_indices)
    assert np.array_equal(result.selected_indices, np.where(result.best_mask == 1)[0])
    assert len(result.history) == 3
    assert result.best_fitness >= 0.0


def test_wso_reproducibility():
    """Verifies that identical random seed produces identical results."""
    d = 6

    def simple_loss(mask: np.ndarray) -> float:
        return float(1.0 - (mask[0] * 0.5 + mask[2] * 0.5))

    wso1 = WhiteSharkOptimizer(num_features=d, population_size=4, max_iterations=2, random_seed=99)
    res1 = wso1.optimize(simple_loss, verbose=False)

    wso2 = WhiteSharkOptimizer(num_features=d, population_size=4, max_iterations=2, random_seed=99)
    res2 = wso2.optimize(simple_loss, verbose=False)

    np.testing.assert_array_equal(res1.best_mask, res2.best_mask)
    assert res1.best_fitness == res2.best_fitness


def test_wso_result_save_to_json(tmp_path: Path):
    """Verifies serializing WSOResult to JSON."""
    mask = np.array([1, 0, 1, 0], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    res = WSOResult(
        best_mask=mask,
        selected_indices=indices,
        best_fitness=0.15,
        total_features=4,
        num_selected=2,
        history=[{"iteration": 1, "best_cost": 0.15}],
    )

    out_file = tmp_path / "wso_test.json"
    res.save(out_file)
    assert out_file.is_file()
