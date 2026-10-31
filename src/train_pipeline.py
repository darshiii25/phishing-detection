"""Run the full phishing URL training pipeline from one entry point.

Order:
1. Preprocessing
2. Character encoding
3. CNN training and feature extraction
4. WSO feature selection using the MLP fitness evaluator
5. LSTM training and test evaluation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG
from src.cnn import build_cnn_models, extract_features, save_cnn_feature_extractor
from src.encoding import UrlCharacterEncoder
from src.evaluation import compute_metrics
from src.lstm import (
    build_lstm_classifier,
    predict_classes,
    predict_probabilities,
    prepare_lstm_input,
    save_lstm_model,
)
from src.mlp import MLPFitnessEvaluator
from src.preprocessing import preprocess_pipeline
from src.wso import WhiteSharkOptimizer


def main() -> None:
    config = DEFAULT_CONFIG
    config.preprocessing.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    config.encoding.vocab_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.cnn.model_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.lstm.model_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.wso.output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/5] Preprocessing dataset...")
    splits = preprocess_pipeline(
        file_path=config.preprocessing.dataset_path,
        url_column=config.preprocessing.url_column,
        label_column=config.preprocessing.label_column,
        train_ratio=config.preprocessing.train_ratio,
        val_ratio=config.preprocessing.val_ratio,
        test_ratio=config.preprocessing.test_ratio,
        drop_duplicates=config.preprocessing.drop_duplicates,
        strip_protocol=config.preprocessing.strip_protocol,
        random_seed=config.preprocessing.random_seed,
        verbose=True,
    )

    print("[2/5] Encoding URLs...")
    encoder = UrlCharacterEncoder(
        max_url_len=config.encoding.max_url_len,
        pad_token=config.encoding.padding_token,
        unk_token=config.encoding.unknown_token,
    )
    encoder.fit(splits.X_train)
    encoder.save_vocabulary(config.encoding.vocab_save_path)

    X_train_encoded = encoder.encode_batch(splits.X_train)
    X_val_encoded = encoder.encode_batch(splits.X_val)
    X_test_encoded = encoder.encode_batch(splits.X_test)

    print("[3/5] Training CNN and extracting features...")
    cnn_classifier, cnn_feature_extractor = build_cnn_models(
        vocab_size=encoder.vocab_size,
        max_url_len=config.encoding.max_url_len,
        embedding_dim=config.cnn.embedding_dim,
        conv_filters=config.cnn.conv_filters,
        kernel_sizes=config.cnn.kernel_sizes,
        pool_sizes=config.cnn.pool_sizes,
        feature_dense_dim=config.cnn.feature_dense_dim,
        dropout_rate=config.cnn.dropout_rate,
        learning_rate=config.cnn.learning_rate,
    )
    cnn_classifier.fit(
        X_train_encoded,
        splits.y_train,
        validation_data=(X_val_encoded, splits.y_val),
        epochs=config.cnn.epochs,
        batch_size=config.cnn.batch_size,
        verbose=1,
    )
    save_cnn_feature_extractor(cnn_feature_extractor, config.cnn.model_save_path)

    train_features = extract_features(cnn_feature_extractor, X_train_encoded, batch_size=config.cnn.batch_size)
    val_features = extract_features(cnn_feature_extractor, X_val_encoded, batch_size=config.cnn.batch_size)
    test_features = extract_features(cnn_feature_extractor, X_test_encoded, batch_size=config.cnn.batch_size)

    print("[4/5] Running WSO feature selection with MLP fitness evaluation...")
    evaluator = MLPFitnessEvaluator(
        X_train_features=train_features,
        y_train=splits.y_train,
        X_val_features=val_features,
        y_val=splits.y_val,
        hidden_units=config.mlp.hidden_units,
        alpha=config.wso.alpha,
        beta=config.wso.beta,
        epochs=config.mlp.epochs,
        batch_size=config.mlp.batch_size,
        patience=config.mlp.patience,
        learning_rate=config.mlp.learning_rate,
        random_seed=config.wso.random_seed,
    )
    optimizer = WhiteSharkOptimizer(
        num_features=train_features.shape[1],
        population_size=config.wso.population_size,
        max_iterations=config.wso.max_iterations,
        p_min=config.wso.p_min,
        p_max=config.wso.p_max,
        tau=config.wso.tau,
        a0=config.wso.a0,
        a1=config.wso.a1,
        a2=config.wso.a2,
        transfer_function=config.wso.transfer_function,
        random_seed=config.wso.random_seed,
    )
    wso_result = optimizer.optimize(fitness_func=evaluator.evaluate, verbose=True)
    wso_result.save(config.wso.output_mask_path)
    with open(config.wso.history_save_path, "w", encoding="utf-8") as f:
        json.dump(wso_result.history, f, indent=2)

    selected = wso_result.selected_indices
    train_selected = train_features[:, selected]
    val_selected = val_features[:, selected]
    test_selected = test_features[:, selected]

    print("[5/5] Training LSTM classifier and evaluating test set...")
    X_train_lstm = prepare_lstm_input(train_selected, sequence_mode=config.lstm.sequence_mode)
    X_val_lstm = prepare_lstm_input(val_selected, sequence_mode=config.lstm.sequence_mode)
    X_test_lstm = prepare_lstm_input(test_selected, sequence_mode=config.lstm.sequence_mode)

    lstm_model = build_lstm_classifier(
        input_shape=X_train_lstm.shape[1:],
        lstm_units=config.lstm.lstm_units,
        bidirectional=config.lstm.bidirectional,
        dropout_rate=config.lstm.dropout_rate,
        dense_units=config.lstm.dense_units,
        learning_rate=config.lstm.learning_rate,
    )

    classes = np.unique(splits.y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=splits.y_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    lstm_model.fit(
        X_train_lstm,
        splits.y_train,
        validation_data=(X_val_lstm, splits.y_val),
        epochs=config.lstm.epochs,
        batch_size=config.lstm.batch_size,
        class_weight=class_weight,
        verbose=1,
    )
    save_lstm_model(lstm_model, config.lstm.model_save_path)

    legitimate_probabilities = predict_probabilities(lstm_model, X_test_lstm, batch_size=config.lstm.batch_size)
    phishing_probabilities = 1.0 - legitimate_probabilities
    phishing_predictions = (phishing_probabilities >= 0.5).astype(np.int32)
    y_test_phishing = 1 - splits.y_test
    metrics = compute_metrics(y_test_phishing, phishing_predictions, phishing_probabilities)
    metrics.print_report("Test Set")

    metrics_path = config.wso.history_save_path.parent / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    print("Training complete.")
    print(f"Vocabulary: {config.encoding.vocab_save_path}")
    print(f"CNN feature extractor: {config.cnn.model_save_path}")
    print(f"WSO selected features: {config.wso.output_mask_path}")
    print(f"LSTM classifier: {config.lstm.model_save_path}")
    print(f"Test metrics: {metrics_path}")


if __name__ == "__main__":
    main()
