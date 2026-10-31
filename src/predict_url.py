"""Predict whether one URL is legitimate or phishing using trained artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG
from src.cnn import extract_features, load_cnn_feature_extractor
from src.encoding import UrlCharacterEncoder
from src.lstm import load_lstm_model, predict_probabilities, prepare_lstm_input


def load_selected_indices(mask_path: Path) -> np.ndarray:
    if not mask_path.is_file():
        raise FileNotFoundError(f"WSO selected feature file not found: {mask_path}")

    with open(mask_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return np.asarray(data["selected_indices"], dtype=np.int32)


def predict_url(url: str, threshold: float = 0.5) -> tuple[str, float, float]:
    config = DEFAULT_CONFIG

    encoder = UrlCharacterEncoder.load_vocabulary(config.encoding.vocab_save_path)
    cnn_feature_extractor = load_cnn_feature_extractor(config.cnn.model_save_path)
    lstm_model = load_lstm_model(config.lstm.model_save_path)
    selected_indices = load_selected_indices(config.wso.output_mask_path)

    encoded = encoder.encode_batch([url])
    features = extract_features(cnn_feature_extractor, encoded, batch_size=1)
    selected_features = features[:, selected_indices]
    lstm_input = prepare_lstm_input(selected_features, sequence_mode=config.lstm.sequence_mode)
    legitimate_probability = float(predict_probabilities(lstm_model, lstm_input, batch_size=1)[0])
    phishing_probability = 1.0 - legitimate_probability

    label = "Legitimate" if legitimate_probability >= threshold else "Phishing"
    return label, legitimate_probability, phishing_probability


def main() -> None:
    url = input("Enter URL: ").strip()
    if not url:
        raise ValueError("URL cannot be empty.")

    label, legitimate_probability, phishing_probability = predict_url(url)
    print(f"Result: {label}")
    print(f"Legitimate probability: {legitimate_probability:.4f}")
    print(f"Phishing probability: {phishing_probability:.4f}")


if __name__ == "__main__":
    main()
