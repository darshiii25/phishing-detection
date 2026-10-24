"""Modular phishing URL detection package.

Components:
- preprocessing: Dataset loading, validation, sanitization, stratified splitting.
- encoding: Character-level vocabulary building, URL-to-tensor mapping.
- cnn: 1D Convolutional Neural Network feature extractor.
- mlp: Fast Multi-Layer Perceptron fitness evaluator.
- wso: Binary White Shark Optimization for feature selection.
- lstm: Bi-LSTM classifier on optimized feature subsets.
- evaluation: Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
"""

__all__ = [
    "preprocessing",
    "encoding",
    "cnn",
    "mlp",
    "wso",
    "lstm",
    "evaluation",
]
