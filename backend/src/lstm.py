"""LSTM / Bi-LSTM Classifier module for Selected CNN Feature Representations.

Architectural Adaptation & Sequence Representation:
The output of White Shark Optimization (WSO) is a subset of k selected continuous
features extracted by the CNN. Because LSTM layers in TensorFlow/Keras require 3D
tensors of shape (batch_size, timesteps, feature_dim), the 2D feature matrix (batch_size, k)
is transformed through a documented sequence formatting strategy:

1. 'steps_1' (Default):
   Reshapes (batch_size, k) -> (batch_size, k, 1).
   Each selected feature represents an ordered step in the sequence, allowing the
   Bidirectional LSTM cells to model progressive contextual dependencies across the
   optimized feature space.

2. 'step_k':
   Reshapes (batch_size, k) -> (batch_size, 1, k).
   Treats the entire selected feature vector as a single time step with k channels.

Model Architecture:
- Input Layer: (batch_size, timesteps, feature_dim)
- Recurrent Layer: Configurable Bidirectional LSTM or standard LSTM
- Dense Bottleneck Layer: Dense with ReLU and Dropout regularization
- Classification Output: Single neuron with Sigmoid activation for binary phishing probability
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def prepare_lstm_input(
    features: np.ndarray,
    sequence_mode: str = "steps_1",
) -> np.ndarray:
    """Adapts a 2D feature matrix into a 3D tensor suitable for LSTM input.

    Args:
        features: 2D array of shape (N, k) containing selected features.
        sequence_mode: Formatting strategy:
            - 'steps_1': Reshapes to (N, k, 1). Each feature is treated as a timestep.
            - 'step_k': Reshapes to (N, 1, k). Whole feature vector is 1 timestep with k channels.

    Returns:
        3D np.ndarray of shape (N, timesteps, feature_dim) with dtype float32.

    Raises:
        ValueError: If features is not 2D or sequence_mode is unrecognized.
    """
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (N, k), got shape {arr.shape}")

    n_samples, k = arr.shape

    if sequence_mode == "steps_1":
        # Shape: (N, k, 1)
        return arr.reshape((n_samples, k, 1))
    elif sequence_mode == "step_k":
        # Shape: (N, 1, k)
        return arr.reshape((n_samples, 1, k))
    else:
        raise ValueError(
            f"Unknown sequence_mode '{sequence_mode}'. Supported modes: 'steps_1', 'step_k'."
        )


def build_lstm_classifier(
    input_shape: Tuple[int, int],
    lstm_units: Optional[List[int]] = None,
    bidirectional: bool = True,
    dropout_rate: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 0.001,
) -> models.Model:
    """Builds and compiles a configurable LSTM / Bi-LSTM binary classification model.

    In accordance with Albahadili et al. (2024) Section 3.5, Bidirectional LSTM
    gathers forward and backward temporal/feature context to classify URLs.

    Args:
        input_shape: Tuple of (timesteps, feature_dim) - e.g. (k, 1) for 'steps_1'.
        lstm_units: List of unit counts for recurrent layers (default: [64, 32]).
        bidirectional: If True, wraps LSTM in keras.layers.Bidirectional (paper Fig. 6).
        dropout_rate: Dropout probability for recurrent and dense layers.
        dense_units: Units in post-LSTM dense projection layer.
        learning_rate: Learning rate for Adam optimizer.

    Returns:
        Compiled Keras Model.
    """
    if lstm_units is None:
        lstm_units = [64, 32]

    inputs = layers.Input(shape=input_shape, dtype="float32", name="lstm_input")
    x = inputs

    num_layers = len(lstm_units)
    for i, units in enumerate(lstm_units):
        is_last = (i == num_layers - 1)
        lstm_layer = layers.LSTM(
            units=units,
            return_sequences=not is_last,
            dropout=dropout_rate,
            name=f"lstm_layer_{i+1}",
        )
        if bidirectional:
            x = layers.Bidirectional(lstm_layer, name=f"bilstm_layer_{i+1}")(x)
        else:
            x = lstm_layer(x)

    # Dense classification head
    if dense_units > 0:
        x = layers.Dense(dense_units, activation="relu", name="dense_projection")(x)
        x = layers.Dropout(dropout_rate, name="dense_dropout")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="phishing_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="bilstm_classifier")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_probabilities(
    model: models.Model,
    X_3d: np.ndarray,
    batch_size: int = 128,
    verbose: int = 0,
) -> np.ndarray:
    """Predicts continuous phishing probabilities (range [0.0, 1.0]).

    Args:
        model: Trained LSTM model.
        X_3d: 3D NumPy array of shape (N, timesteps, feature_dim).
        batch_size: Batch size for inference.
        verbose: Keras verbosity.

    Returns:
        1D np.ndarray of probabilities of length N (dtype float32).
    """
    probs = model.predict(X_3d, batch_size=batch_size, verbose=verbose)
    return np.asarray(probs, dtype=np.float32).flatten()


def predict_classes(
    model: models.Model,
    X_3d: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 128,
) -> np.ndarray:
    """Predicts discrete binary labels: 1 = Phishing, 0 = Legitimate.

    Args:
        model: Trained LSTM model.
        X_3d: 3D NumPy array.
        threshold: Decision threshold (default 0.5).
        batch_size: Inference batch size.

    Returns:
        1D np.ndarray of integers {0, 1} of length N.
    """
    probs = predict_probabilities(model, X_3d, batch_size=batch_size)
    return (probs >= threshold).astype(np.int32)


def save_lstm_model(model: models.Model, file_path: Union[str, Path]) -> None:
    """Serializes the trained LSTM model to disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_lstm_model(file_path: Union[str, Path]) -> models.Model:
    """Deserializes a saved LSTM model from disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return models.load_model(str(path))
