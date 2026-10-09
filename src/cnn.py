"""1D Convolutional Neural Network (CNN) Feature Extractor for Character-Encoded URLs.

Architecture Overview:
- Input: Integer sequence of shape (batch_size, max_url_len) representing URL characters.
- Embedding: Projects character IDs into dense continuous representations (batch_size, max_url_len, embedding_dim).
- Conv1D Stack: Extracts local character n-gram motifs (e.g. substrings, tlds, patterns).
- Pooling: Condenses temporal feature maps into a fixed-length representation.
- Dense Feature Bottleneck: Produces the primary d-dimensional feature vector (batch_size, d).
- Feature Extractor: Sub-model returning exclusively the d-dimensional feature vector for downstream WSO and LSTM.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_cnn_models(
    vocab_size: int,
    max_url_len: int = 200,
    embedding_dim: int = 64,
    conv_filters: Optional[List[int]] = None,
    kernel_sizes: Optional[List[int]] = None,
    pool_sizes: Optional[List[int]] = None,
    feature_dense_dim: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> Tuple[models.Model, models.Model]:
    """Constructs the full CNN classifier and the feature-extractor submodel.

    Args:
        vocab_size: Total number of tokens in character vocabulary (including PAD and UNK).
        max_url_len: Maximum length of character sequences (input length).
        embedding_dim: Dimensionality of character embedding space.
        conv_filters: Filter counts for each 1D convolution layer (default: [128, 128, 64]).
        kernel_sizes: Window size for each convolution layer (default: [7, 5, 3]).
        pool_sizes: Max-pooling pool size after each conv layer (default: [3, 3, 2]).
        feature_dense_dim: Dimension d of extracted feature vector (bottleneck representation).
        dropout_rate: Dropout probability for regularization.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Tuple of (classification_model, feature_extractor_model):
        - classification_model: Accepts (batch, max_url_len) -> outputs (batch, 1) phishing probability.
        - feature_extractor_model: Accepts (batch, max_url_len) -> outputs (batch, feature_dense_dim) feature vectors.
    """
    if conv_filters is None:
        conv_filters = [128, 128, 64]
    if kernel_sizes is None:
        kernel_sizes = [7, 5, 3]
    if pool_sizes is None:
        pool_sizes = [3, 3, 2]

    if not (len(conv_filters) == len(kernel_sizes) == len(pool_sizes)):
        raise ValueError(
            f"conv_filters ({len(conv_filters)}), kernel_sizes ({len(kernel_sizes)}), "
            f"and pool_sizes ({len(pool_sizes)}) must have identical lengths."
        )

    # 1. Input Layer: (batch_size, max_url_len)
    inputs = layers.Input(shape=(max_url_len,), dtype="int32", name="url_char_sequence")

    # 2. Embedding Layer: (batch_size, max_url_len, embedding_dim)
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="char_embedding",
    )(inputs)

    # 3. Stack of 1D Convolutional Layers with Pooling
    for i, (filters, k_size, p_size) in enumerate(zip(conv_filters, kernel_sizes, pool_sizes)):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=k_size,
            activation="relu",
            padding="same",
            name=f"conv1d_layer_{i+1}",
        )(x)
        if p_size > 1:
            x = layers.MaxPooling1D(pool_size=p_size, padding="same", name=f"maxpool_layer_{i+1}")(x)

    # 4. Global Max Pooling: aggregates highest activations across sequence length
    x = layers.GlobalMaxPooling1D(name="global_max_pooling")(x)

    # 5. Feature Bottleneck Layer: Dense layer of size d
    features = layers.Dense(
        feature_dense_dim,
        activation="relu",
        name="extracted_features",
    )(x)

    # 6. Feature Extractor Model: Input -> d-dimensional feature vector
    feature_extractor_model = models.Model(
        inputs=inputs,
        outputs=features,
        name="cnn_feature_extractor",
    )

    # 7. Classification Head: for supervised training of CNN features
    dropout = layers.Dropout(dropout_rate, name="classifier_dropout")(features)
    classifier_output = layers.Dense(
        1,
        activation="sigmoid",
        name="phishing_prediction",
    )(dropout)

    classification_model = models.Model(
        inputs=inputs,
        outputs=classifier_output,
        name="cnn_classifier",
    )
    classification_model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return classification_model, feature_extractor_model


def extract_features(
    feature_extractor: models.Model,
    X_encoded: np.ndarray,
    batch_size: int = 128,
    verbose: int = 0,
) -> np.ndarray:
    """Extracts dense d-dimensional feature vectors from character-encoded URLs.

    Args:
        feature_extractor: Keras Model producing feature bottleneck vectors.
        X_encoded: Integer matrix of shape (N, max_url_len).
        batch_size: Batch size for prediction.
        verbose: Keras predict verbosity (0 = silent).

    Returns:
        np.ndarray of shape (N, d) with float32 feature representations.
    """
    if X_encoded.ndim != 2:
        raise ValueError(f"X_encoded must be 2D array of shape (N, max_url_len), got shape {X_encoded.shape}")

    features = feature_extractor.predict(X_encoded, batch_size=batch_size, verbose=verbose)
    return np.asarray(features, dtype=np.float32)


def save_cnn_feature_extractor(model: models.Model, save_path: Union[str, Path]) -> None:
    """Saves the Keras feature extractor model to disk."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_cnn_feature_extractor(load_path: Union[str, Path]) -> models.Model:
    """Loads a saved Keras feature extractor model from disk."""
    path = Path(load_path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model file not found at: {path}")
    return models.load_model(str(path))
