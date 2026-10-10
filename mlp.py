"""Multi-Layer Perceptron (MLP) module for WSO candidate feature evaluation.

Role:
- Serves as the fitness evaluator during White Shark Optimization (WSO).
- For each binary mask candidate, trains a lightweight MLP on selected training features
  and measures classification error on the validation partition.
- Computes fitness cost combining validation error and feature subset ratio (paper Eq. 25):
    Cost = alpha * Validation_Error + beta * (selected_features / total_features)
- Includes in-memory caching to avoid redundant evaluations of identical candidate subsets.
- Strictly never evaluates on the test partition.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers


def build_mlp_classifier(
    input_dim: int,
    hidden_units: Optional[List[int]] = None,
    dropout_rate: float = 0.1,
    activation: str = "relu",
    learning_rate: float = 0.001,
    random_seed: Optional[int] = 42,
) -> models.Model:
    """Builds a lightweight sequential MLP for evaluating candidate feature subsets.

    Args:
        input_dim: Number of selected features (k >= 1).
        hidden_units: List of hidden layer neuron counts (default: [32, 16]).
        dropout_rate: Dropout rate for regularization.
        activation: Hidden layer activation function.
        learning_rate: Learning rate for Adam optimizer.
        random_seed: Random seed for weight initialization reproducibility.

    Returns:
        Compiled Keras Model.
    """
    if input_dim < 1:
        raise ValueError(f"input_dim must be >= 1, got {input_dim}")
    if hidden_units is None:
        hidden_units = [32, 16]

    initializer = tf.keras.initializers.GlorotUniform(seed=random_seed)

    inputs = layers.Input(shape=(input_dim,), dtype="float32", name="selected_features")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units=units,
            activation=activation,
            kernel_initializer=initializer,
            name=f"mlp_dense_{i+1}",
        )(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=f"mlp_dropout_{i+1}")(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=initializer,
        name="mlp_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mlp_evaluator")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class MLPFitnessEvaluator:
    """Evaluates candidate feature masks on validation data with caching."""

    def __init__(
        self,
        X_train_features: np.ndarray,
        y_train: np.ndarray,
        X_val_features: np.ndarray,
        y_val: np.ndarray,
        hidden_units: Optional[List[int]] = None,
        alpha: float = 0.99,
        beta: float = 0.01,
        epochs: int = 3,
        batch_size: int = 64,
        patience: int = 2,
        learning_rate: float = 0.001,
        random_seed: int = 42,
    ):
        """Initializes the fitness evaluator.

        Args:
            X_train_features: Full feature matrix from CNN for training set (N_train, d).
            y_train: Ground truth training labels (N_train,).
            X_val_features: Full feature matrix from CNN for validation set (N_val, d).
            y_val: Ground truth validation labels (N_val,).
            hidden_units: Neuron counts for MLP hidden layers.
            alpha: Weight for classification error E in fitness cost (paper Eq. 25).
            beta: Weight for feature subset ratio in fitness cost.
            epochs: Training epochs per candidate subset.
            batch_size: Mini-batch size.
            patience: Early stopping patience.
            learning_rate: Adam optimizer learning rate.
            random_seed: Random seed for reproducible initialization.
        """
        self.X_train = np.asarray(X_train_features, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        self.X_val = np.asarray(X_val_features, dtype=np.float32)
        self.y_val = np.asarray(y_val, dtype=np.float32)

        self.total_features = self.X_train.shape[1]
        self.hidden_units = hidden_units or [32, 16]
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.learning_rate = float(learning_rate)
        self.random_seed = int(random_seed)

        # In-memory cache: tuple(mask) -> fitness cost
        self.cache: Dict[Tuple[int, ...], float] = {}
        self.evaluation_count = 0
        self.cache_hits = 0

    def evaluate(self, mask: Union[np.ndarray, Sequence[int]]) -> float:
        """Evaluates a binary feature mask and returns its fitness cost.

        Fitness Cost Formula (Paper Eq. 25):
            Cost = alpha * E + beta * (|mask| / d)
            where E is classification error (1.0 - val_accuracy).

        Empty masks (|mask| == 0) are assigned a penalty cost of 1.0.

        Args:
            mask: 1D array of 0s and 1s of length d.

        Returns:
            float fitness cost (lower is better).
        """
        mask_arr = np.asarray(mask, dtype=np.int32).flatten()
        if len(mask_arr) != self.total_features:
            raise ValueError(
                f"Mask length ({len(mask_arr)}) does not match feature dimension ({self.total_features})"
            )

        mask_key = tuple(mask_arr.tolist())
        if mask_key in self.cache:
            self.cache_hits += 1
            return self.cache[mask_key]

        selected_indices = np.where(mask_arr == 1)[0]
        k = len(selected_indices)

        # Safety: Empty feature subset handling
        if k == 0:
            penalty_cost = 1.0  # Maximum error penalty
            self.cache[mask_key] = penalty_cost
            return penalty_cost

        # Extract selected columns
        X_train_sub = self.X_train[:, selected_indices]
        X_val_sub = self.X_val[:, selected_indices]

        # Build and train lightweight MLP
        tf.random.set_seed(self.random_seed + self.evaluation_count)
        mlp = build_mlp_classifier(
            input_dim=k,
            hidden_units=self.hidden_units,
            dropout_rate=0.1,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
        )

        es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            verbose=0,
        )

        mlp.fit(
            X_train_sub,
            self.y_train,
            validation_data=(X_val_sub, self.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[es],
            verbose=0,
        )

        # Evaluate on validation partition ONLY
        val_loss, val_acc = mlp.evaluate(X_val_sub, self.y_val, verbose=0)
        classification_error = 1.0 - float(val_acc)
        feature_ratio = float(k) / float(self.total_features)

        # Cost function from paper Eq. 25
        cost = (self.alpha * classification_error) + (self.beta * feature_ratio)

        self.cache[mask_key] = float(cost)
        self.evaluation_count += 1
        return float(cost)
