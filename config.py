"""Configuration module for the modular phishing URL detection codebase.

Centralizes all dataset parameters, preprocessing rules, model architectures,
WSO optimization hyper-parameters, and file system paths to prevent hard-coding.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


@dataclass
class PreprocessingConfig:
    """Settings for dataset loading, cleaning, and train/val/test splits."""
    dataset_path: Path = DATA_DIR / "Dataset.csv"
    url_column: str = "url"
    label_column: str = "label"
    random_seed: int = 42
    # Stratified split proportions (Train / Validation / Test)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    # Cleaning policies
    drop_duplicates: bool = True
    strip_protocol: bool = False  # If True, removes http://, https://, www.
    lower_case: bool = False      # URL paths can be case-sensitive; keep False by default


@dataclass
class EncodingConfig:
    """Settings for character-level URL tokenization and sequence padding."""
    # 200 characters covers 99.91% of URLs in Dataset.csv without truncation
    max_url_len: int = 200
    vocab_save_path: Path = OUTPUTS_DIR / "vocab.json"
    padding_token: str = "<PAD>"
    unknown_token: str = "<UNK>"
    # Reserved IDs: 0 for PAD, 1 for UNK
    pad_id: int = 0
    unk_id: int = 1


@dataclass
class CNNConfig:
    """Settings for the character-level CNN feature extractor."""
    embedding_dim: int = 64
    conv_filters: List[int] = field(default_factory=lambda: [128, 128, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])
    pool_sizes: List[int] = field(default_factory=lambda: [3, 3, 2])
    feature_dense_dim: int = 64  # Output feature vector dimension d
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 5
    model_save_path: Path = MODELS_DIR / "cnn_feature_extractor.keras"


@dataclass
class MLPConfig:
    """Settings for the lightweight MLP used as WSO fitness evaluator."""
    hidden_units: List[int] = field(default_factory=lambda: [32, 16])
    dropout_rate: float = 0.1
    activation: str = "relu"
    learning_rate: float = 0.001
    epochs: int = 3           # Kept fast for rapid evaluation during WSO search
    batch_size: int = 64
    patience: int = 2          # Early stopping patience


@dataclass
class WSOConfig:
    """Settings for the binary White Shark Optimization (WSO) algorithm.

    Formulated in accordance with Albahadili et al. (2024), Eqs. 8-25.
    """
    population_size: int = 20    # Number of candidate white sharks
    max_iterations: int = 10     # Configurable search iterations (e.g., 10 for dev, 50-100 for prod)
    random_seed: int = 42
    # WSO kinematic hyperparameters (paper Eqs. 9-11, 17, 20)
    p_min: float = 0.5
    p_max: float = 1.5
    tau: float = 4.125          # Acceleration coefficient
    a0: float = 1.0
    a1: float = 1.0
    a2: float = 1.0
    # Fitness weights (Eq. 25: Cost = alpha * Error + beta * (selected_feats / d))
    alpha: float = 0.99         # Importance of classification error
    beta: float = 0.01          # Penalty for selecting excess features
    # Transfer function: 'v_shape' (arctan, Eq. 23) or 's_shape' (sigmoid, Eq. 22)
    transfer_function: str = "v_shape"
    output_mask_path: Path = OUTPUTS_DIR / "wso_selected_features.json"
    history_save_path: Path = OUTPUTS_DIR / "wso_history.json"


@dataclass
class LSTMConfig:
    """Settings for the LSTM / Bi-LSTM final classifier."""
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    bidirectional: bool = True   # Paper Fig. 6 uses Bi-LSTM
    dropout_rate: float = 0.2
    dense_units: int = 32
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 10
    # Adaptation: reshape 1D selected features into 2D sequence (timesteps, channels)
    # Options: 'steps_1' -> shape (k, 1), 'step_k' -> shape (1, k)
    sequence_mode: str = "steps_1"
    model_save_path: Path = MODELS_DIR / "lstm_classifier.keras"


@dataclass
class ProjectConfig:
    """Master configuration aggregating all submodule configs."""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    wso: WSOConfig = field(default_factory=WSOConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)


# Default singleton instance
DEFAULT_CONFIG = ProjectConfig()
