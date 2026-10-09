# Phishing URL Detection: Modular Research Codebase

A modular machine-learning codebase for character-level phishing URL detection using Convolutional Neural Networks (CNN), binary White Shark Optimization (WSO) with Multi-Layer Perceptron (MLP) subset evaluation, and Long Short-Term Memory (Bi-LSTM) sequence classification.

---

## 1. Project Overview & Research Context

This project implements a modified, modular architecture inspired by the research paper:
> **"Detection of phishing URLs with deep learning based on GAN-CNN-LSTM network and swarm intelligence algorithms"**  
> *Abbas Jabr Saleh Albahadili, Ayhan Akbas, Javad Rahebi (Signal, Image and Video Processing, Springer, 2024)*

### Architectural Adaptation & Project Scope
- **GAN is excluded:** In accordance with project requirements, Generative Adversarial Networks (GANs) are omitted.
- **Natural Class Distribution Preserved:** Synthetic oversampling (e.g., SMOTE or GAN-based generation) is not applied. The natural dataset imbalance (~63% legitimate, ~37% phishing) is preserved throughout preprocessing, and stratified splitting maintains identical class proportions across Train, Validation, and Test partitions.
- **MLP Fitness Evaluator is included:** An MLP classifier is integrated specifically to evaluate candidate CNN feature subsets during WSO meta-heuristic optimization (paper Eq. 25).
- **Bi-LSTM Classifier is included:** A Bidirectional LSTM network performs final sequence-based classification on the feature subset selected by WSO.
- **Strict Role Boundaries:**
  - **Darshi (This Codebase):** Preprocessing, character-level encoding, CNN feature extraction, WSO feature selection, MLP fitness evaluation, Bi-LSTM classifier, evaluation metrics, configuration, unit tests, and documentation.
  - **Aditya (Teammate 1):** Model training experiments, learning rate scheduling, class weighting, hyperparameter tuning, and comparative baseline benchmarks.
  - **Teammate 2:** User interface (UI) and demonstration dashboard.

---

## 2. End-to-End Pipeline Architecture

```
Raw URL Dataset (data/Dataset.csv)
  │
  ▼
[1] Preprocessing (src/preprocessing.py)
  │  - Schema validation ('url', 'label')
  │  - Non-destructive sanitization (drop nulls, trim whitespace, deduplicate)
  │  - Stratified Train (80%) / Val (10%) / Test (10%) splitting
  │
  ▼
[2] Character-Level Encoding (src/encoding.py)
  │  - Vocabulary mapped exclusively on training partition
  │  - Reserved tokens: <PAD>=0, <UNK>=1; 86 ASCII dataset characters >= 2
  │  - Sequence length: max_url_len=200 -> Matrix shape (N, 200) int32
  │
  ▼
[3] CNN Feature Extraction (src/cnn.py)
  │  - Embedding (200, embedding_dim) -> Conv1D Stack -> GlobalMaxPooling1D
  │  - Feature Bottleneck Layer -> Dense(d) -> Feature matrix (N, d) float32
  │
  ▼
[4] WSO Feature Selection (src/wso.py)
  │  - Swarm Intelligence: Continuous velocity and position updates (Eqs. 8-21)
  │  - V-shape / S-shape transfer function binarization (Eqs. 22-24)
  │  - Candidate feature mask (d,) in {0, 1}
  │
  ├──► [4b] MLP Fitness Evaluator (src/mlp.py)
  │      - Evaluates candidate mask on Validation partition ONLY
  │      - Cost = alpha * Val_Error + beta * (selected_feats / d) [Eq. 25]
  │      - In-memory hashing cache eliminates redundant MLP training
  │
  ▼ Optimal Feature Subset: k features selected (k <= d)
[5] LSTM Sequence Classification (src/lstm.py)
  │  - Sequence Adaptation: Reshapes (N, k) -> (N, k, 1) ('steps_1')
  │  - Bidirectional LSTM with recurrent dropout and dense sigmoid head
  │
  ▼
[6] Performance Evaluation (src/evaluation.py)
     - Accuracy, Precision, Recall/Sensitivity, F1-Score, ROC-AUC
     - Confusion Matrix: TP, FP, TN, FN
```

---

## 3. Project Directory Structure

```
phishing_detection/
├── data/
│   └── Dataset.csv              # 100,873-row URL dataset
├── models/
│   ├── .gitkeep
│   ├── cnn_feature_extractor.keras  # Saved CNN weights (after training)
│   └── lstm_classifier.keras        # Saved Bi-LSTM weights (after training)
├── outputs/
│   ├── .gitkeep
│   ├── vocab.json               # Serialized character vocabulary
│   ├── wso_selected_features.json # Selected binary mask & feature indices
│   └── wso_history.json         # Iteration fitness progression log
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # CSV loading, validation, sanitization, stratified splits
│   ├── encoding.py              # Character-to-integer tokenizer and batch encoder
│   ├── cnn.py                   # 1D-CNN feature extractor and submodel interfaces
│   ├── mlp.py                   # Lightweight MLP candidate fitness evaluator with caching
│   ├── wso.py                   # Binary White Shark Optimizer (paper Eqs. 8-25)
│   ├── lstm.py                  # Bi-LSTM sequence classifier and input adapter
│   └── evaluation.py            # Classification metrics and confusion matrix plotter
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py    # 11 unit tests for preprocessing & splits
│   ├── test_encoding.py         # 9 unit tests for vocabulary & encoding
│   ├── test_wso.py              # 9 unit tests for CNN shapes, MLP caching, WSO search
│   └── test_lstm_eval.py        # 8 unit tests for LSTM 3D inputs & evaluation metrics
├── config.py                    # Central typed configuration dataclasses
├── requirements.txt             # Verified package dependencies
└── README.md                    # System documentation and developer handoff
```

---

## 4. Dependencies & Environment Setup

### Prerequisites
- Python 3.10+ (tested and verified on Python 3.14 64-bit on Windows)
- Windows PowerShell

### Verified Dependencies (`requirements.txt`)
- `numpy>=1.26.0`
- `pandas>=2.0.0`
- `tensorflow>=2.15.0`
- `scikit-learn>=1.3.0`
- `matplotlib>=3.8.0`
- `pytest>=7.4.0`

### Installation Steps (Windows PowerShell)

```powershell
# Navigate into the project folder
cd phishing_detection

# (Optional) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. Dataset Specification

- **File Path:** `data/Dataset.csv`
- **Schema:**
  - `url` (string): URL string (e.g., `http://keraekken-loagginnusa.godaddysites.com/`)
  - `label` (integer/float): Target label
    - `0` = Legitimate URL (negative class)
    - `1` = Phishing URL (positive class)
- **Empirical Dataset Audit:**
  - **Total Raw Rows:** 100,873
  - **Missing Values:** `url` has 0 missing; `label` had 1 missing row (`row 64772`), safely dropped during sanitization.
  - **Duplicates:** 0 identical duplicate URL rows.
  - **Cleaned Rows:** 100,872
  - **Class Distribution:**
    - Legitimate (`0`): 63,492 URLs (62.94%)
    - Phishing (`1`): 37,380 URLs (37.06%)
  - **URL Length Percentiles:**
    - Min: 7 | 50% (Median): 31 | 75%: 35 | 90%: 56 | 95%: 63 | 99%: 87 | 99.9%: 196 | Max: 474 characters.
    - Default `max_url_len = 200` covers **99.91%** of all URLs without truncation.

---

## 6. Module Interfaces & Individual Usage Guide

Each module in `src/` is independently importable. The code examples below reflect the exact implemented APIs:

### A. Preprocessing Module (`src/preprocessing.py`)
```python
from config import DEFAULT_CONFIG
from src.preprocessing import preprocess_pipeline

# Loads Dataset.csv, sanitizes rows, and performs stratified 80/10/10 split
splits = preprocess_pipeline(
    file_path=DEFAULT_CONFIG.preprocessing.dataset_path,
    train_ratio=0.80,
    val_ratio=0.10,
    test_ratio=0.10,
    random_seed=42,
    verbose=True  # Prints class distribution table across all splits
)

# Access splits (pd.Series of URLs and np.ndarray of binary int32 labels)
X_train, y_train = splits.X_train, splits.y_train
X_val, y_val     = splits.X_val, splits.y_val
X_test, y_test   = splits.X_test, splits.y_test
```

### B. Character-Level Encoding (`src/encoding.py`)
```python
from src.encoding import UrlCharacterEncoder

# Initialize encoder with max URL length
encoder = UrlCharacterEncoder(max_url_len=200)

# Fit vocabulary EXCLUSIVELY on training data to prevent data leakage
encoder.fit(X_train)

# Transform URL strings into 2D integer matrices of shape (N, 200)
X_train_encoded = encoder.encode_batch(X_train)
X_val_encoded   = encoder.encode_batch(X_val)
X_test_encoded  = encoder.encode_batch(X_test)

# Save vocabulary for inference
encoder.save_vocabulary("outputs/vocab.json")

# Later reload:
# encoder = UrlCharacterEncoder.load_vocabulary("outputs/vocab.json")
```

### C. CNN Feature Extraction (`src/cnn.py`)
```python
from src.cnn import build_cnn_models, extract_features

# Build classification model and intermediate feature extractor
cnn_classifier, cnn_feature_extractor = build_cnn_models(
    vocab_size=encoder.vocab_size,
    max_url_len=200,
    embedding_dim=64,
    conv_filters=[128, 128, 64],
    kernel_sizes=[7, 5, 3],
    pool_sizes=[3, 3, 2],
    feature_dense_dim=64,   # Feature dimension d
    dropout_rate=0.2,
    learning_rate=0.001
)

# Extract d-dimensional feature representations: output shape (N, 64)
train_features = extract_features(cnn_feature_extractor, X_train_encoded, batch_size=128)
val_features   = extract_features(cnn_feature_extractor, X_val_encoded, batch_size=128)
test_features  = extract_features(cnn_feature_extractor, X_test_encoded, batch_size=128)
```

### D. WSO Feature Selection & MLP Evaluator (`src/mlp.py` & `src/wso.py`)
```python
from src.mlp import MLPFitnessEvaluator
from src.wso import WhiteSharkOptimizer

# 1. Instantiate MLP Fitness Evaluator (evaluates candidates on validation data)
evaluator = MLPFitnessEvaluator(
    X_train_features=train_features,
    y_train=y_train,
    X_val_features=val_features,
    y_val=y_val,
    hidden_units=[32, 16],
    alpha=0.99,   # Weight on classification error
    beta=0.01,    # Weight on feature subset ratio
    epochs=3,     # Fast evaluation epochs
    batch_size=64,
    patience=2
)

# 2. Instantiate and run Binary White Shark Optimizer
wso = WhiteSharkOptimizer(
    num_features=train_features.shape[1],  # d = 64
    population_size=20,                    # 20 candidate sharks
    max_iterations=10,                     # Search iterations (e.g. 10 dev, 50-100 prod)
    tau=4.125,                             # Acceleration coefficient
    transfer_function="v_shape",           # Arctan transfer function
    random_seed=42
)

# Run optimization
wso_result = wso.optimize(fitness_func=evaluator.evaluate, verbose=True)

# Inspect selected subset
print("Selected feature count:", wso_result.num_selected)
print("Selected feature indices:", wso_result.selected_indices)
wso_result.save("outputs/wso_selected_features.json")

# Filter feature matrices using selected indices
train_selected = train_features[:, wso_result.selected_indices]
val_selected   = val_features[:, wso_result.selected_indices]
test_selected  = test_features[:, wso_result.selected_indices]
```

### E. Bi-LSTM Classification (`src/lstm.py`)
```python
from src.lstm import prepare_lstm_input, build_lstm_classifier, predict_classes, predict_probabilities

# 1. Reshape 2D features (N, k) into 3D sequence tensor (N, k, 1)
X_train_lstm = prepare_lstm_input(train_selected, sequence_mode="steps_1")
X_val_lstm   = prepare_lstm_input(val_selected, sequence_mode="steps_1")
X_test_lstm  = prepare_lstm_input(test_selected, sequence_mode="steps_1")

# 2. Construct Bi-LSTM Classifier
k = wso_result.num_selected
lstm_model = build_lstm_classifier(
    input_shape=(k, 1),
    lstm_units=[64, 32],
    bidirectional=True,
    dropout_rate=0.2,
    dense_units=32,
    learning_rate=0.001
)

# (Teammate Aditya trains lstm_model here using fit(...))

# 3. Predict on Test Set
test_probabilities = predict_probabilities(lstm_model, X_test_lstm)
test_predictions   = predict_classes(lstm_model, X_test_lstm, threshold=0.5)
```

### F. Performance Evaluation (`src/evaluation.py`)
```python
from src.evaluation import compute_metrics, plot_confusion_matrix

# Calculate all metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
metrics_result = compute_metrics(
    y_true=y_test,
    y_pred=test_predictions,
    y_probs=test_probabilities
)

# Print comprehensive formatted report
metrics_result.print_report(dataset_name="Test Set")

# Plot and save confusion matrix figure
plot_confusion_matrix(
    cm=metrics_result.confusion_matrix,
    save_path="outputs/confusion_matrix.png",
    title="Bi-LSTM Phishing URL Confusion Matrix"
)
```

---

## 7. Tensor & Array Shape Contract

| Processing Stage | Input Shape | Output Shape | Data Type | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Preprocessing** | CSV file | `pd.Series` of strings | `object` / `str` | Offline loading & splitting |
| **URL Encoding** | List of $N$ URLs | `(N, 200)` | `int32` | Right-padded with `<PAD>=0` |
| **CNN Embedding** | `(N, 200)` | `(N, 200, 64)` | `float32` | Continuous character representations |
| **CNN Features** | `(N, 200)` | `(N, 64)` | `float32` | $d=64$ bottleneck feature vector |
| **WSO Selection** | `(N, 64)` | `(N, k)` | `float32` | $k \le 64$ features retained |
| **LSTM Reshaping** | `(N, k)` | `(N, k, 1)` | `float32` | Sequence adaptation mode `'steps_1'` |
| **Bi-LSTM Output** | `(N, k, 1)` | `(N,)` | `float32` / `int32` | Continuous $[0, 1]$ or binary $\{0, 1\}$ |

---

## 8. Unit Testing & Verification

The test suite uses synthetic data fixtures to test each module thoroughly without requiring long GPU runs or downloading full checkpoints.

### Run All Test Suites (37 Unit Tests)
From the `phishing_detection` project root:
```powershell
python -m pytest tests/ -v
```

### Run Subsystem Tests Individually
```powershell
# Preprocessing tests (11 tests: loading, cleaning, stratification, reproducibility)
python -m pytest tests/test_preprocessing.py -v

# Character encoding tests (9 tests: padding, truncation, UNK, vocabulary consistency)
python -m pytest tests/test_encoding.py -v

# CNN, MLP, and WSO tests (9 tests: tensor shapes, caching, binarization, convergence)
python -m pytest tests/test_wso.py -v

# LSTM and evaluation tests (8 tests: 3D reshaping, Bi-LSTM forward, metrics, CM plot)
python -m pytest tests/test_lstm_eval.py -v
```

---

## 9. Developer Handoff Guide for Teammate Aditya (Model Training)

Aditya is responsible for model training experiments, hyperparameter tuning, and benchmark comparisons. Here is how to integrate these modules into experimental training scripts:

1. **Step 1: Data Partitioning**
   Import [preprocess_pipeline](file:///c:/Users/Darshi/study/clg/sem/sem%205/IS/code/phishing_detection/src/preprocessing.py#L372) from `src.preprocessing`. Pass `train_ratio=0.8, val_ratio=0.1, test_ratio=0.1`. Do not apply artificial balancing; preserve the natural class ratio.
2. **Step 2: URL Tokenizer Fitting**
   Import [UrlCharacterEncoder](file:///c:/Users/Darshi/study/clg/sem/sem%205/IS/code/phishing_detection/src/encoding.py#L31) from `src.encoding`. Call `encoder.fit(splits.X_train)`. Never fit on validation or test data. Save vocabulary to `outputs/vocab.json`.
3. **Step 3: CNN Pretraining & Feature Extraction**
   Import [build_cnn_models](file:///c:/Users/Darshi/study/clg/sem/sem%205/IS/code/phishing_detection/src/cnn.py#L20) and [extract_features](file:///c:/Users/Darshi/study/clg/sem/sem%205/IS/code/phishing_detection/src/cnn.py#L112) from `src.cnn`. You can train `cnn_classifier` on `X_train_encoded` with callbacks (e.g., `EarlyStopping`, `ReduceLROnPlateau`), save the weights, and then extract features from all partitions using `cnn_feature_extractor`.
4. **Step 4: WSO Feature Optimization**
   Import [MLPFitnessEvaluator](file:///c:/Users/Darshi/study/clg/sem/sem%205/IS/code/phishing_detection/src/mlp.py#L65) from `src.mlp` and [WhiteSharkOptimizer](file:///c:/Users/Darshi/study/clg/sem/sem%205/IS/code/phishing_detection/src/wso.py#L50) from `src.wso`. Instantiate the optimizer and run `.optimize(evaluator.evaluate)`. Tune `population_size` (e.g., 20) and `max_iterations` (e.g., 25–50).
5. **Step 5: Bi-LSTM Final Classifier Training**
   Import [prepare_lstm_input](file:///c:/Users/Darshi/study/clg/sem/sem%205/IS/code/phishing_detection/src/lstm.py#L32) and [build_lstm_classifier](file:///c:/Users/Darshi/study/clg/sem/sem%205/IS/code/phishing_detection/src/lstm.py#L64) from `src.lstm`. Format features using `steps_1` mode. Train using class weights (e.g., via `sklearn.utils.class_weight.compute_class_weight`) to account for the ~63/37 class distribution if desired.
6. **Step 6: Final Test Evaluation**
   Import [compute_metrics](file:///c:/Users/Darshi/study/clg/sem/sem%205/IS/code/phishing_detection/src/evaluation.py#L82) and [plot_confusion_matrix](file:///c:/Users/Darshi/study/clg/sem/sem%205/IS/code/phishing_detection/src/evaluation.py#L137) from `src.evaluation`. Run inference on the untouched `X_test_lstm` and generate final metrics and confusion matrix figures.

---

## 10. Limitations & Assumptions

1. **Synthetic Unit Tests vs. Real Generalization:** The 37 unit tests verify programmatic correctness, data structures, tensor dimensions, caching, and convergence on synthetic batches. They do not claim or establish real-world classification accuracy. Full accuracy metrics will be established during Aditya's training experiments.
2. **Fixed Sequence Length Assumption:** `max_url_len = 200` was chosen based on empirical percentile analysis (covering 99.91% of URLs in `Dataset.csv`). URLs exceeding 200 characters are truncated.
3. **Fitness Cost Trade-off:** The fitness formula uses $\alpha = 0.99$ and $\beta = 0.01$. This strongly prioritizes minimizing classification error while applying a mild regularization penalty on the number of selected features.
4. **Offline Operation:** All preprocessing and encoding runs strictly offline. The codebase never issues DNS lookups or HTTP requests to URLs.
