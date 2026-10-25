# Character-Level Phishing URL Detection

A Python research codebase for binary URL classification. The implemented pipeline combines character-level URL encoding, a one-dimensional CNN feature extractor, binary White Shark Optimization (WSO) for feature selection, an MLP fitness evaluator, and a Bi-LSTM classifier.

The design is inspired by *Detection of phishing URLs with deep learning based on GAN-CNN-LSTM network and swarm intelligence algorithms* by Abbas Jabr Saleh Albahadili, Ayhan Akbas, and Javad Rahebi (2024). This repository is a modular adaptation, not a claim of exact reproduction of that paper.

## 1. Scopegit 

- Uses URL text and binary labels; it does not fetch URLs, resolve DNS, or inspect page content.
- Omits the paper's GAN component and does not generate synthetic samples.
- Preserves the cleaned dataset's natural class distribution.
- Trains a CNN on the URL labels, uses WSO to select from its 64-dimensional learned representation, and evaluates candidate feature subsets with an MLP.
- Trains a Bi-LSTM on the selected CNN feature values. In the default `steps_1` input mode, the selected feature indices are treated as sequence steps; they are not the original URL characters or time steps.
- The repository contains training and prediction entry points, model artifacts, output JSON files, and unit tests. Existing output files are artifacts from particular runs, not guaranteed results of every run.

## 2. Pipeline and Data Flow

```text
src/Dataset.csv
    |
    v
Load, sanitize, and stratified split
    |  configured run: 80% train / 10% validation / 10% test
    v
Fit character vocabulary on training URLs only
    |
    v
Encode each URL as 200 integer character IDs
    |
    v
Train character embedding + Conv1D CNN classifier
    |
    v
Extract one 64-value vector per URL
    |
    +---- train and validation vectors ---> WSO + MLP fitness evaluation
    |                                        selects k of 64 features
    v
Select the same k columns from each split
    |
    v
Reshape (N, k) to (N, k, 1)
    |
    v
Train Bi-LSTM classifier and evaluate on the held-out test partition
```

The WSO evaluator trains an MLP per distinct candidate mask using the training features and scores it on the validation features. The test partition is not passed to the fitness evaluator. The feature extractor does produce test features so the final selected columns can be evaluated after feature selection.

### Split configuration

The end-to-end script reads split ratios from `DEFAULT_CONFIG`: **80% train, 10% validation, and 10% test (8:1:1)**, with random seed `42`. Splitting is stratified. This is the setting used by `src/train_pipeline.py` unless the configuration is changed. The standalone `preprocess_pipeline()` function uses the same 80/10/10 defaults.

## 3. Dataset

The configured dataset path is `src/Dataset.csv`; the expected columns are `url` and `label`.

| Property | Observed in the current CSV |
| :--- | :--- |
| Raw rows | 100,873 |
| Rows with missing URL | 0 |
| Rows with missing label | 1 |
| Rows after current cleaning | 100,872 |
| Duplicate URLs removed | 0 |
| Label `0` (phishing) | 63,492 (62.94%) |
| Label `1` (legitimate) | 37,380 (37.06%) |
| Shortest URL | 7 characters |
| Median URL length | 31 characters |
| 90th / 95th / 99th percentile length | 56 / 63 / 87 characters |
| 99.9th percentile / maximum length | approximately 196 / 474 characters |

The label convention used by this project is **`1 = legitimate` and `0 = phishing`**. Preprocessing drops rows missing either required value, trims surrounding URL whitespace, rejects non-binary labels, and removes duplicate URL strings when `drop_duplicates=True`. Protocol stripping and lowercasing are disabled by default; URL strings are otherwise kept case-sensitive and unchanged. The configured sequence limit of 200 characters covers approximately 99.9% of this CSV's URL lengths; longer strings are truncated from the right.

These counts describe the CSV currently in the repository. Re-audit the dataset if it is replaced or modified.

## 4. Technical Design and Dimensions

Let `N` denote the number of examples in a partition, `d = 64` the CNN feature dimension, and `k` the number of features selected by WSO (`1 <= k <= d`).

### 4.1 Preprocessing and character encoding

1. Load `url` and `label`, sanitize rows, and create stratified train, validation, and test partitions.
2. Fit `UrlCharacterEncoder` on training URLs only. The saved vocabulary uses IDs `0` for `<PAD>` and `1` for `<UNK>`; characters have IDs starting at `2`.
3. Encode each URL to a fixed-length `int32` vector of length 200. Short URLs are right-padded; long URLs are truncated to their first 200 characters.
4. The vocabulary is data-derived at training time. The current saved vocabulary contains 86 observed characters plus the two reserved tokens (88 entries total); a different training split can produce a different vocabulary.

### 4.2 CNN feature extractor and classifier

The CNN uses `embedding_dim=64`, Conv1D filters `[128, 128, 64]`, kernel sizes `[7, 5, 3]`, max-pool sizes `[3, 3, 2]`, and a 64-unit dense feature bottleneck. Convolutions use ReLU and `padding="same"`; pooling uses `padding="same"`. The classifier head uses dropout `0.2` and a single sigmoid output.

| Layer / operation | Output shape | Default details |
| :--- | :--- | :--- |
| Integer URL input | `(N, 200)` | `int32` |
| Character embedding | `(N, 200, 64)` | 64-dimensional embedding |
| Conv1D + max pool, block 1 | `(N, 67, 128)` | 128 filters, kernel 7, pool 3 |
| Conv1D + max pool, block 2 | `(N, 23, 128)` | 128 filters, kernel 5, pool 3 |
| Conv1D + max pool, block 3 | `(N, 12, 64)` | 64 filters, kernel 3, pool 2 |
| Global max pooling | `(N, 64)` | Reduces across remaining sequence positions |
| Dense feature bottleneck | `(N, 64)` | ReLU; `d = 64` features |
| Dropout + binary output | `(N, 1)` | Dropout 0.2; sigmoid |

The model builder returns both a compiled CNN classifier and a feature-extractor model that ends at the 64-value bottleneck. The classifier uses Adam (`learning_rate=0.001`) and binary cross-entropy. The default configuration specifies batch size 64 and 15 CNN training epochs.

### 4.3 WSO feature selection and MLP fitness

Each WSO candidate is a binary mask of length `d`; `1` keeps a feature and `0` discards it. Continuous positions and velocities are updated during the search, then converted to binary masks with the configured transfer function (`v_shape` by default). The implementation guarantees at least one feature is selected.

The MLP candidate evaluator uses hidden layers `[32, 16]`, ReLU activations, dropout `0.1`, Adam (`0.001`), and binary cross-entropy. Its default evaluation budget is 3 epochs per uncached candidate, batch size 64, and early-stopping patience 2. Identical masks are cached in memory.

The fitness objective implemented is:

```text
cost = alpha * (1 - validation_accuracy) + beta * (k / d)
```

Defaults are `alpha=0.99` and `beta=0.01`. The configured WSO run uses population size 30, 25 iterations, random seed 42, `tau=4.125`, and `v_shape`. Lower fitness is preferred. These are experiment settings, not architecture-independent performance guarantees.

The current `outputs/wso_selected_features.json` records a particular run selecting 2 of 64 features (indices 6 and 27). That is an observed artifact, not a fixed property of the algorithm or a promise that a fresh run will select the same features.

### 4.4 Bi-LSTM classifier

The selected CNN feature matrix `(N, k)` is converted by the default `steps_1` adapter into `(N, k, 1)`. Thus, each selected feature is treated as one LSTM timestep with one channel. The alternative `step_k` mode produces `(N, 1, k)`, treating the vector as one timestep with `k` channels.

The default classifier has two bidirectional LSTM layers with 64 and 32 units per direction. The first returns sequences; the second returns its final representation. With `steps_1`, the principal dimensions are:

| Layer / operation | Output shape | Default details |
| :--- | :--- | :--- |
| Selected features | `(N, k)` | `k` is determined by WSO |
| `steps_1` adaptation | `(N, k, 1)` | `float32` |
| First Bi-LSTM | `(N, k, 128)` | 64 units in each direction; sequences returned |
| Second Bi-LSTM | `(N, 64)` | 32 units in each direction |
| Dense projection + dropout | `(N, 32)` | ReLU; dropout 0.2 |
| Sigmoid output | `(N, 1)` | Binary probability output |

The LSTM uses Adam (`learning_rate=0.001`) and binary cross-entropy; configuration defaults are batch size 64 and 20 epochs. The training pipeline calculates balanced class weights for LSTM training. No class weighting is applied by the pipeline to CNN or MLP training.

### 4.5 Shape contract

| Stage | Input | Output | Type |
| :--- | :--- | :--- | :--- |
| Cleaned partition | CSV rows | `N` URL strings and `N` labels | strings / `int32` |
| Character encoding | `N` URLs | `(N, 200)` | `int32` |
| Embedding | `(N, 200)` | `(N, 200, 64)` | floating point |
| CNN features | `(N, 200)` | `(N, 64)` | `float32` |
| WSO-selected features | `(N, 64)` | `(N, k)` | `float32` |
| LSTM input, `steps_1` | `(N, k)` | `(N, k, 1)` | `float32` |
| Keras classifier output | model-specific input | `(N, 1)` | sigmoid probability |
| Prediction helper output | `(N, 1)` | `(N,)` | `float32` probability or `int32` class |

## 5. Label Handling and Evaluation

The dataset convention is `1 = legitimate` and `0 = phishing`. The CNN and LSTM are trained directly on these dataset labels, so their sigmoid output represents the probability of class `1` (legitimate). `src/predict_url.py` reports this value as the legitimate probability and computes phishing probability as `1 - legitimate_probability`.

For security metrics, `src/train_pipeline.py` converts the held-out labels and predictions to a phishing-positive representation (`phishing = 1 - dataset_label`) before calling the generic evaluator. Thus, reported precision, recall, F1, ROC-AUC, and confusion-matrix TP/FN refer to phishing URLs. The current `outputs/test_metrics.json` is a saved result from a particular run; regenerate it after changing the split or training setup before reporting results.

## 6. Repository Layout

```text
phishing-detection/
├── config.py
├── requirements.txt
├── README.md
├── src/
│   ├── Dataset.csv
│   ├── cnn.py
│   ├── encoding.py
│   ├── evaluation.py
│   ├── lstm.py
│   ├── mlp.py
│   ├── predict_url.py
│   ├── preprocessing.py
│   ├── train_pipeline.py
│   └── wso.py
├── models/
│   ├── cnn_feature_extractor.keras
│   └── lstm_classifier.keras
├── outputs/
│   ├── test_metrics.json
│   ├── vocab.json
│   ├── wso_history.json
│   └── wso_selected_features.json
└── tests/
    ├── test_encoding.py
    ├── test_lstm_eval.py
    ├── test_preprocessing.py
    └── test_wso.py
```

`config.py` defines data paths and model/training defaults. `src/train_pipeline.py` runs the integrated training and evaluation flow. `src/predict_url.py` loads the saved vocabulary, CNN feature extractor, selected feature indices, and LSTM model to classify one entered URL. `outputs/` contains JSON artifacts; trained Keras files are stored in `models/`.

## 7. Setup and Usage

### Environment

Use a Python version supported by the TensorFlow release available for your operating system.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dependencies declared in `requirements.txt`:

- NumPy `>=1.26.0`
- pandas `>=2.0.0`
- TensorFlow `>=2.15.0`
- scikit-learn `>=1.3.0`
- matplotlib `>=3.8.0`
- pytest `>=7.4.0`

Run from the repository root:

```powershell
# Train models, select features, and write evaluation/output artifacts
python -m src.train_pipeline

# Predict one URL using the saved artifacts
python -m src.predict_url

# Run unit tests
python -m pytest tests/ -v
```

Training requires the dataset at `src/Dataset.csv`. It writes the character vocabulary and WSO results under `outputs/`, and saves the CNN feature extractor and LSTM classifier under `models/`. Run training before using the prediction entry point if those artifacts are missing or out of date.

## 8. Outputs and Evaluation

The pipeline writes:

- `outputs/vocab.json`: encoder settings and character-to-ID mapping.
- `outputs/wso_selected_features.json`: selected mask, selected indices, and fitness summary.
- `outputs/wso_history.json`: best and mean fitness information per iteration.
- `outputs/test_metrics.json`: saved metric values from the latest pipeline run.
- `models/cnn_feature_extractor.keras` and `models/lstm_classifier.keras`: serialized Keras models.

The evaluator reports accuracy, precision, recall/sensitivity, F1-score, ROC-AUC, and the confusion matrix. Interpret these according to the label-orientation warning above. Unit tests check implementation behavior and tensor contracts; they do not by themselves establish generalization or real-world phishing detection performance.

## 9. Limitations and Reporting Notes

## 10. Experimental Random Forest adapter

An isolated char-TF-IDF + Random Forest option is available without changing the CNN/WSO/Bi-LSTM predictor. Start the existing API with `PHISHING_ACTIVE_MODEL=random_forest` to select it, or omit the variable / set it to `deep` for the original pipeline. The API response shape is unchanged. The adapter loads `models/url_classifier.joblib` and reports dataset class `0` as phishing and `1` as legitimate. Train a fresh local artifact and write validation plus held-out test metrics with `python -m src.experimental_ml.train` from `backend`; this uses the current `data/Dataset.csv`, removes exact duplicate URLs, and keeps vectorizer fitting inside the training pipeline. The uploaded artifact was copied into `models/` as a starting artifact; its original evaluation split and exact data provenance have not been independently verified, so regenerate it before presenting metrics.

- A random stratified split from one CSV does not establish performance on future URLs, different sources, or a temporally independent dataset.
- URL strings longer than 200 characters are truncated. The vocabulary maps unseen characters to `<UNK>`.
- The LSTM receives selected CNN latent features, not raw character embeddings. In `steps_1` mode, it treats the retained feature order as a sequence; this is an architectural adaptation and should be described as such.
- WSO fitness is validation accuracy combined with a feature-count penalty. Search settings and results can vary with the data, seed, and configuration.
- The configured experimental split is 80/10/10 (8:1:1), stratified by the dataset label. State the split and label mapping (`1 = legitimate`, `0 = phishing`) explicitly when reporting results.
