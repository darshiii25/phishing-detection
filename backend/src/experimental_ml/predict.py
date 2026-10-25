"""Inference adapter for the separately trained TF-IDF/Random Forest model."""
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "url_classifier.joblib"


def predict_url(url: str, threshold: float = 0.5) -> tuple[str, float, float]:
    value = url.strip() if isinstance(url, str) else ""
    if not value:
        raise ValueError("URL cannot be empty.")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Random Forest model not found: {MODEL_PATH}. Train it with python -m src.experimental_ml.train.")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "named_steps") or "classifier" not in model.named_steps:
        raise ValueError("Random Forest artifact has an incompatible pipeline format.")
    classifier = model.named_steps["classifier"]
    probs = model.predict_proba([value])[0]
    by_class = dict(zip(classifier.classes_, probs))
    if 0 not in by_class or 1 not in by_class:
        raise ValueError("Model must contain dataset labels 0 (phishing) and 1 (legitimate).")
    phishing, legitimate = float(by_class[0]), float(by_class[1])
    return ("Legitimate" if legitimate >= threshold else "Phishing", legitimate, phishing)


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        url = sys.argv[1].strip()
    else:
        url = input("Enter URL to check: ").strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    label, legitimate, phishing = predict_url(url)
    print(f"\nTarget URL:             {url}")
    print(f"Prediction:             {label}")
    print(f"Legitimate Probability: {legitimate * 100:.2f}%")
    print(f"Phishing Probability:   {phishing * 100:.2f}%")


if __name__ == "__main__":
    main()

