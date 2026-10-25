
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "url_classifier.joblib"

# Labels in your dataset:
# 0 = Phishing
# 1 = Legitimate

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_url_classifier.py first."
    )

model = joblib.load(MODEL_PATH)


def check_url(url: str) -> None:
    url = url.strip()

    if not url:
        print("Please enter a URL.")
        return

    # The model analyzes the URL text; it does not visit the website.
    prediction = model.predict([url])[0]
    probabilities = model.predict_proba([url])[0]

    # Get probabilities by class label, not by array position.
    classes = list(model.named_steps["classifier"].classes_)
    probability_by_class = dict(zip(classes, probabilities))

    phishing_probability = probability_by_class.get(0, 0.0)
    legitimate_probability = probability_by_class.get(1, 0.0)

    print("\n" + "=" * 55)
    print(f"URL: {url}")
    print("-" * 55)

    if prediction == 0:
        print("Prediction: PHISHING")
    elif prediction == 1:
        print("Prediction: LEGITIMATE")
    else:
        print(f"Prediction: Unknown label ({prediction})")

    print(
        f"Phishing score:    {phishing_probability * 100:.2f}%"
    )
    print(
        f"Legitimate score:  {legitimate_probability * 100:.2f}%"
    )
    print("=" * 55)

    print(
        "Note: These are model scores, not verified probabilities "
        "that the website is safe or malicious."
    )


if __name__ == "__main__":
    print("URL Phishing Detection")
    print("Enter 'exit' to quit.")

    while True:
        user_url = input("\nEnter URL: ").strip()

        if user_url.lower() == "exit":
            print("Exiting URL checker.")
            break

        check_url(user_url)
