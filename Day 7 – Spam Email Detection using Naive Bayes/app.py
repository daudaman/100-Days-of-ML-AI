import os
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, render_template, request, jsonify

# ── NLTK setup ────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
stemmer    = PorterStemmer()

app = Flask(__name__)

# ── Load artefacts ────────────────────────────────────────────────────────────
def _load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found – run `python train.py` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

model      = _load("model.pkl")
vectorizer = _load("vectorizer.pkl")


# ── Text cleaning (must mirror train.py) ─────────────────────────────────────
def clean_text(text: str) -> str:
    text   = str(text).lower()
    text   = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    cleaned = clean_text(message)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]

    return jsonify({
        "prediction":   "Spam"     if pred == 1 else "Not Spam",
        "is_spam":      bool(pred),
        "spam_prob":    round(float(proba[1]) * 100, 2),
        "ham_prob":     round(float(proba[0]) * 100, 2),
        "confidence":   round(float(max(proba)) * 100, 2),
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  Spam Detector – Flask App")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
