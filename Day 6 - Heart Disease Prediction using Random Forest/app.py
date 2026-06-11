"""
app.py — Flask Web App for Heart Disease Prediction
Day 6 Project
"""

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE  = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE, "model.pkl"), "rb"))
feat  = pickle.load(open(os.path.join(BASE, "feature_names.pkl"), "rb"))


@app.route("/")
def home():
    return render_template("index.html", features=feat)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json()
        values = [float(data[f]) for f in feat]
        arr    = np.array(values).reshape(1, -1)
        pred   = model.predict(arr)[0]
        proba  = model.predict_proba(arr)[0]

        result = {
            "prediction": int(pred),
            "label": "Heart Disease Detected" if pred == 1 else "No Heart Disease",
            "confidence": round(float(max(proba)) * 100, 1),
            "risk_score": round(float(proba[1]) * 100, 1),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
