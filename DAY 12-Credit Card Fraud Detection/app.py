"""
Day 12 - Credit Card Fraud Detection
Flask app: accepts transaction features and returns fraud prediction.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load saved artifacts
model        = joblib.load("model.pkl")
scaler       = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")

@app.route("/")
def index():
    return render_template("index.html", feature_cols=feature_cols)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Build feature vector in correct order
        features = []
        for col in feature_cols:
            val = data.get(col, 0.0)
            features.append(float(val))

        X = np.array(features).reshape(1, -1)

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        result = {
            "prediction": int(prediction),
            "label": "Fraud Detected" if prediction == 1 else "Legitimate Transaction",
            "fraud_probability": round(float(probability) * 100, 2),
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
