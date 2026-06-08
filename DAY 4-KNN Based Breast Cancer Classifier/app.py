import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)


diagnosis_colors = {
    "Benign": "#4CAF50",
    "Malignant": "#F44336"
}

diagnosis_messages = {
    "Benign": "No signs of breast cancer detected. Regular monitoring is recommended.",
    "Malignant": "Potential signs of breast cancer detected. Consult a medical professional."
}


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = request.get_json()

        features = np.array([[
            float(data["mean_radius"]),
            float(data["mean_texture"]),
            float(data["mean_perimeter"]),
            float(data["mean_area"]),
            float(data["mean_smoothness"]),
            float(data["mean_compactness"]),
            float(data["mean_concavity"]),
            float(data["mean_concave_points"])
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        probabilities = model.predict_proba(features_scaled)[0]

        diagnosis = "Benign" if prediction == 1 else "Malignant"

        return jsonify({
            "diagnosis": diagnosis,
            "confidence": round(float(probabilities[prediction]) * 100, 1),
            "color": diagnosis_colors[diagnosis],
            "message": diagnosis_messages[diagnosis],
            "malignant_prob": round(float(probabilities[0]) * 100, 1),
            "benign_prob": round(float(probabilities[1]) * 100, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">

<title>Breast Cancer Predictor - Day 4</title>

<style>

body{
    background:#0e1117;
    color:white;
    font-family:Arial;
    padding:40px;
}

.container{
    max-width:800px;
    margin:auto;
}

.card{
    background:#161b22;
    padding:25px;
    border-radius:12px;
    margin-bottom:20px;
}

.field{
    margin-bottom:20px;
}

label{
    display:block;
    margin-bottom:8px;
}

input[type=range]{
    width:100%;
}

button{
    width:100%;
    padding:12px;
    border:none;
    border-radius:8px;
    background:#2196F3;
    color:white;
    cursor:pointer;
}

.result{
    display:none;
    margin-top:20px;
    background:#161b22;
    padding:25px;
    border-radius:12px;
}

.bar{
    background:#333;
    height:20px;
    border-radius:10px;
    margin-bottom:10px;
    overflow:hidden;
}

.fill{
    height:100%;
}

</style>
</head>

<body>

<div class="container">

<h1>Breast Cancer Predictor</h1>

<div class="card">

<div class="field">
<label>Mean Radius</label>
<input type="range" id="mean_radius" min="5" max="30" step="0.1" value="15">
</div>

<div class="field">
<label>Mean Texture</label>
<input type="range" id="mean_texture" min="10" max="40" step="0.1" value="20">
</div>

<div class="field">
<label>Mean Perimeter</label>
<input type="range" id="mean_perimeter" min="40" max="200" step="1" value="90">
</div>

<div class="field">
<label>Mean Area</label>
<input type="range" id="mean_area" min="100" max="2500" step="10" value="700">
</div>

<div class="field">
<label>Mean Smoothness</label>
<input type="range" id="mean_smoothness" min="0.05" max="0.2" step="0.001" value="0.1">
</div>

<div class="field">
<label>Mean Compactness</label>
<input type="range" id="mean_compactness" min="0" max="0.4" step="0.001" value="0.1">
</div>

<div class="field">
<label>Mean Concavity</label>
<input type="range" id="mean_concavity" min="0" max="0.5" step="0.001" value="0.1">
</div>

<div class="field">
<label>Mean Concave Points</label>
<input type="range" id="mean_concave_points" min="0" max="0.25" step="0.001" value="0.05">
</div>

<button onclick="predict()">Predict Diagnosis</button>

</div>

<div id="result" class="result">

<h2 id="diagnosis"></h2>

<p id="confidence"></p>

<p id="message"></p>

<h3>Probability Breakdown</h3>

<p>Benign</p>
<div class="bar">
<div id="benignBar" class="fill" style="background:#4CAF50;width:0%"></div>
</div>

<p>Malignant</p>
<div class="bar">
<div id="malignantBar" class="fill" style="background:#F44336;width:0%"></div>
</div>

</div>

</div>

<script>

async function predict(){

    const payload = {

        mean_radius:
        parseFloat(document.getElementById("mean_radius").value),

        mean_texture:
        parseFloat(document.getElementById("mean_texture").value),

        mean_perimeter:
        parseFloat(document.getElementById("mean_perimeter").value),

        mean_area:
        parseFloat(document.getElementById("mean_area").value),

        mean_smoothness:
        parseFloat(document.getElementById("mean_smoothness").value),

        mean_compactness:
        parseFloat(document.getElementById("mean_compactness").value),

        mean_concavity:
        parseFloat(document.getElementById("mean_concavity").value),

        mean_concave_points:
        parseFloat(document.getElementById("mean_concave_points").value)
    };

    const response = await fetch("/predict",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify(payload)
    });

    const data = await response.json();

    document.getElementById("result").style.display="block";

    document.getElementById("diagnosis").innerHTML =
        data.diagnosis;

    document.getElementById("diagnosis").style.color =
        data.color;

    document.getElementById("confidence").innerHTML =
        "Confidence: " + data.confidence + "%";

    document.getElementById("message").innerHTML =
        data.message;

    document.getElementById("benignBar").style.width =
        data.benign_prob + "%";

    document.getElementById("malignantBar").style.width =
        data.malignant_prob + "%";
}

</script>

</body>
</html>
"""


if __name__ == "__main__":
    print("Starting Breast Cancer Predictor...")
    print("Open http://localhost:5000")
    app.run(debug=True, port=5000)