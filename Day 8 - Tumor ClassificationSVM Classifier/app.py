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


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([[
            float(data["radius"]),
            float(data["texture"]),
            float(data["perimeter"]),
            float(data["area"]),
            float(data["smoothness"]),
            float(data["compactness"]),
            float(data["concavity"]),
            float(data["symmetry"]),
            float(data["fractal_dimension"])
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        malignant_pct = round(float(probabilities[1]) * 100, 1)
        benign_pct = round(float(probabilities[0]) * 100, 1)

        if malignant_pct >= 65:
            color = "#F44336"
        elif malignant_pct >= 40:
            color = "#FF9800"
        else:
            color = "#4CAF50"

        return jsonify({
            "prediction": "Malignant" if prediction == 1 else "Benign",
            "malignant_pct": malignant_pct,
            "benign_pct": benign_pct,
            "color": color
        })

    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Tumor Classifier - Day 8</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #f0fdf4;
      --surface: #ffffff;
      --border: #d1fae5;
      --text: #111827;
      --muted: #6b7280;
      --accent: #059669;
    }
    body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; min-height: 100vh; padding: 48px 20px; }
    .wrapper { max-width: 700px; margin: 0 auto; }
    .tag { font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--accent); margin-bottom: 12px; }
    h1 { font-size: 2.2rem; font-weight: 600; margin-bottom: 8px; line-height: 1.15; }
    .sub { font-size: 14px; color: var(--muted); margin-bottom: 36px; line-height: 1.7; font-weight: 300; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 28px 32px; margin-bottom: 16px; }
    .sec-label { font-size: 10px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 22px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px 32px; }
    @media (max-width: 500px) { .grid { grid-template-columns: 1fr; } h1 { font-size: 1.7rem; } }
    .field label { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .field label span { font-size: 15px; font-weight: 500; color: var(--text); }
    input[type="range"] { -webkit-appearance: none; width: 100%; height: 2px; background: var(--border); border-radius: 2px; outline: none; cursor: pointer; }
    input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 15px; height: 15px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 3px rgba(5,150,105,0.15); transition: box-shadow 0.15s; }
    input[type="range"]::-webkit-slider-thumb:hover { box-shadow: 0 0 0 6px rgba(5,150,105,0.2); }
    .range-hints { display: flex; justify-content: space-between; font-size: 10px; color: #9ca3af; margin-top: 4px; }
    .btn { width: 100%; margin-top: 24px; padding: 13px; background: var(--accent); color: white; border: none; border-radius: 50px; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 500; cursor: pointer; transition: opacity 0.15s, transform 0.12s; }
    .btn:hover { opacity: 0.85; transform: translateY(-1px); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    .result { display: none; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 28px 32px; animation: up 0.35s ease both; }
    @keyframes up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .r-prediction { font-size: 2rem; font-weight: 600; margin-bottom: 6px; }
    .meter-label { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 8px; margin-top: 16px; }
    .meter-label span { font-size: 14px; font-weight: 500; color: var(--text); }
    .meter-track { width: 100%; height: 7px; background: #f3f4f6; border-radius: 7px; overflow: hidden; margin-bottom: 20px; }
    .meter-fill { height: 100%; border-radius: 7px; transition: width 0.7s cubic-bezier(0.22, 1, 0.36, 1); }
    .divider { border: none; border-top: 1px solid var(--border); margin: 4px 0 18px; }
    .prob-lbl { font-size: 10px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }
    .prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
    .prob-name { font-size: 12px; width: 80px; color: var(--muted); }
    .prob-track { flex: 1; height: 4px; background: #f3f4f6; border-radius: 4px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 4px; transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1); }
    .prob-pct { font-size: 12px; width: 36px; text-align: right; color: var(--muted); }
    .disclaimer { font-size: 11px; color: var(--muted); margin-top: 16px; padding: 10px 12px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; }
    footer { text-align: center; margin-top: 32px; font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
<div class="wrapper">
 
  <h1>Tumor Classifier</h1>
  <p class="sub">SVM with RBF kernel — 88% accuracy. Classifies tumors as benign or malignant based on 9 cell nucleus measurements.</p>

  <div class="card">
    <div class="sec-label">Cell Nucleus Measurements</div>
    <div class="grid">

      <div class="field">
        <label>Radius <span id="v-rad">14.0</span></label>
        <input type="range" id="radius" min="6" max="28" step="0.1" value="14.0">
        <div class="range-hints"><span>6</span><span>28</span></div>
      </div>

      <div class="field">
        <label>Texture <span id="v-tex">20.0</span></label>
        <input type="range" id="texture" min="9" max="40" step="0.1" value="20.0">
        <div class="range-hints"><span>9</span><span>40</span></div>
      </div>

      <div class="field">
        <label>Perimeter <span id="v-per">92.0</span></label>
        <input type="range" id="perimeter" min="40" max="190" step="1" value="92">
        <div class="range-hints"><span>40</span><span>190</span></div>
      </div>

      <div class="field">
        <label>Area <span id="v-area">650</span></label>
        <input type="range" id="area" min="100" max="2500" step="10" value="650">
        <div class="range-hints"><span>100</span><span>2500</span></div>
      </div>

      <div class="field">
        <label>Smoothness <span id="v-sm">0.100</span></label>
        <input type="range" id="smoothness" min="0.05" max="0.16" step="0.001" value="0.100">
        <div class="range-hints"><span>0.05</span><span>0.16</span></div>
      </div>

      <div class="field">
        <label>Compactness <span id="v-com">0.100</span></label>
        <input type="range" id="compactness" min="0.02" max="0.35" step="0.005" value="0.100">
        <div class="range-hints"><span>0.02</span><span>0.35</span></div>
      </div>

      <div class="field">
        <label>Concavity <span id="v-con">0.100</span></label>
        <input type="range" id="concavity" min="0.0" max="0.43" step="0.005" value="0.100">
        <div class="range-hints"><span>0.0</span><span>0.43</span></div>
      </div>

      <div class="field">
        <label>Symmetry <span id="v-sym">0.180</span></label>
        <input type="range" id="symmetry" min="0.1" max="0.3" step="0.005" value="0.180">
        <div class="range-hints"><span>0.1</span><span>0.3</span></div>
      </div>

      <div class="field">
        <label>Fractal Dimension <span id="v-fd">0.0700</span></label>
        <input type="range" id="fractal_dimension" min="0.05" max="0.1" step="0.001" value="0.070">
        <div class="range-hints"><span>0.05</span><span>0.1</span></div>
      </div>

    </div>
    <button class="btn" id="pred-btn" onclick="runPredict()">Classify Tumor</button>
  </div>

  <div class="result" id="result">
    <div class="r-prediction" id="r-pred">—</div>
    <div class="meter-label">Malignant Probability <span id="r-pct">0%</span></div>
    <div class="meter-track"><div class="meter-fill" id="r-meter" style="width:0%"></div></div>
    <hr class="divider">
    <div class="prob-lbl">Probability Breakdown</div>
    <div class="prob-row">
      <div class="prob-name">Benign</div>
      <div class="prob-track"><div class="prob-fill" id="b-ben" style="width:0%;background:#4CAF50"></div></div>
      <div class="prob-pct" id="p-ben">0%</div>
    </div>
    <div class="prob-row">
      <div class="prob-name">Malignant</div>
      <div class="prob-track"><div class="prob-fill" id="b-mal" style="width:0%;background:#F44336"></div></div>
      <div class="prob-pct" id="p-mal">0%</div>
    </div>
    
  </div>

  <footer>Day 8 of ML/AI · SVM · scikit-learn · Flask</footer>
</div>

<script>
  const sliders = [
    { id: "radius",             out: "v-rad",  dp: 1 },
    { id: "texture",            out: "v-tex",  dp: 1 },
    { id: "perimeter",          out: "v-per",  dp: 1 },
    { id: "area",               out: "v-area", dp: 0 },
    { id: "smoothness",         out: "v-sm",   dp: 3 },
    { id: "compactness",        out: "v-com",  dp: 3 },
    { id: "concavity",          out: "v-con",  dp: 3 },
    { id: "symmetry",           out: "v-sym",  dp: 3 },
    { id: "fractal_dimension",  out: "v-fd",   dp: 4 }
  ];

  sliders.forEach(({ id, out, dp }) => {
    document.getElementById(id).addEventListener("input", e => {
      document.getElementById(out).textContent = parseFloat(e.target.value).toFixed(dp);
    });
  });

  async function runPredict() {
    const btn = document.getElementById("pred-btn");
    btn.disabled = true;
    btn.textContent = "Classifying...";

    const payload = {};
    sliders.forEach(({ id }) => {
      payload[id] = parseFloat(document.getElementById(id).value);
    });

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      document.getElementById("r-pred").textContent = data.prediction;
      document.getElementById("r-pred").style.color = data.color;
      document.getElementById("r-pct").textContent = data.malignant_pct + "%";
      document.getElementById("r-meter").style.width = data.malignant_pct + "%";
      document.getElementById("r-meter").style.background = data.color;
      document.getElementById("b-ben").style.width = data.benign_pct + "%";
      document.getElementById("p-ben").textContent = data.benign_pct + "%";
      document.getElementById("b-mal").style.width = data.malignant_pct + "%";
      document.getElementById("p-mal").textContent = data.malignant_pct + "%";

      const el = document.getElementById("result");
      el.style.display = "block";
      el.style.animation = "none";
      el.offsetHeight;
      el.style.animation = "";

    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Classify Tumor";
    }
  }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting Tumor Classifier (SVM)...")
    print("Open http://localhost:5000")
    app.run(debug=True, port=5000)
