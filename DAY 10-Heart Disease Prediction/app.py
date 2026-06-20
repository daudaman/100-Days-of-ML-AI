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
with open("best_model_name.pkl", "rb") as f:
    best_model_name = pickle.load(f)


@app.route("/")
def index():
    return render_template_string(HTML_PAGE, model_name=best_model_name)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([[
            float(data["age"]),
            float(data["resting_bp"]),
            float(data["cholesterol"]),
            float(data["max_heart_rate"]),
            float(data["oldpeak"]),
            int(data["chest_pain_type"]),
            int(data["exercise_angina"]),
            int(data["st_slope"])
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        risk_pct = round(float(probabilities[1]) * 100, 1)
        healthy_pct = round(float(probabilities[0]) * 100, 1)

        if risk_pct >= 65:
            color = "#F44336"
            level = "High Risk"
        elif risk_pct >= 40:
            color = "#FF9800"
            level = "Moderate Risk"
        else:
            color = "#4CAF50"
            level = "Low Risk"

        return jsonify({
            "prediction": "Disease Detected" if prediction == 1 else "No Disease Detected",
            "risk_level": level,
            "risk_pct": risk_pct,
            "healthy_pct": healthy_pct,
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
  <title>Heart Disease Predictor - Day 10</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #fdf2f2;
      --surface: #ffffff;
      --border: #fde0e0;
      --text: #1f2937;
      --muted: #6b7280;
      --accent: #e11d48;
    }
    body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; min-height: 100vh; padding: 48px 20px; }
    .wrapper { max-width: 720px; margin: 0 auto; }
    .top-row { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
    .tag { font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--accent); }
    .model-chip { font-size: 11px; background: #fff1f2; color: var(--accent); padding: 4px 10px; border-radius: 20px; border: 1px solid #fecdd3; font-weight: 500; }
    h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 8px; line-height: 1.15; }
    .sub { font-size: 14px; color: var(--muted); margin-bottom: 36px; line-height: 1.7; font-weight: 300; max-width: 540px; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 28px 32px; margin-bottom: 16px; }
    .sec-label { font-size: 10px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 22px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px 32px; }
    @media (max-width: 500px) { .grid { grid-template-columns: 1fr; } h1 { font-size: 1.7rem; } }
    .field label { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .field label span { font-size: 15px; font-weight: 500; color: var(--text); }
    input[type="range"] { -webkit-appearance: none; width: 100%; height: 2px; background: var(--border); border-radius: 2px; outline: none; cursor: pointer; }
    input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 15px; height: 15px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 3px rgba(225,29,72,0.15); transition: box-shadow 0.15s; }
    input[type="range"]::-webkit-slider-thumb:hover { box-shadow: 0 0 0 6px rgba(225,29,72,0.2); }
    .range-hints { display: flex; justify-content: space-between; font-size: 10px; color: #d1aaaa; margin-top: 4px; }
    select.sel { width: 100%; padding: 9px 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-family: 'Inter', sans-serif; font-size: 13px; outline: none; }
    .toggle-group { display: flex; gap: 6px; margin-top: 2px; }
    .tog { flex: 1; padding: 8px 0; border: 1px solid var(--border); border-radius: 8px; background: transparent; color: var(--muted); font-size: 12px; cursor: pointer; transition: all 0.15s; font-family: 'Inter', sans-serif; }
    .tog.on { background: var(--accent); border-color: var(--accent); color: white; font-weight: 500; }
    .btn { width: 100%; margin-top: 24px; padding: 13px; background: var(--accent); color: white; border: none; border-radius: 50px; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity 0.15s, transform 0.12s; }
    .btn:hover { opacity: 0.85; transform: translateY(-1px); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    .result { display: none; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 28px 32px; animation: up 0.35s ease both; }
    @keyframes up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .r-prediction { font-size: 1.7rem; font-weight: 700; margin-bottom: 6px; }
    .r-badge { display: inline-block; font-size: 11px; font-weight: 500; padding: 3px 10px; border-radius: 4px; margin-bottom: 20px; }
    .meter-label { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .meter-label span { font-size: 14px; font-weight: 500; color: var(--text); }
    .meter-track { width: 100%; height: 7px; background: #f3f4f6; border-radius: 7px; overflow: hidden; margin-bottom: 20px; }
    .meter-fill { height: 100%; border-radius: 7px; transition: width 0.7s cubic-bezier(0.22, 1, 0.36, 1); }
    .disclaimer { font-size: 11px; color: var(--muted); margin-top: 16px; padding: 10px 12px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; }
    footer { text-align: center; margin-top: 32px; font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
<div class="wrapper">
  <div class="top-row">
    <div class="tag">Day 10 of 100</div>
    <div class="model-chip">{{ model_name }}</div>
  </div>
  <h1>Heart Disease Predictor </h1>
  <p class="sub">7 algorithms were compared with cross-validation. {{ model_name }} won with the highest test accuracy.</p>

  <div class="card">
    <div class="sec-label">Patient Details</div>
    <div class="grid">

      <div class="field">
        <label>Age <span id="v-age">50</span></label>
        <input type="range" id="age" min="28" max="78" step="1" value="50">
        <div class="range-hints"><span>28</span><span>78</span></div>
      </div>

      <div class="field">
        <label>Resting BP (mm Hg) <span id="v-bp">130</span></label>
        <input type="range" id="resting_bp" min="94" max="178" step="1" value="130">
        <div class="range-hints"><span>94</span><span>178</span></div>
      </div>

      <div class="field">
        <label>Cholesterol (mg/dL) <span id="v-chol">220</span></label>
        <input type="range" id="cholesterol" min="126" max="410" step="2" value="220">
        <div class="range-hints"><span>126</span><span>410</span></div>
      </div>

      <div class="field">
        <label>Max Heart Rate <span id="v-hr">145</span></label>
        <input type="range" id="max_heart_rate" min="72" max="198" step="1" value="145">
        <div class="range-hints"><span>72</span><span>198</span></div>
      </div>

      <div class="field">
        <label>Oldpeak (ST depression) <span id="v-op">1.0</span></label>
        <input type="range" id="oldpeak" min="0" max="5.5" step="0.1" value="1.0">
        <div class="range-hints"><span>0</span><span>5.5</span></div>
      </div>

      <div class="field">
        <label style="margin-bottom:8px;">Chest Pain Type</label>
        <select class="sel" id="chest_pain_type">
          <option value="0">Typical Angina</option>
          <option value="1">Atypical Angina</option>
          <option value="2" selected>Non-Anginal Pain</option>
          <option value="3">Asymptomatic</option>
        </select>
      </div>

      <div class="field">
        <label style="margin-bottom:8px;">ST Slope</label>
        <select class="sel" id="st_slope">
          <option value="0">Upsloping</option>
          <option value="1" selected>Flat</option>
          <option value="2">Downsloping</option>
        </select>
      </div>

      <div class="field">
        <label style="margin-bottom:10px;">Exercise Induced Angina</label>
        <div class="toggle-group">
          <button class="tog on" id="ea-no" onclick="setTog(0)">No</button>
          <button class="tog" id="ea-yes" onclick="setTog(1)">Yes</button>
        </div>
      </div>

    </div>
    <button class="btn" id="pred-btn" onclick="runPredict()">Predict Risk</button>
  </div>

  <div class="result" id="result">
    <div class="r-prediction" id="r-pred">—</div>
    <div class="r-badge" id="r-badge">—</div>
    <div class="meter-label">Disease Probability <span id="r-pct">0%</span></div>
    <div class="meter-track"><div class="meter-fill" id="r-meter" style="width:0%"></div></div>
    <div class="disclaimer">This is a machine learning demo only. Not for medical use.</div>
  </div>


</div>

<script>
  let exerciseAngina = 0;

  function setTog(val) {
    exerciseAngina = val;
    document.getElementById("ea-no").className = "tog" + (val===0?" on":"");
    document.getElementById("ea-yes").className = "tog" + (val===1?" on":"");
  }

  const sliders = [
    { id: "age",            out: "v-age",  dp: 0 },
    { id: "resting_bp",     out: "v-bp",   dp: 0 },
    { id: "cholesterol",    out: "v-chol", dp: 0 },
    { id: "max_heart_rate", out: "v-hr",   dp: 0 },
    { id: "oldpeak",        out: "v-op",   dp: 1 }
  ];

  sliders.forEach(({ id, out, dp }) => {
    document.getElementById(id).addEventListener("input", e => {
      document.getElementById(out).textContent = parseFloat(e.target.value).toFixed(dp);
    });
  });

  async function runPredict() {
    const btn = document.getElementById("pred-btn");
    btn.disabled = true;
    btn.textContent = "Predicting...";

    const payload = {
      age: parseFloat(document.getElementById("age").value),
      resting_bp: parseFloat(document.getElementById("resting_bp").value),
      cholesterol: parseFloat(document.getElementById("cholesterol").value),
      max_heart_rate: parseFloat(document.getElementById("max_heart_rate").value),
      oldpeak: parseFloat(document.getElementById("oldpeak").value),
      chest_pain_type: parseInt(document.getElementById("chest_pain_type").value),
      exercise_angina: exerciseAngina,
      st_slope: parseInt(document.getElementById("st_slope").value)
    };

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

      const badge = document.getElementById("r-badge");
      badge.textContent = data.risk_level;
      badge.style.background = data.color + "22";
      badge.style.color = data.color;
      badge.style.border = "1px solid " + data.color + "44";

      document.getElementById("r-pct").textContent = data.risk_pct + "%";
      document.getElementById("r-meter").style.width = data.risk_pct + "%";
      document.getElementById("r-meter").style.background = data.color;

      const el = document.getElementById("result");
      el.style.display = "block";
      el.style.animation = "none";
      el.offsetHeight;
      el.style.animation = "";

    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Predict Risk";
    }
  }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"Starting Heart Disease Predictor  (best model: {best_model_name})...")
    print("Open http://localhost:5000")
    app.run(debug=True, port=5000)
