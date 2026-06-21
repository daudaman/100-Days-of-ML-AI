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
        tenure = float(data["tenure_months"])
        monthly = float(data["monthly_charges"])
        total = tenure * monthly

        features = np.array([[
            tenure,
            monthly,
            total,
            int(data["contract_type"]),
            int(data["support_calls"]),
            float(data["satisfaction_score"]),
            int(data["has_addon_services"])
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        churn_pct = round(float(probabilities[1]) * 100, 1)
        stay_pct = round(float(probabilities[0]) * 100, 1)

        if churn_pct >= 60:
            color = "#F44336"
            level = "High Churn Risk"
        elif churn_pct >= 30:
            color = "#FF9800"
            level = "Moderate Risk"
        else:
            color = "#4CAF50"
            level = "Low Risk"

        return jsonify({
            "prediction": "Likely to Churn" if prediction == 1 else "Likely to Stay",
            "risk_level": level,
            "churn_pct": churn_pct,
            "stay_pct": stay_pct,
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
  <title>Customer Churn Predictor - Day 11</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #f5f7fa;
      --surface: #ffffff;
      --border: #e3e8ef;
      --text: #1a202c;
      --muted: #718096;
      --accent: #6d28d9;
    }
    body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; min-height: 100vh; padding: 48px 20px; }
    .wrapper { max-width: 700px; margin: 0 auto; }
    .tag { font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--accent); margin-bottom: 12px; }
    h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 8px; line-height: 1.15; }
    .sub { font-size: 14px; color: var(--muted); margin-bottom: 36px; line-height: 1.7; font-weight: 300; max-width: 520px; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 28px 32px; margin-bottom: 16px; }
    .sec-label { font-size: 10px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 22px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px 32px; }
    @media (max-width: 500px) { .grid { grid-template-columns: 1fr; } h1 { font-size: 1.7rem; } }
    .field label { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .field label span { font-size: 15px; font-weight: 500; color: var(--text); }
    input[type="range"] { -webkit-appearance: none; width: 100%; height: 2px; background: var(--border); border-radius: 2px; outline: none; cursor: pointer; }
    input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 15px; height: 15px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 3px rgba(109,40,217,0.15); transition: box-shadow 0.15s; }
    input[type="range"]::-webkit-slider-thumb:hover { box-shadow: 0 0 0 6px rgba(109,40,217,0.2); }
    .range-hints { display: flex; justify-content: space-between; font-size: 10px; color: #cbd5e0; margin-top: 4px; }
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
    .meter-track { width: 100%; height: 7px; background: #f3f4f6; border-radius: 7px; overflow: hidden; }
    .meter-fill { height: 100%; border-radius: 7px; transition: width 0.7s cubic-bezier(0.22, 1, 0.36, 1); }
    footer { text-align: center; margin-top: 32px; font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
<div class="wrapper">
  <div class="tag">Day 11 of 100</div>
  <h1>Customer Churn Predictor</h1>
  <p class="sub">Random Forest trained on 2,500 customers with a 22% churn rate. Handles imbalanced classes using class_weight balancing.</p>

  <div class="card">
    <div class="sec-label">Customer Details</div>
    <div class="grid">

      <div class="field">
        <label>Tenure (months) <span id="v-ten">24</span></label>
        <input type="range" id="tenure_months" min="1" max="72" step="1" value="24">
        <div class="range-hints"><span>1</span><span>72</span></div>
      </div>

      <div class="field">
        <label>Monthly Charges ($) <span id="v-mc">65</span></label>
        <input type="range" id="monthly_charges" min="20" max="120" step="1" value="65">
        <div class="range-hints"><span>20</span><span>120</span></div>
      </div>

      <div class="field">
        <label>Support Calls <span id="v-sc">2</span></label>
        <input type="range" id="support_calls" min="0" max="10" step="1" value="2">
        <div class="range-hints"><span>0</span><span>10</span></div>
      </div>

      <div class="field">
        <label>Satisfaction (1-10) <span id="v-sat">6.0</span></label>
        <input type="range" id="satisfaction_score" min="1" max="10" step="0.5" value="6.0">
        <div class="range-hints"><span>1</span><span>10</span></div>
      </div>

      <div class="field">
        <label style="margin-bottom:8px;">Contract Type</label>
        <select class="sel" id="contract_type">
          <option value="0">Month-to-month</option>
          <option value="1" selected>One year</option>
          <option value="2">Two year</option>
        </select>
      </div>

      <div class="field">
        <label style="margin-bottom:10px;">Add-on Services</label>
        <div class="toggle-group">
          <button class="tog on" id="addon-no" onclick="setTog(0)">No</button>
          <button class="tog" id="addon-yes" onclick="setTog(1)">Yes</button>
        </div>
      </div>

    </div>
    <button class="btn" id="pred-btn" onclick="runPredict()">Predict Churn</button>
  </div>

  <div class="result" id="result">
    <div class="r-prediction" id="r-pred">—</div>
    <div class="r-badge" id="r-badge">—</div>
    <div class="meter-label">Churn Probability <span id="r-pct">0%</span></div>
    <div class="meter-track"><div class="meter-fill" id="r-meter" style="width:0%"></div></div>
  </div>

  
</div>

<script>
  let addon = 0;

  function setTog(val) {
    addon = val;
    document.getElementById("addon-no").className = "tog" + (val===0?" on":"");
    document.getElementById("addon-yes").className = "tog" + (val===1?" on":"");
  }

  const sliders = [
    { id: "tenure_months",       out: "v-ten", dp: 0 },
    { id: "monthly_charges",     out: "v-mc",  dp: 0 },
    { id: "support_calls",       out: "v-sc",  dp: 0 },
    { id: "satisfaction_score",  out: "v-sat", dp: 1 }
  ];

  sliders.forEach(({ id, out, dp }) => {
    document.getElementById(id).addEventListener("input", e => {
      document.getElementById(out).textContent = parseFloat(e.target.value).toFixed(dp);
    });
  });

  async function runPredict() {
    const btn = document.getElementById("pred-btn");
    btn.disabled = true;
    btn.textContent = "Predicting";

    const payload = {
      tenure_months: parseFloat(document.getElementById("tenure_months").value),
      monthly_charges: parseFloat(document.getElementById("monthly_charges").value),
      contract_type: parseInt(document.getElementById("contract_type").value),
      support_calls: parseInt(document.getElementById("support_calls").value),
      satisfaction_score: parseFloat(document.getElementById("satisfaction_score").value),
      has_addon_services: addon
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

      document.getElementById("r-pct").textContent = data.churn_pct + "%";
      document.getElementById("r-meter").style.width = data.churn_pct + "%";
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
      btn.textContent = "Predict Churn";
    }
  }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting Customer Churn Predictor")
    print("Open http://localhost:5000")
    app.run(debug=True, port=5000)
