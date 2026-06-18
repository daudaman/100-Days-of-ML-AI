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
            float(data["income"]),
            float(data["loan_amount"]),
            float(data["credit_score"]),
            float(data["employment_years"]),
            float(data["existing_debt"]),
            int(data["dependents"]),
            int(data["loan_term_months"])
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        approval_pct = round(float(probabilities[1]) * 100, 1)
        rejection_pct = round(float(probabilities[0]) * 100, 1)

        color = "#4CAF50" if prediction == 1 else "#F44336"

        return jsonify({
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "approval_pct": approval_pct,
            "rejection_pct": rejection_pct,
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
  <title>Loan Approval Predictor - Day 9</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0a0e1a;
      --surface: #131826;
      --border: #232b40;
      --text: #eef1f8;
      --muted: #6b7590;
      --accent: #4f9eff;
    }
    body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; min-height: 100vh; padding: 48px 20px; }
    .wrapper { max-width: 700px; margin: 0 auto; }
    .tag { font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--accent); margin-bottom: 12px; }
    h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 8px; line-height: 1.15; }
    .sub { font-size: 14px; color: var(--muted); margin-bottom: 36px; line-height: 1.7; font-weight: 300; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 28px 32px; margin-bottom: 16px; }
    .sec-label { font-size: 10px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 22px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px 32px; }
    @media (max-width: 500px) { .grid { grid-template-columns: 1fr; } h1 { font-size: 1.7rem; } }
    .field label { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .field label span { font-size: 15px; font-weight: 500; color: var(--text); }
    input[type="range"] { -webkit-appearance: none; width: 100%; height: 2px; background: var(--border); border-radius: 2px; outline: none; cursor: pointer; }
    input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 15px; height: 15px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 3px rgba(79,158,255,0.2); transition: box-shadow 0.15s; }
    input[type="range"]::-webkit-slider-thumb:hover { box-shadow: 0 0 0 6px rgba(79,158,255,0.2); }
    .range-hints { display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); margin-top: 4px; }
    select.sel { width: 100%; padding: 9px 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-family: 'Inter', sans-serif; font-size: 13px; outline: none; }
    .btn { width: 100%; margin-top: 24px; padding: 13px; background: var(--accent); color: #0a0e1a; border: none; border-radius: 50px; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity 0.15s, transform 0.12s; }
    .btn:hover { opacity: 0.85; transform: translateY(-1px); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    .result { display: none; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 28px 32px; animation: up 0.35s ease both; }
    @keyframes up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .r-prediction { font-size: 2rem; font-weight: 700; margin-bottom: 14px; }
    .meter-label { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .meter-label span { font-size: 14px; font-weight: 500; color: var(--text); }
    .meter-track { width: 100%; height: 7px; background: var(--border); border-radius: 7px; overflow: hidden; margin-bottom: 6px; }
    .meter-fill { height: 100%; border-radius: 7px; transition: width 0.7s cubic-bezier(0.22, 1, 0.36, 1); }
    footer { text-align: center; margin-top: 32px; font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
<div class="wrapper">
  <div class="tag">Day 9 of 100</div>
  <h1>Loan Approval Predictor</h1>
  <p class="sub">Logistic Regression — 97.25% accuracy, 0.998 ROC AUC. Predicts loan approval from financial details.</p>

  <div class="card">
    <div class="sec-label">Applicant Details</div>
    <div class="grid">

      <div class="field">
        <label>Monthly Income ($) <span id="v-inc">8000</span></label>
        <input type="range" id="income" min="2000" max="25000" step="100" value="8000">
        <div class="range-hints"><span>2000</span><span>25000</span></div>
      </div>

      <div class="field">
        <label>Loan Amount ($) <span id="v-loan">300000</span></label>
        <input type="range" id="loan_amount" min="50000" max="800000" step="5000" value="300000">
        <div class="range-hints"><span>50k</span><span>800k</span></div>
      </div>

      <div class="field">
        <label>Credit Score <span id="v-cs">650</span></label>
        <input type="range" id="credit_score" min="300" max="850" step="1" value="650">
        <div class="range-hints"><span>300</span><span>850</span></div>
      </div>

      <div class="field">
        <label>Employment (years) <span id="v-emp">5.0</span></label>
        <input type="range" id="employment_years" min="0" max="30" step="0.5" value="5.0">
        <div class="range-hints"><span>0</span><span>30</span></div>
      </div>

      <div class="field">
        <label>Existing Debt ($) <span id="v-debt">10000</span></label>
        <input type="range" id="existing_debt" min="0" max="50000" step="500" value="10000">
        <div class="range-hints"><span>0</span><span>50k</span></div>
      </div>

      <div class="field">
        <label>Dependents <span id="v-dep">1</span></label>
        <input type="range" id="dependents" min="0" max="4" step="1" value="1">
        <div class="range-hints"><span>0</span><span>4</span></div>
      </div>

      <div class="field">
        <label style="margin-bottom:8px;">Loan Term</label>
        <select class="sel" id="loan_term_months">
          <option value="12">12 months</option>
          <option value="24">24 months</option>
          <option value="36" selected>36 months</option>
          <option value="48">48 months</option>
          <option value="60">60 months</option>
        </select>
      </div>

    </div>
    <button class="btn" id="pred-btn" onclick="runPredict()">Check Approval</button>
  </div>

  <div class="result" id="result">
    <div class="r-prediction" id="r-pred">—</div>
    <div class="meter-label">Approval Probability <span id="r-pct">0%</span></div>
    <div class="meter-track"><div class="meter-fill" id="r-meter" style="width:0%"></div></div>
  </div>


</div>

<script>
  const sliders = [
    { id: "income",            out: "v-inc",  dp: 0 },
    { id: "loan_amount",       out: "v-loan", dp: 0 },
    { id: "credit_score",      out: "v-cs",   dp: 0 },
    { id: "employment_years",  out: "v-emp",  dp: 1 },
    { id: "existing_debt",     out: "v-debt", dp: 0 },
    { id: "dependents",        out: "v-dep",  dp: 0 }
  ];

  sliders.forEach(({ id, out, dp }) => {
    document.getElementById(id).addEventListener("input", e => {
      document.getElementById(out).textContent = parseFloat(e.target.value).toFixed(dp);
    });
  });

  async function runPredict() {
    const btn = document.getElementById("pred-btn");
    btn.disabled = true;
    btn.textContent = "Checking...";

    const payload = {
      income: parseFloat(document.getElementById("income").value),
      loan_amount: parseFloat(document.getElementById("loan_amount").value),
      credit_score: parseFloat(document.getElementById("credit_score").value),
      employment_years: parseFloat(document.getElementById("employment_years").value),
      existing_debt: parseFloat(document.getElementById("existing_debt").value),
      dependents: parseInt(document.getElementById("dependents").value),
      loan_term_months: parseInt(document.getElementById("loan_term_months").value)
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
      document.getElementById("r-pct").textContent = data.approval_pct + "%";
      document.getElementById("r-meter").style.width = data.approval_pct + "%";
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
      btn.textContent = "Check Approval";
    }
  }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting Loan Approval Predictor...")
    print("Open http://localhost:5000")
    app.run(debug=True, port=5000)
