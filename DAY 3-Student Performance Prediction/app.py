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

grade_labels = {0: "F", 1: "C", 2: "B", 3: "A"}
grade_colors = {"A": "#4CAF50", "B": "#2196F3", "C": "#FF9800", "F": "#F44336"}
grade_messages = {
    "A": "Excellent performance. Keep it up.",
    "B": "Good work. A bit more effort and you can reach A.",
    "C": "Average. Focus on study hours and attendance.",
    "F": "At risk of failing. Immediate improvement needed."
}


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data["study_hours_per_day"]),
            float(data["attendance_percent"]),
            float(data["previous_score"]),
            float(data["sleep_hours"]),
            float(data["assignments_completed"]),
            int(data["parental_education_level"]),
            int(data["internet_access"]),
            int(data["tutoring"])
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        grade = grade_labels[prediction]

        all_probs = {
            grade_labels[i]: round(float(p) * 100, 1)
            for i, p in enumerate(probabilities)
        }

        return jsonify({
            "grade": grade,
            "confidence": round(float(probabilities[prediction]) * 100, 1),
            "color": grade_colors[grade],
            "message": grade_messages[grade],
            "all_probs": all_probs
        })

    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Student Performance Predictor - Day 3</title>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0e1117;
      --surface: #161b27;
      --border: #222840;
      --text: #e8edf5;
      --muted: #6b7694;
      --accent: #7eb8f7;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Source Sans 3', sans-serif;
      min-height: 100vh;
      padding: 48px 20px;
    }

    .wrapper { max-width: 700px; margin: 0 auto; }

    .day-tag {
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 14px;
    }

    h1 {
      font-family: 'Playfair Display', serif;
      font-size: 2.6rem;
      font-weight: 700;
      line-height: 1.1;
      margin-bottom: 10px;
    }

    .subtitle {
      font-size: 15px;
      color: var(--muted);
      font-weight: 300;
      margin-bottom: 36px;
      line-height: 1.6;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 32px;
      margin-bottom: 18px;
    }

    .section-label {
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 24px;
    }

    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px 32px; }

    @media (max-width: 520px) {
      .grid { grid-template-columns: 1fr; }
      h1 { font-size: 2rem; }
    }

    .field label {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 9px;
      font-weight: 400;
    }

    .field label span {
      font-family: 'Playfair Display', serif;
      font-size: 17px;
      color: var(--text);
    }

    input[type="range"] {
      -webkit-appearance: none;
      width: 100%;
      height: 2px;
      background: var(--border);
      border-radius: 2px;
      outline: none;
      cursor: pointer;
    }

    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 15px;
      height: 15px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(126,184,247,0.15);
      transition: box-shadow 0.15s;
    }

    input[type="range"]::-webkit-slider-thumb:hover {
      box-shadow: 0 0 0 6px rgba(126,184,247,0.2);
    }

    .range-hints {
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: var(--muted);
      margin-top: 5px;
    }

    .toggle-group {
      display: flex;
      gap: 8px;
      margin-top: 2px;
    }

    .toggle-btn {
      flex: 1;
      padding: 8px 0;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: transparent;
      color: var(--muted);
      font-size: 13px;
      cursor: pointer;
      transition: all 0.15s;
      font-family: 'Source Sans 3', sans-serif;
    }

    .toggle-btn.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #0e1117;
      font-weight: 500;
    }

    .select-field {
      width: 100%;
      padding: 9px 12px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-family: 'Source Sans 3', sans-serif;
      font-size: 13px;
      outline: none;
      cursor: pointer;
    }

    .btn {
      width: 100%;
      margin-top: 28px;
      padding: 14px;
      background: var(--accent);
      color: #0e1117;
      border: none;
      border-radius: 50px;
      font-family: 'Source Sans 3', sans-serif;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.12s;
    }

    .btn:hover { opacity: 0.85; transform: translateY(-1px); }
    .btn:active { transform: translateY(0); }
    .btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }

    .result-card {
      display: none;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 28px 32px;
      animation: rise 0.4s ease both;
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .grade-display {
      display: flex;
      align-items: center;
      gap: 20px;
      margin-bottom: 16px;
    }

    .grade-letter {
      font-family: 'Playfair Display', serif;
      font-size: 4rem;
      font-weight: 700;
      line-height: 1;
    }

    .grade-info-text { font-size: 13px; color: var(--muted); margin-top: 4px; }

    .grade-message {
      font-size: 14px;
      color: var(--muted);
      line-height: 1.6;
      padding-top: 16px;
      border-top: 1px solid var(--border);
      margin-bottom: 20px;
    }

    .prob-label {
      font-size: 11px;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 12px;
    }

    .prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
    .prob-name { font-size: 12px; width: 24px; color: var(--muted); }
    .prob-track { flex: 1; height: 4px; background: var(--border); border-radius: 4px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 4px; transition: width 0.55s cubic-bezier(0.22, 1, 0.36, 1); }
    .prob-pct { font-size: 12px; width: 36px; text-align: right; color: var(--muted); }

    footer { text-align: center; margin-top: 32px; font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>

<div class="wrapper">

  <div class="day-tag">Day 3 of 100</div>
  <h1>Student Performance<br>Predictor</h1>
  <p class="subtitle">Logistic Regression model trained on 2,000 students. Fill in the details to predict the grade.</p>

  <div class="card">
    <div class="section-label">Academic Factors</div>
    <div class="grid">

      <div class="field">
        <label>Study Hours/Day <span id="v-study">5.0</span></label>
        <input type="range" id="study_hours_per_day" min="0" max="10" step="0.5" value="5.0">
        <div class="range-hints"><span>0 hrs</span><span>10 hrs</span></div>
      </div>

      <div class="field">
        <label>Attendance <span id="v-att">75</span>%</label>
        <input type="range" id="attendance_percent" min="40" max="100" step="1" value="75">
        <div class="range-hints"><span>40%</span><span>100%</span></div>
      </div>

      <div class="field">
        <label>Previous Score <span id="v-prev">65</span></label>
        <input type="range" id="previous_score" min="30" max="100" step="1" value="65">
        <div class="range-hints"><span>30</span><span>100</span></div>
      </div>

      <div class="field">
        <label>Sleep Hours <span id="v-sleep">7.0</span></label>
        <input type="range" id="sleep_hours" min="4" max="9" step="0.5" value="7.0">
        <div class="range-hints"><span>4 hrs</span><span>9 hrs</span></div>
      </div>

      <div class="field">
        <label>Assignments Done <span id="v-assign">70</span>%</label>
        <input type="range" id="assignments_completed" min="0" max="100" step="1" value="70">
        <div class="range-hints"><span>0%</span><span>100%</span></div>
      </div>

      <div class="field">
        <label>Parent Education Level</label>
        <select class="select-field" id="parental_education_level">
          <option value="0">No formal education</option>
          <option value="1">High school</option>
          <option value="2" selected>Bachelor's degree</option>
          <option value="3">Master's or higher</option>
        </select>
      </div>

    </div>
  </div>

  <div class="card">
    <div class="section-label">Resources</div>
    <div class="grid">

      <div class="field">
        <label style="margin-bottom:10px;">Internet Access</label>
        <div class="toggle-group">
          <button class="toggle-btn" id="inet-no" onclick="setToggle('internet_access', 0)">No</button>
          <button class="toggle-btn active" id="inet-yes" onclick="setToggle('internet_access', 1)">Yes</button>
        </div>
      </div>

      <div class="field">
        <label style="margin-bottom:10px;">Private Tutoring</label>
        <div class="toggle-group">
          <button class="toggle-btn active" id="tutor-no" onclick="setToggle('tutoring', 0)">No</button>
          <button class="toggle-btn" id="tutor-yes" onclick="setToggle('tutoring', 1)">Yes</button>
        </div>
      </div>

    </div>

    <button class="btn" id="predict-btn" onclick="predict()">Predict Grade</button>
  </div>

  <div class="result-card" id="result-card">
    <div class="grade-display">
      <div class="grade-letter" id="r-grade">A</div>
      <div>
        <div style="font-size:16px;font-weight:500;" id="r-grade-name">Grade A</div>
        <div class="grade-info-text" id="r-conf">Confidence: 0%</div>
      </div>
    </div>
    <div class="grade-message" id="r-message"></div>
    <div class="prob-label">Probability breakdown</div>
    <div class="prob-row">
      <div class="prob-name">A</div>
      <div class="prob-track"><div class="prob-fill" id="b-A" style="width:0%;background:#4CAF50"></div></div>
      <div class="prob-pct" id="p-A">0%</div>
    </div>
    <div class="prob-row">
      <div class="prob-name">B</div>
      <div class="prob-track"><div class="prob-fill" id="b-B" style="width:0%;background:#2196F3"></div></div>
      <div class="prob-pct" id="p-B">0%</div>
    </div>
    <div class="prob-row">
      <div class="prob-name">C</div>
      <div class="prob-track"><div class="prob-fill" id="b-C" style="width:0%;background:#FF9800"></div></div>
      <div class="prob-pct" id="p-C">0%</div>
    </div>
    <div class="prob-row">
      <div class="prob-name">F</div>
      <div class="prob-track"><div class="prob-fill" id="b-F" style="width:0%;background:#F44336"></div></div>
      <div class="prob-pct" id="p-F">0%</div>
    </div>
  </div>

  <footer></footer>

</div>

<script>
  const toggleState = { internet_access: 1, tutoring: 0 };

  function setToggle(field, val) {
    toggleState[field] = val;
    if (field === "internet_access") {
      document.getElementById("inet-no").classList.toggle("active", val === 0);
      document.getElementById("inet-yes").classList.toggle("active", val === 1);
    } else {
      document.getElementById("tutor-no").classList.toggle("active", val === 0);
      document.getElementById("tutor-yes").classList.toggle("active", val === 1);
    }
  }

  const sliders = [
    { id: "study_hours_per_day",  out: "v-study", dp: 1 },
    { id: "attendance_percent",   out: "v-att",   dp: 0 },
    { id: "previous_score",       out: "v-prev",  dp: 0 },
    { id: "sleep_hours",          out: "v-sleep", dp: 1 },
    { id: "assignments_completed",out: "v-assign",dp: 0 }
  ];

  sliders.forEach(({ id, out, dp }) => {
    document.getElementById(id).addEventListener("input", e => {
      document.getElementById(out).textContent = parseFloat(e.target.value).toFixed(dp);
    });
  });

  async function predict() {
    const btn = document.getElementById("predict-btn");
    btn.disabled = true;
    btn.textContent = "Predicting...";

    const payload = {
      study_hours_per_day: parseFloat(document.getElementById("study_hours_per_day").value),
      attendance_percent: parseFloat(document.getElementById("attendance_percent").value),
      previous_score: parseFloat(document.getElementById("previous_score").value),
      sleep_hours: parseFloat(document.getElementById("sleep_hours").value),
      assignments_completed: parseFloat(document.getElementById("assignments_completed").value),
      parental_education_level: parseInt(document.getElementById("parental_education_level").value),
      internet_access: toggleState.internet_access,
      tutoring: toggleState.tutoring
    };

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      document.getElementById("r-grade").textContent = data.grade;
      document.getElementById("r-grade").style.color = data.color;
      document.getElementById("r-grade-name").textContent = "Grade " + data.grade;
      document.getElementById("r-conf").textContent = "Confidence: " + data.confidence + "%";
      document.getElementById("r-message").textContent = data.message;

      for (const g of ["A", "B", "C", "F"]) {
        const pct = data.all_probs[g] || 0;
        document.getElementById("b-" + g).style.width = pct + "%";
        document.getElementById("p-" + g).textContent = pct + "%";
      }

      const card = document.getElementById("result-card");
      card.style.display = "block";
      card.style.animation = "none";
      card.offsetHeight;
      card.style.animation = "";

    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Predict Grade";
    }
  }
</script>

</body>
</html>
"""

if __name__ == "__main__":
    print("Starting Student Performance Predictor...")
    print("Open http://localhost:5000")
    app.run(debug=True, port=5000)
