import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import os

app = Flask(__name__)

# load model and scaler once at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# brief descriptions so the UI can show something useful
class_info = {
    "Iris-setosa": {
        "desc": "Small and compact. Usually found in Arctic and alpine regions. "
                "Easy to tell apart from the other two — petal length alone is enough.",
        "emoji": "🌸",
        "color": "#4CAF50"
    },
    "Iris-versicolor": {
        "desc": "The blue flag iris. Common across eastern North America. "
                "Overlaps quite a bit with virginica, which is why the model sometimes mixes them up.",
        "emoji": "💜",
        "color": "#2196F3"
    },
    "Iris-virginica": {
        "desc": "Largest of the three. Found in the eastern United States. "
                "Petals and sepals are noticeably bigger than versicolor.",
        "emoji": "🌺",
        "color": "#FF5722"
    }
}


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        sepal_length = float(data["sepal_length"])
        sepal_width  = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width  = float(data["petal_width"])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        predicted_class = class_names[prediction]
        info = class_info[predicted_class]

        result = {
            "prediction": predicted_class,
            "confidence": round(float(probabilities[prediction]) * 100, 1),
            "description": info["desc"],
            "emoji": info["emoji"],
            "color": info["color"],
            "all_probs": {
                class_names[i]: round(float(p) * 100, 1)
                for i, p in enumerate(probabilities)
            }
        }

        return jsonify(result)

    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


# ── HTML page (inline so the project stays single-file-ish) ──────────────────

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Iris Classifier · Day 1</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0d0f12;
      --surface:  #161a20;
      --border:   #252a33;
      --muted:    #5a6278;
      --text:     #e8eaf0;
      --accent:   #a8e063;
      --setosa:   #4CAF50;
      --versi:    #2196F3;
      --virgi:    #FF5722;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'DM Sans', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    /* faint radial glow behind the card */
    body::before {
      content: '';
      position: fixed;
      top: -200px; left: 50%;
      transform: translateX(-50%);
      width: 800px; height: 600px;
      background: radial-gradient(ellipse at center, rgba(168,224,99,0.06) 0%, transparent 70%);
      pointer-events: none;
    }

    header {
      width: 100%; max-width: 700px;
      padding: 56px 24px 32px;
      text-align: center;
    }

    .day-tag {
      display: inline-block;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--accent);
      border: 1px solid rgba(168,224,99,0.3);
      padding: 4px 12px;
      border-radius: 20px;
      margin-bottom: 20px;
    }

    h1 {
      font-family: 'DM Serif Display', serif;
      font-size: clamp(2rem, 6vw, 3.2rem);
      line-height: 1.1;
      margin-bottom: 12px;
    }

    h1 em {
      font-style: italic;
      color: var(--accent);
    }

    .subtitle {
      color: var(--muted);
      font-size: 15px;
      font-weight: 300;
      line-height: 1.6;
      max-width: 420px;
      margin: 0 auto;
    }

    /* ── card ── */
    .card {
      width: 100%; max-width: 700px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 36px;
      margin: 0 24px 48px;
    }

    .card-title {
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 28px;
    }

    /* ── sliders ── */
    .sliders-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 28px 36px;
    }

    @media (max-width: 520px) {
      .sliders-grid { grid-template-columns: 1fr; }
    }

    .slider-group label {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      font-size: 13px;
      font-weight: 500;
      color: var(--muted);
      margin-bottom: 10px;
    }

    .slider-group label span {
      font-family: 'DM Serif Display', serif;
      font-size: 22px;
      color: var(--text);
    }

    input[type="range"] {
      -webkit-appearance: none;
      width: 100%;
      height: 3px;
      background: var(--border);
      border-radius: 3px;
      outline: none;
      cursor: pointer;
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 18px; height: 18px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 4px rgba(168,224,99,0.15);
      transition: box-shadow 0.15s;
    }
    input[type="range"]::-webkit-slider-thumb:hover {
      box-shadow: 0 0 0 7px rgba(168,224,99,0.2);
    }

    .range-labels {
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: var(--muted);
      margin-top: 4px;
    }

    /* ── button ── */
    .btn-wrap {
      margin-top: 36px;
      text-align: center;
    }

    button {
      background: var(--accent);
      color: #0d0f12;
      border: none;
      padding: 14px 48px;
      border-radius: 50px;
      font-family: 'DM Sans', sans-serif;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
    }
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(168,224,99,0.25);
    }
    button:active { transform: translateY(0); }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

    /* ── result ── */
    #result {
      margin-top: 36px;
      display: none;
      animation: fadeUp 0.4s ease both;
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(14px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .result-top {
      display: flex;
      align-items: center;
      gap: 20px;
      padding: 24px;
      border-radius: 14px;
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      margin-bottom: 20px;
    }

    .result-emoji { font-size: 3rem; }

    .result-name {
      font-family: 'DM Serif Display', serif;
      font-size: 1.9rem;
    }

    .result-conf {
      font-size: 13px;
      color: var(--muted);
      margin-top: 4px;
    }

    .result-desc {
      font-size: 14px;
      color: var(--muted);
      line-height: 1.7;
      margin-top: 12px;
      font-weight: 300;
    }

    /* probability bars */
    .prob-title {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 14px;
    }

    .prob-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
    }

    .prob-label {
      font-size: 12px;
      width: 130px;
      flex-shrink: 0;
    }

    .prob-bar-track {
      flex: 1;
      height: 6px;
      background: var(--border);
      border-radius: 6px;
      overflow: hidden;
    }

    .prob-bar-fill {
      height: 100%;
      border-radius: 6px;
      transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
    }

    .prob-pct {
      font-size: 12px;
      width: 38px;
      text-align: right;
      color: var(--muted);
    }

    footer {
      color: var(--muted);
      font-size: 12px;
      padding-bottom: 32px;
      text-align: center;
    }
  </style>
</head>
<body>

  <header>
    <div class="day-tag">Day 1</div>
    <h1>Iris Flower<br><em>Classifier</em></h1>
    <p class="subtitle">
      Adjust the measurements below and the Random Forest model
      will tell you which species you're looking at.
    </p>
  </header>

  <div class="card">
    <div class="card-title">Measurements (cm)</div>

    <div class="sliders-grid">

      <div class="slider-group">
        <label>Sepal Length <span id="sl-val">5.1</span></label>
        <input type="range" id="sepal_length" min="4.3" max="7.9" step="0.1" value="5.1">
        <div class="range-labels"><span>4.3</span><span>7.9</span></div>
      </div>

      <div class="slider-group">
        <label>Sepal Width <span id="sw-val">3.5</span></label>
        <input type="range" id="sepal_width" min="2.0" max="4.4" step="0.1" value="3.5">
        <div class="range-labels"><span>2.0</span><span>4.4</span></div>
      </div>

      <div class="slider-group">
        <label>Petal Length <span id="pl-val">1.4</span></label>
        <input type="range" id="petal_length" min="1.0" max="6.9" step="0.1" value="1.4">
        <div class="range-labels"><span>1.0</span><span>6.9</span></div>
      </div>

      <div class="slider-group">
        <label>Petal Width <span id="pw-val">0.2</span></label>
        <input type="range" id="petal_width" min="0.1" max="2.5" step="0.1" value="0.2">
        <div class="range-labels"><span>0.1</span><span>2.5</span></div>
      </div>

    </div>

    <div class="btn-wrap">
      <button id="classify-btn" onclick="classify()">Classify Flower</button>
    </div>

    <div id="result">

      <div class="result-top">
        <div class="result-emoji" id="r-emoji">🌸</div>
        <div>
          <div class="result-name" id="r-name">—</div>
          <div class="result-conf" id="r-conf">—</div>
          <div class="result-desc" id="r-desc">—</div>
        </div>
      </div>

      <div class="prob-title">Probability breakdown</div>

      <div class="prob-row">
        <div class="prob-label">Iris-setosa</div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" id="bar-setosa" style="width:0%;background:#4CAF50"></div>
        </div>
        <div class="prob-pct" id="pct-setosa">0%</div>
      </div>

      <div class="prob-row">
        <div class="prob-label">Iris-versicolor</div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" id="bar-versi" style="width:0%;background:#2196F3"></div>
        </div>
        <div class="prob-pct" id="pct-versi">0%</div>
      </div>

      <div class="prob-row">
        <div class="prob-label">Iris-virginica</div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" id="bar-virgi" style="width:0%;background:#FF5722"></div>
        </div>
        <div class="prob-pct" id="pct-virgi">0%</div>
      </div>

    </div>
  </div>

  <footer>100 Days of ML/AI · Built with scikit-learn &amp; Flask</footer>

  <script>
    // update display values live as sliders move
    const sliders = {
      sepal_length: "sl-val",
      sepal_width:  "sw-val",
      petal_length: "pl-val",
      petal_width:  "pw-val"
    };

    for (const [id, labelId] of Object.entries(sliders)) {
      const el = document.getElementById(id);
      el.addEventListener("input", () => {
        document.getElementById(labelId).textContent = parseFloat(el.value).toFixed(1);
      });
    }

    async function classify() {
      const btn = document.getElementById("classify-btn");
      btn.disabled = true;
      btn.textContent = "Classifying…";

      const payload = {
        sepal_length: parseFloat(document.getElementById("sepal_length").value),
        sepal_width:  parseFloat(document.getElementById("sepal_width").value),
        petal_length: parseFloat(document.getElementById("petal_length").value),
        petal_width:  parseFloat(document.getElementById("petal_width").value)
      };

      try {
        const res  = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        // populate result
        document.getElementById("r-emoji").textContent  = data.emoji;
        document.getElementById("r-name").textContent   = data.prediction;
        document.getElementById("r-name").style.color   = data.color;
        document.getElementById("r-conf").textContent   = `Confidence: ${data.confidence}%`;
        document.getElementById("r-desc").textContent   = data.description;

        const probs = data.all_probs;
        setBar("setosa", probs["Iris-setosa"]);
        setBar("versi",  probs["Iris-versicolor"]);
        setBar("virgi",  probs["Iris-virginica"]);

        const resultEl = document.getElementById("result");
        resultEl.style.display = "block";
        // re-trigger animation
        resultEl.style.animation = "none";
        resultEl.offsetHeight;  // reflow
        resultEl.style.animation = "";

      } catch (err) {
        alert("Something went wrong: " + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = "Classify Flower";
      }
    }

    function setBar(key, pct) {
      document.getElementById("bar-" + key).style.width = pct + "%";
      document.getElementById("pct-" + key).textContent = pct + "%";
    }
  </script>

</body>
</html>
"""


if __name__ == "__main__":
    print("Starting Iris Classifier...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
