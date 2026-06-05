import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data["bedrooms"]),
            float(data["bathrooms"]),
            float(data["sqft_living"]),
            float(data["age_years"]),
            float(data["garage_spaces"]),
            float(data["floors"]),
            float(data["location_score"]),
            float(data["distance_to_city_km"])
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            "price": round(float(prediction), 0),
            "price_formatted": f"${prediction:,.0f}"
        })

    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>House Price Predictor - Day 2</title>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=Karla:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #f5f2ee;
      --card: #ffffff;
      --border: #e0dbd4;
      --text: #1a1a1a;
      --muted: #7a7570;
      --accent: #c8522a;
      --accent-light: #fdf0eb;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Karla', sans-serif;
      min-height: 100vh;
      padding: 48px 20px;
    }

    .wrapper {
      max-width: 680px;
      margin: 0 auto;
    }

    .tag {
      font-family: 'Syne', sans-serif;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 16px;
    }

    h1 {
      font-family: 'Syne', sans-serif;
      font-size: 2.6rem;
      font-weight: 700;
      line-height: 1.1;
      margin-bottom: 10px;
    }

    .subtitle {
      font-size: 15px;
      color: var(--muted);
      font-weight: 300;
      margin-bottom: 40px;
      line-height: 1.6;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 36px;
      margin-bottom: 20px;
    }

    .section-label {
      font-family: 'Syne', sans-serif;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 24px;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }

    @media (max-width: 500px) {
      .grid { grid-template-columns: 1fr; }
      h1 { font-size: 2rem; }
    }

    .field label {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 500;
      color: var(--muted);
      margin-bottom: 8px;
    }

    .field label span {
      font-family: 'Syne', sans-serif;
      font-size: 16px;
      font-weight: 600;
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
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(200, 82, 42, 0.15);
      transition: box-shadow 0.15s;
    }

    input[type="range"]::-webkit-slider-thumb:hover {
      box-shadow: 0 0 0 6px rgba(200, 82, 42, 0.2);
    }

    .range-hints {
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: #bbb;
      margin-top: 4px;
    }

    .btn {
      width: 100%;
      margin-top: 28px;
      padding: 15px;
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 50px;
      font-family: 'Syne', sans-serif;
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.5px;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.12s;
    }

    .btn:hover { opacity: 0.88; transform: translateY(-1px); }
    .btn:active { transform: translateY(0); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    .result-card {
      display: none;
      background: var(--accent-light);
      border: 1px solid rgba(200,82,42,0.2);
      border-radius: 16px;
      padding: 32px 36px;
      animation: rise 0.4s ease both;
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .result-label {
      font-size: 13px;
      font-weight: 500;
      color: var(--accent);
      margin-bottom: 6px;
      font-family: 'Syne', sans-serif;
      letter-spacing: 1px;
      text-transform: uppercase;
    }

    .result-price {
      font-family: 'Syne', sans-serif;
      font-size: 3rem;
      font-weight: 700;
      color: var(--accent);
      line-height: 1;
      margin-bottom: 14px;
    }

    .result-note {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.6;
    }

    footer {
      text-align: center;
      margin-top: 32px;
      font-size: 12px;
      color: var(--muted);
    }
  </style>
</head>
<body>

<div class="wrapper">

  <div class="tag"></div>
  <h1>House Price<br>Predictor</h1>
  <p class="subtitle">
    Gradient Boosting model trained on 2,000 houses. Adjust the details and get an instant estimate.
  </p>

  <div class="card">
    <div class="section-label">Property Details</div>

    <div class="grid">

      <div class="field">
        <label>Bedrooms <span id="v-bed">3</span></label>
        <input type="range" id="bedrooms" min="1" max="6" step="1" value="3">
        <div class="range-hints"><span>1</span><span>6</span></div>
      </div>

      <div class="field">
        <label>Bathrooms <span id="v-bath">2</span></label>
        <input type="range" id="bathrooms" min="1" max="5" step="1" value="2">
        <div class="range-hints"><span>1</span><span>5</span></div>
      </div>

      <div class="field">
        <label>Living Area (sqft) <span id="v-sqft">1500</span></label>
        <input type="range" id="sqft_living" min="500" max="5000" step="50" value="1500">
        <div class="range-hints"><span>500</span><span>5000</span></div>
      </div>

      <div class="field">
        <label>Age (years) <span id="v-age">10</span></label>
        <input type="range" id="age_years" min="0" max="60" step="1" value="10">
        <div class="range-hints"><span>0</span><span>60</span></div>
      </div>

      <div class="field">
        <label>Garage Spaces <span id="v-garage">1</span></label>
        <input type="range" id="garage_spaces" min="0" max="3" step="1" value="1">
        <div class="range-hints"><span>0</span><span>3</span></div>
      </div>

      <div class="field">
        <label>Floors <span id="v-floors">1</span></label>
        <input type="range" id="floors" min="1" max="3" step="1" value="1">
        <div class="range-hints"><span>1</span><span>3</span></div>
      </div>

      <div class="field">
        <label>Location Score <span id="v-loc">7.0</span></label>
        <input type="range" id="location_score" min="1" max="10" step="0.1" value="7.0">
        <div class="range-hints"><span>1</span><span>10</span></div>
      </div>

      <div class="field">
        <label>Distance to City (km) <span id="v-dist">15</span></label>
        <input type="range" id="distance_to_city_km" min="1" max="50" step="1" value="15">
        <div class="range-hints"><span>1 km</span><span>50 km</span></div>
      </div>

    </div>

    <button class="btn" id="predict-btn" onclick="predict()">Predict Price</button>
  </div>

  <div class="result-card" id="result-card">
    <div class="result-label">Estimated Price</div>
    <div class="result-price" id="result-price">$0</div>
    <div class="result-note" id="result-note"></div>
  </div>

  

</div>

<script>
  const fields = [
    { id: "bedrooms",            out: "v-bed",    dp: 0 },
    { id: "bathrooms",           out: "v-bath",   dp: 0 },
    { id: "sqft_living",         out: "v-sqft",   dp: 0 },
    { id: "age_years",           out: "v-age",    dp: 0 },
    { id: "garage_spaces",       out: "v-garage", dp: 0 },
    { id: "floors",              out: "v-floors", dp: 0 },
    { id: "location_score",      out: "v-loc",    dp: 1 },
    { id: "distance_to_city_km", out: "v-dist",   dp: 0 }
  ];

  fields.forEach(({ id, out, dp }) => {
    const el = document.getElementById(id);
    el.addEventListener("input", () => {
      document.getElementById(out).textContent = parseFloat(el.value).toFixed(dp);
    });
  });

  async function predict() {
    const btn = document.getElementById("predict-btn");
    btn.disabled = true;
    btn.textContent = "Predicting...";

    const payload = {};
    fields.forEach(({ id }) => {
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

      document.getElementById("result-price").textContent = data.price_formatted;

      const sqft = payload["sqft_living"];
      const pricePerSqft = Math.round(data.price / sqft);
      const age = payload["age_years"];
      const ageText = age === 0 ? "brand new" : `${age} years old`;

      document.getElementById("result-note").textContent =
        `${sqft.toLocaleString()} sqft at $${pricePerSqft.toLocaleString()}/sqft. ` +
        `The house is ${ageText} with a location score of ${payload["location_score"].toFixed(1)}/10.`;

      const card = document.getElementById("result-card");
      card.style.display = "block";
      card.style.animation = "none";
      card.offsetHeight;
      card.style.animation = "";

    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Predict Price";
    }
  }
</script>

</body>
</html>
"""


if __name__ == "__main__":
    print("Starting House Price Predictor...")
    print("Open http://localhost:5000")
    app.run(debug=True, port=5000)
