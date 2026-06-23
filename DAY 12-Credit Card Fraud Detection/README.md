# Day 12 - Credit Card Fraud Detection

**100 Days of ML & AI**

Predict whether a credit card transaction is legitimate or fraudulent using a highly imbalanced real-world dataset.

---

## Project Overview

| Item | Detail |
|------|--------|
| Dataset | Credit Card Fraud Detection (Kaggle) |
| Transactions | 284,807 |
| Fraud cases | 492 (0.172%) |
| Model | Random Forest (class_weight=balanced) |
| Key metric | Recall, ROC-AUC |

---

## Why Accuracy Is Misleading

A model that predicts "Not Fraud" for every transaction achieves **99.8% accuracy** while detecting zero fraud cases.  
This project demonstrates why **Recall** and **ROC-AUC** matter far more in imbalanced classification.

---

## Folder Structure

```
DAY 12-Credit Card Fraud Detection/
|
+-- dataset/
|   +-- creditcard.csv
|
+-- screenshots/
|   +-- class_distribution.png
|   +-- correlation_heatmap.png
|   +-- confusion_matrix.png
|   +-- roc_curve.png
|   +-- feature_importance.png
|
+-- templates/
|   +-- index.html
|
+-- notebook.ipynb
+-- train.py
+-- app.py
+-- model.pkl          (generated after training)
+-- scaler.pkl         (generated after training)
+-- feature_cols.pkl   (generated after training)
+-- requirements.txt
+-- README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

Download from Kaggle:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file at:

```
dataset/creditcard.csv
```

### 3. Train the model

```bash
python train.py
```

This will:
- Run EDA and save visualizations to `screenshots/`
- Train Logistic Regression and Random Forest
- Print evaluation metrics
- Save `model.pkl`, `scaler.pkl`, `feature_cols.pkl`

### 4. Run the Flask app

```bash
python app.py
```

Open your browser at: http://localhost:5000

---

## Model Evaluation

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | ~0.97 | ~0.9996 |
| Precision | ~0.88 | ~0.95+ |
| Recall | ~0.92 | ~0.83+ |
| F1 Score | ~0.90 | ~0.89+ |
| ROC-AUC | ~0.97 | ~0.97+ |

> Results may vary slightly. Random Forest is saved as the production model.

---

## Key Concepts Covered

- **Imbalanced datasets** - only 0.172% of transactions are fraud
- **class_weight="balanced"** - compensates for class imbalance without resampling
- **Recall vs Precision** - missing a fraud is more costly than a false alarm
- **ROC-AUC** - threshold-independent measure of model quality
- **Feature Importance** - understanding which PCA components drive predictions

---

## Flask API

**POST** `/predict`

Send a JSON body with all feature column values:

```json
{
  "V1": -1.35,
  "V2": -0.07,
  ...
  "Amount_scaled": 0.24,
  "Time_scaled": -0.99
}
```

Response:

```json
{
  "prediction": 1,
  "label": "Fraud Detected",
  "fraud_probability": 94.5
}
```

---

## Dataset Source

Kaggle - Credit Card Fraud Detection  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains transactions made by European cardholders in September 2013.  
Features V1 through V28 are PCA-transformed for confidentiality.  
Only `Time` and `Amount` are the original features.
