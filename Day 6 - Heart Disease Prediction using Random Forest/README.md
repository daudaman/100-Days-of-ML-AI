# Day 6 - Heart Disease Prediction using Random Forest

Predicting whether a patient is at risk of heart disease using the Random Forest algorithm. This is a binary classification problem based on various medical attributes and patient health indicators.

## Dataset

303 patient records with 13 medical features.

### Features Used

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

### Target Classes

- No Heart Disease (0)
- Heart Disease (1)

The dataset contains patient information commonly used for cardiovascular disease prediction.

---

## Model

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to improve prediction accuracy and reduce overfitting.

Model configuration:

- n_estimators = 100
- random_state = 42

The model is trained using an 80/20 train-test split and evaluated using standard classification metrics.

---

## Results

| Metric | Value |
|----------|----------|
| Accuracy | ~85–95% |
| Algorithm | Random Forest |
| Classes | Heart Disease / No Heart Disease |

The Random Forest model outperformed a single Decision Tree by combining predictions from multiple trees and reducing variance.

---

## Visualizations

The training pipeline automatically generates:

- Target Distribution
- Correlation Heatmap
- Confusion Matrix
- Feature Importance Plot
- Model Comparison Plot

---

## Flask Web App

The project includes an interactive Flask application where users can enter patient medical information and receive:

- Heart Disease Prediction
- Prediction Confidence
- Disease Probability
- No Disease Probability

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open:

```text
http://localhost:5000
```

---

## Project Structure

```text
DAY 6-Random Forest Heart Disease Prediction/
│
├── train.py
├── app.py
├── requirements.txt
├── model.pkl
├── feature_names.pkl
│
├── dataset/
│   └── heart.csv
│
├── templates/
│   └── index.html
│
└── screenshots/
    ├── target_distribution.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── model_comparison.png
```

---

## Tech Stack

- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Flask

---

## Concepts Covered

- Random Forest
- Ensemble Learning
- Bagging
- Classification
- Feature Importance
- Train/Test Split
- Model Evaluation
- Confusion Matrix
- Model Serialization with Pickle
- Flask Deployment

---
