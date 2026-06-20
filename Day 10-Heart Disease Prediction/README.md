# Day 10 - Heart Disease Prediction (Model Comparison)

This day takes a different approach than the earlier classification projects: instead of picking one algorithm upfront, it trains and compares 7 different models on the same problem and picks the winner objectively.

## Dataset

2000 patient records (newly generated, separate from Day 6) with 8 features: age, resting blood pressure, cholesterol, max heart rate, oldpeak, chest pain type, exercise-induced angina, and ST slope.

## Models compared

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Gradient Boosting | 74.2% | **79.3%** (best) |
| Naive Bayes | 75.7% | 78.3% |
| Logistic Regression | 76.4% | 77.8% |
| SVM | 73.9% | 77.0% |
| KNN | 72.9% | 75.8% |
| Random Forest | 73.9% | 76.5% |
| Decision Tree | 68.2% | 69.0% |

Gradient Boosting won. It builds trees sequentially, where each new tree corrects the errors of the previous ones — this usually gives it an edge on tabular data like this.

## Results (best model)

Test Accuracy: 79.25%

| Class      | Precision | Recall | F1   |
|------------|-----------|--------|------|
| No Disease | 0.81      | 0.78   | 0.80 |
| Disease    | 0.77      | 0.80   | 0.79 |

## Why compare models instead of picking one

Different algorithms make different assumptions about the data. Logistic Regression assumes roughly linear relationships. KNN assumes similar points have similar outcomes. Tree-based models can capture non-linear interactions automatically. Running them all and comparing on the same train/test split removes the guesswork.

## How to run

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open http://localhost:5000

## Project structure

```
Day-010-Heart-Disease-Prediction/
├── train.py
├── app.py
├── notebook.ipynb
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── feature_cols.pkl
├── best_model_name.pkl
├── dataset/
│   └── heart_v2.csv
└── screenshots/
    ├── distribution.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    └── roc_comparison.png
```

## Stack

Python, scikit-learn, pandas, matplotlib, seaborn, Flask
