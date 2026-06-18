# Day 9 - Logistic Regression

Predicting loan approval using Logistic Regression. This is the classic go-to algorithm for binary classification — fast, interpretable, and a strong baseline before trying anything fancier.

## Dataset

2000 loan applications with 7 features: income, loan amount, credit score, employment years, existing debt, dependents, and loan term.

## How Logistic Regression works

Unlike Linear Regression which predicts continuous values, Logistic Regression passes a weighted sum of features through a sigmoid function to output a probability between 0 and 1. A threshold (usually 0.5) converts that probability into a class.

## Results

Test Accuracy: 97.25%
ROC AUC: 0.998

| Class    | Precision | Recall | F1   |
|----------|-----------|--------|------|
| Rejected | 0.97      | 0.96   | 0.97 |
| Approved | 0.98      | 0.98   | 0.98 |

This dataset has strong, mostly linear relationships between features and approval (credit score and debt-to-income ratio are the biggest drivers), which is exactly the kind of problem Logistic Regression excels at.

## Feature impact

Credit score and debt-to-income ratio had the largest coefficients. Higher credit score increases approval odds; higher existing debt relative to income decreases it.

## How to run

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open http://localhost:5000

## Project structure

```
Day-009-Logistic-Regression/
├── train.py
├── app.py
├── notebook.ipynb
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── feature_cols.pkl
├── dataset/
│   └── loans.csv
└── screenshots/
    ├── distribution.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_coefficients.png
```

## Stack

Python, scikit-learn, pandas, matplotlib, seaborn, Flask
