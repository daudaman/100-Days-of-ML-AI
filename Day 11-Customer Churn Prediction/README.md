# Day 11 - Customer Churn Prediction

Predicting which customers are likely to cancel their subscription. The dataset is imbalanced (22% churn rate), so this project focuses on handling that properly instead of just running a model and reporting accuracy.

## Dataset

2500 customers with 7 features: tenure, monthly charges, total charges, contract type, support calls, satisfaction score, and add-on services.

## Why imbalance matters

When only 22% of customers churn, a model can score 78% accuracy just by predicting "stayed" every single time. Accuracy alone is misleading here — precision, recall, and F1 per class matter more.

## Plain vs balanced comparison

| Approach | Stayed Recall | Churned Recall | ROC AUC |
|---|---|---|---|
| Plain Random Forest | 0.96 | 0.70 | 0.9453 |
| class_weight="balanced" | 0.96 | 0.65 | 0.9426 |

Interesting finding: balancing did not actually improve churn recall here. This happens sometimes — class weighting helps more when the imbalance is more extreme or the classes overlap less cleanly. It is a good reminder to always measure rather than assume a technique will help.

The balanced model is used in the app since catching churners is the business priority, even with a similar tradeoff.

## Results

ROC AUC: 0.943
Churned customers precision: 0.83, recall: 0.65

## How to run

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open http://localhost:5000

## Project structure

```
Day-011-Customer-Churn-Prediction/
├── train.py
├── app.py
├── notebook.ipynb
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── feature_cols.pkl
├── dataset/
│   └── churn.csv
└── screenshots/
    ├── distribution.png
    ├── confusion_matrix_comparison.png
    ├── feature_importance.png
    └── correlation.png
```

## Stack

Python, scikit-learn, pandas, matplotlib, seaborn, Flask
