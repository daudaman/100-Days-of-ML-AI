# Day 2 - House Price Prediction

Predicting house prices using three regression models. The best model is Gradient Boosting with an MAE of ~$18,000.

## Dataset

2000 houses with 8 features: bedrooms, bathrooms, living area, age, garage spaces, floors, location score, and distance to city. Generated to reflect realistic housing price patterns.

## Models compared

- Linear Regression — R2: 0.9855
- Random Forest — R2: 0.9760
- Gradient Boosting — R2: 0.9809 (best overall, used in the app)

Gradient Boosting was chosen because it generalizes better on unseen data.

## Results

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | $15,495 | $19,118 | 0.9855 |
| Random Forest | $19,954 | $24,660 | 0.9760 |
| Gradient Boosting | $18,120 | $21,954 | 0.9809 |

sqft_living is the most important feature by a significant margin.

## How to run

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open http://localhost:5000

## Project structure

```
Day-002-House-Price-Prediction/
├── train.py
├── app.py
├── notebook.ipynb
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── feature_names.pkl
├── dataset/
│   └── housing.csv
└── screenshots/
    ├── distributions.png
    ├── correlation.png
    ├── actual_vs_predicted.png
    └── feature_importance.png
```

## Stack

Python, scikit-learn, pandas, matplotlib, seaborn, Flask
