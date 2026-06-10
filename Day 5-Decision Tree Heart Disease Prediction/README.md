# Day 5 - Heart Disease Prediction

Predicting whether a patient has heart disease based on medical measurements such as age, chest pain type, cholesterol level, blood pressure, and heart rate. This is a binary classification problem.

## Dataset

303 patient records with 13 medical features and 1 target variable.

Features include: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, maximum heart rate achieved, exercise-induced angina, ST depression, slope of ST segment, number of major vessels, and thalassemia type.

Target:
- 1 = Heart Disease
- 0 = No Heart Disease

## Models compared

- Decision Tree (Gini) — Best Model
- Decision Tree (Entropy)

Both criteria produced similar results, but the Gini-based tree achieved slightly better performance and was selected for deployment.

## Results

| Metric | Value |
|---|---|
| Accuracy | 78% - 88% |
| Criterion | Gini |
| Max Depth | 5 |

The most important features were chest pain type (`cp`) and maximum heart rate achieved (`thalach`).

A maximum depth of 5 helped reduce overfitting while maintaining good predictive performance.

## How to run

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open:

```text
http://localhost:5000
```

## Project structure

```text
Day-005-Heart-Disease-Prediction/
├── train.py
├── app.py
├── notebook.ipynb
├── requirements.txt
├── model.pkl
├── feature_names.pkl
├── dataset/
│   └── heart.csv
├── templates/
│   └── index.html
└── screenshots/
    ├── target_distribution.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── decision_tree.png
```

## Visualizations

- Target Distribution
- Correlation Heatmap
- Confusion Matrix
- Feature Importance
- Decision Tree Visualization

## Stack

Python, scikit-learn, pandas, numpy, matplotlib, seaborn, Flask, pickle