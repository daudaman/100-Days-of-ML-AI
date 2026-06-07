# Day 3 - Student Performance Prediction

Predicting a student's final grade (A, B, C, or F) from study habits, attendance, previous scores, and other factors. This is a multiclass classification problem.

## Dataset

2000 students with 8 features. Generated with realistic relationships between study behavior and academic outcomes. Grades are perfectly balanced — 500 students per grade.

Features: study hours per day, attendance %, previous score, sleep hours, assignments completed %, parental education level, internet access, tutoring.

## Models compared

- Logistic Regression — 70.5% accuracy (best)
- Gradient Boosting — 63.8% accuracy
- Random Forest — 62.8% accuracy

Logistic Regression won because the class boundaries in this dataset are roughly linear. This is a good reminder that simpler models often outperform complex ones.

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 70.5% |
| Gradient Boosting | 63.8% |
| Random Forest | 62.8% |

The hardest predictions are B vs C — students in the middle of the distribution overlap a lot. F and A students are predicted more reliably.

Most important features: study_hours_per_day and previous_score.

## How to run

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open http://localhost:5000

## Project structure

```
Day-003-Student-Performance-Prediction/
├── train.py
├── app.py
├── notebook.ipynb
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── feature_cols.pkl
├── dataset/
│   └── students.csv
└── screenshots/
    ├── grade_distribution.png
    ├── feature_distributions.png
    ├── correlation.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

## Stack

Python, scikit-learn, pandas, matplotlib, seaborn, Flask
