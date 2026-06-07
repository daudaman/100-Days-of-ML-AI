import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


os.makedirs("dataset", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)


def generate_dataset(n=2000):
    np.random.seed(42)

    study_hours = np.random.uniform(0, 10, n)
    attendance = np.random.uniform(40, 100, n)
    prev_score = np.random.uniform(30, 100, n)
    sleep_hours = np.random.uniform(4, 9, n)
    assignments_done = np.random.uniform(0, 100, n)
    parental_education = np.random.randint(0, 4, n)
    internet_access = np.random.randint(0, 2, n)
    tutoring = np.random.randint(0, 2, n)

    score = (
        study_hours * 3.5
        + attendance * 0.22
        + prev_score * 0.3
        + sleep_hours * 1.0
        + assignments_done * 0.18
        + parental_education * 1.5
        + internet_access * 3
        + tutoring * 4
        + np.random.normal(0, 7, n)
    )

    p75 = np.percentile(score, 75)
    p50 = np.percentile(score, 50)
    p25 = np.percentile(score, 25)

    grades = []
    for s in score:
        if s >= p75:
            grades.append("A")
        elif s >= p50:
            grades.append("B")
        elif s >= p25:
            grades.append("C")
        else:
            grades.append("F")

    df = pd.DataFrame({
        "study_hours_per_day": study_hours.round(1),
        "attendance_percent": attendance.round(1),
        "previous_score": prev_score.round(1),
        "sleep_hours": sleep_hours.round(1),
        "assignments_completed": assignments_done.round(1),
        "parental_education_level": parental_education,
        "internet_access": internet_access,
        "tutoring": tutoring,
        "grade": grades
    })

    return df


if not os.path.exists("dataset/students.csv"):
    df = generate_dataset()
    df.to_csv("dataset/students.csv", index=False)
    print("Dataset created.")

df = pd.read_csv("dataset/students.csv")

print("Shape:", df.shape)
print("\nGrade distribution:")
print(df["grade"].value_counts().sort_index())


colors = {"A": "#4CAF50", "B": "#2196F3", "C": "#FF9800", "F": "#F44336"}

plt.figure(figsize=(7, 5))
grade_counts = df["grade"].value_counts().sort_index()
plt.bar(grade_counts.index, grade_counts.values,
        color=[colors[g] for g in grade_counts.index])
plt.title("Grade Distribution")
plt.xlabel("Grade")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("screenshots/grade_distribution.png", dpi=130, bbox_inches="tight")
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, feature in zip(axes.flatten(),
                        ["study_hours_per_day", "attendance_percent",
                         "previous_score", "sleep_hours"]):
    for grade, color in colors.items():
        subset = df[df["grade"] == grade]
        ax.hist(subset[feature], alpha=0.6, label=grade, color=color, bins=20)
    ax.set_title(feature)
    ax.legend(fontsize=8)

plt.suptitle("Feature Distributions by Grade", fontsize=13)
plt.tight_layout()
plt.savefig("screenshots/feature_distributions.png", dpi=130, bbox_inches="tight")
plt.close()


plt.figure(figsize=(9, 7))
numeric_cols = df.select_dtypes(include=np.number).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("screenshots/correlation.png", dpi=130, bbox_inches="tight")
plt.close()


feature_cols = [
    "study_hours_per_day", "attendance_percent", "previous_score",
    "sleep_hours", "assignments_completed", "parental_education_level",
    "internet_access", "tutoring"
]

grade_map = {"A": 3, "B": 2, "C": 1, "F": 0}
X = df[feature_cols]
y = df["grade"].map(grade_map)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name}  Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds,
          target_names=["F", "C", "B", "A"], zero_division=0))


best_model = models["Logistic Regression"]
best_preds = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["F", "C", "B", "A"],
            yticklabels=["F", "C", "B", "A"])
plt.title("Confusion Matrix - Logistic Regression")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.close()


rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(9, 5))
plt.bar(
    [feature_cols[i] for i in sorted_idx],
    importances[sorted_idx],
    color="#2196F3"
)
plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("screenshots/feature_importance.png", dpi=130, bbox_inches="tight")
plt.close()


with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("\nModel saved.")
print("Done. Run app.py to start the web app.")
