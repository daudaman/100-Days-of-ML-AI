import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc


os.makedirs("dataset", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)


def generate_dataset(n=2000):
    np.random.seed(7)

    age = np.random.randint(28, 78, n)
    resting_bp = np.random.uniform(94, 178, n)
    cholesterol = np.random.uniform(126, 410, n)
    max_hr = np.random.uniform(72, 198, n)
    oldpeak = np.random.uniform(0, 5.5, n)
    chest_pain = np.random.randint(0, 4, n)
    exercise_angina = np.random.randint(0, 2, n)
    st_slope = np.random.randint(0, 3, n)

    risk = (
        (age - 50) * 0.025
        + (resting_bp - 120) * 0.012
        + (cholesterol - 220) * 0.004
        + (180 - max_hr) * 0.012
        + oldpeak * 0.35
        + chest_pain * 0.25
        + exercise_angina * 0.55
        + (st_slope == 2) * 0.4
        + np.random.normal(0, 0.85, n)
    )

    threshold = np.percentile(risk, 52)
    target = (risk >= threshold).astype(int)

    df = pd.DataFrame({
        "age": age,
        "resting_bp": resting_bp.round(1),
        "cholesterol": cholesterol.round(1),
        "max_heart_rate": max_hr.round(1),
        "oldpeak": oldpeak.round(2),
        "chest_pain_type": chest_pain,
        "exercise_angina": exercise_angina,
        "st_slope": st_slope,
        "target": target
    })

    return df


if not os.path.exists("dataset/heart_v2.csv"):
    df = generate_dataset()
    df.to_csv("dataset/heart_v2.csv", index=False)
    print("Dataset created.")

df = pd.read_csv("dataset/heart_v2.csv")

print("Shape:", df.shape)
print("\nClass distribution:")
print(df["target"].value_counts())

feature_cols = ["age", "resting_bp", "cholesterol", "max_heart_rate",
                "oldpeak", "chest_pain_type", "exercise_angina", "st_slope"]

X = df[feature_cols]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Naive Bayes": GaussianNB()
}

results = {}

print("\nModel comparison (5-fold CV on training set):")
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    model.fit(X_train_scaled, y_train)
    test_preds = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    results[name] = {
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "test_acc": test_acc,
        "model": model
    }
    print(f"  {name:22s} CV: {scores.mean():.4f}  Test: {test_acc:.4f}")


best_name = max(results, key=lambda k: results[k]["test_acc"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (test accuracy {results[best_name]['test_acc']:.4f})")

best_preds = best_model.predict(X_test_scaled)
print("\nClassification Report (best model):")
print(classification_report(y_test, best_preds, target_names=["No Disease", "Disease"]))


plt.figure(figsize=(10, 6))
names = list(results.keys())
test_accs = [results[n]["test_acc"] for n in names]
sorted_pairs = sorted(zip(names, test_accs), key=lambda x: x[1])
names_sorted = [p[0] for p in sorted_pairs]
accs_sorted = [p[1] for p in sorted_pairs]

colors_bar = ["#4CAF50" if n == best_name else "#90A4AE" for n in names_sorted]
plt.barh(names_sorted, accs_sorted, color=colors_bar)
plt.xlabel("Test Accuracy")
plt.title("Model Comparison - Heart Disease Prediction")
plt.xlim(0.5, 1.0)
for i, v in enumerate(accs_sorted):
    plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("screenshots/model_comparison.png", dpi=130, bbox_inches="tight")
plt.close()


cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.close()


plt.figure(figsize=(7, 6))
for name in ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"]:
    model = results[name]["model"]
    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("screenshots/roc_comparison.png", dpi=130, bbox_inches="tight")
plt.close()


plt.figure(figsize=(7, 5))
counts = df["target"].value_counts()
plt.bar(["No Disease", "Disease"], counts.values, color=["#4CAF50", "#F44336"])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("screenshots/distribution.png", dpi=130, bbox_inches="tight")
plt.close()


with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
with open("best_model_name.pkl", "wb") as f:
    pickle.dump(best_name, f)

print(f"\nBest model ({best_name}) saved.")
print("Done. Run app.py to start the web app.")
