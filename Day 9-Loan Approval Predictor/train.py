import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc


os.makedirs("dataset", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)


def generate_dataset(n=2000):
    np.random.seed(42)

    income = np.random.uniform(2000, 25000, n)
    loan_amount = np.random.uniform(50000, 800000, n)
    credit_score = np.random.uniform(300, 850, n)
    employment_years = np.random.uniform(0, 30, n)
    existing_debt = np.random.uniform(0, 50000, n)
    dependents = np.random.randint(0, 5, n)
    loan_term_months = np.random.choice([12, 24, 36, 48, 60], n)

    debt_to_income = existing_debt / (income + 1)
    loan_to_income = loan_amount / (income * 12 + 1)

    approval_score = (
        (credit_score - 580) * 0.01
        + (income - 8000) * 0.0003
        - loan_to_income * 2.0
        - debt_to_income * 1.5
        + employment_years * 0.05
        - dependents * 0.1
        + np.random.normal(0, 0.8, n)
    )

    threshold = np.percentile(approval_score, 40)
    approved = (approval_score >= threshold).astype(int)

    df = pd.DataFrame({
        "income": income.round(0),
        "loan_amount": loan_amount.round(0),
        "credit_score": credit_score.round(0),
        "employment_years": employment_years.round(1),
        "existing_debt": existing_debt.round(0),
        "dependents": dependents,
        "loan_term_months": loan_term_months,
        "approved": approved
    })

    return df


if not os.path.exists("dataset/loans.csv"):
    df = generate_dataset()
    df.to_csv("dataset/loans.csv", index=False)
    print("Dataset created.")

df = pd.read_csv("dataset/loans.csv")

print("Shape:", df.shape)
print("\nApproval distribution:")
print(df["approved"].value_counts())

feature_cols = ["income", "loan_amount", "credit_score", "employment_years",
                "existing_debt", "dependents", "loan_term_months"]

X = df[feature_cols]
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

preds = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]
acc = accuracy_score(y_test, preds)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))


plt.figure(figsize=(7, 5))
counts = df["approved"].value_counts()
plt.bar(["Rejected", "Approved"], counts.values, color=["#F44336", "#4CAF50"])
plt.title("Loan Approval Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("screenshots/distribution.png", dpi=130, bbox_inches="tight")
plt.close()

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"])
plt.title("Confusion Matrix - Logistic Regression")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.close()

fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("screenshots/roc_curve.png", dpi=130, bbox_inches="tight")
plt.close()

coefficients = model.coef_[0]
sorted_idx = np.argsort(np.abs(coefficients))[::-1]

plt.figure(figsize=(9, 5))
colors_bar = ["#4CAF50" if c > 0 else "#F44336" for c in coefficients[sorted_idx]]
plt.bar([feature_cols[i] for i in sorted_idx], coefficients[sorted_idx], color=colors_bar)
plt.title("Feature Coefficients - Logistic Regression")
plt.ylabel("Coefficient (impact on approval)")
plt.xticks(rotation=25, ha="right")
plt.axhline(y=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("screenshots/feature_coefficients.png", dpi=130, bbox_inches="tight")
plt.close()

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print(f"\nROC AUC: {roc_auc:.4f}")
print("Model saved. Run app.py to start the web app.")
