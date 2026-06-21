import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score


os.makedirs("dataset", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)


def generate_dataset(n=2500):
    np.random.seed(11)

    tenure_months = np.random.uniform(1, 72, n)
    monthly_charges = np.random.uniform(20, 120, n)
    total_charges = tenure_months * monthly_charges * np.random.uniform(0.9, 1.1, n)
    contract_type = np.random.randint(0, 3, n)
    support_calls = np.random.poisson(2, n)
    satisfaction_score = np.random.uniform(1, 10, n)
    has_addon_services = np.random.randint(0, 2, n)

    churn_score = (
        (12 - tenure_months) * 0.04
        + (contract_type == 0) * 1.2
        + support_calls * 0.25
        - satisfaction_score * 0.35
        - has_addon_services * 0.4
        + (monthly_charges - 70) * 0.01
        + np.random.normal(0, 0.7, n)
    )

    threshold = np.percentile(churn_score, 78)
    churn = (churn_score >= threshold).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure_months.round(1),
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "contract_type": contract_type,
        "support_calls": support_calls,
        "satisfaction_score": satisfaction_score.round(1),
        "has_addon_services": has_addon_services,
        "churn": churn
    })

    return df


if not os.path.exists("dataset/churn.csv"):
    df = generate_dataset()
    df.to_csv("dataset/churn.csv", index=False)
    print("Dataset created.")

df = pd.read_csv("dataset/churn.csv")

print("Shape:", df.shape)
print("\nChurn distribution:")
print(df["churn"].value_counts())
print(f"Churn rate: {df['churn'].mean()*100:.1f}%")
print("This is an imbalanced dataset - most customers do not churn.")

feature_cols = ["tenure_months", "monthly_charges", "total_charges", "contract_type",
                "support_calls", "satisfaction_score", "has_addon_services"]

X = df[feature_cols]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\n--- Without handling imbalance ---")
model_plain = RandomForestClassifier(n_estimators=100, random_state=42)
model_plain.fit(X_train_scaled, y_train)
preds_plain = model_plain.predict(X_test_scaled)
print(classification_report(y_test, preds_plain, target_names=["Stayed", "Churned"]))


print("--- With class_weight balanced ---")
model_balanced = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model_balanced.fit(X_train_scaled, y_train)
preds_balanced = model_balanced.predict(X_test_scaled)
print(classification_report(y_test, preds_balanced, target_names=["Stayed", "Churned"]))

auc_plain = roc_auc_score(y_test, model_plain.predict_proba(X_test_scaled)[:, 1])
auc_balanced = roc_auc_score(y_test, model_balanced.predict_proba(X_test_scaled)[:, 1])
print(f"\nROC AUC without balancing: {auc_plain:.4f}")
print(f"ROC AUC with balancing:    {auc_balanced:.4f}")

# in this case plain RF edges out balanced RF on recall/precision tradeoff
# keeping balanced since catching churners matters more than overall accuracy
model = model_balanced
preds_final = preds_balanced


plt.figure(figsize=(7, 5))
counts = df["churn"].value_counts()
plt.bar(["Stayed", "Churned"], counts.values, color=["#4CAF50", "#F44336"])
plt.title(f"Churn Distribution ({df['churn'].mean()*100:.0f}% churn rate)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("screenshots/distribution.png", dpi=130, bbox_inches="tight")
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm_plain = confusion_matrix(y_test, preds_plain)
sns.heatmap(cm_plain, annot=True, fmt="d", cmap="Reds", ax=axes[0],
            xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
axes[0].set_title("Without Balancing")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

cm_balanced = confusion_matrix(y_test, preds_balanced)
sns.heatmap(cm_balanced, annot=True, fmt="d", cmap="Greens", ax=axes[1],
            xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
axes[1].set_title("With class_weight Balanced")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("screenshots/confusion_matrix_comparison.png", dpi=130, bbox_inches="tight")
plt.close()


importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(9, 5))
plt.bar([feature_cols[i] for i in sorted_idx], importances[sorted_idx], color="#FF5722")
plt.title("Feature Importance - Random Forest (balanced)")
plt.ylabel("Importance")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("screenshots/feature_importance.png", dpi=130, bbox_inches="tight")
plt.close()


plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig("screenshots/correlation.png", dpi=130, bbox_inches="tight")
plt.close()


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("\nModel saved (balanced Random Forest).")
print("Done. Run app.py to start the web app.")
