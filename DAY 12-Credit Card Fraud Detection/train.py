"""
Day 12 - Credit Card Fraud Detection
Train script: loads data, runs EDA, trains model, saves artifacts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

#  Paths 
DATA_PATH   = "dataset/creditcard.csv"
SHOT_DIR    = "screenshots"
MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEAT_PATH   = "feature_cols.pkl"

os.makedirs(SHOT_DIR, exist_ok=True)

#  1. Load 
print("Loading dataset ")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}  |  Fraud rate: {df['Class'].mean()*100:.3f}%")

#  2. EDA 
# Class distribution
counts = df["Class"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Legitimate", "Fraud"], [counts[0], counts[1]], color=["#2196F3", "#F44336"])
ax.set_title("Class Distribution")
ax.set_ylabel("Number of Transactions")
for i, v in enumerate([counts[0], counts[1]]):
    ax.text(i, v + 500, f"{v:,}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{SHOT_DIR}/class_distribution.png", dpi=100)
plt.close()
print("Saved class_distribution.png")

# Correlation heatmap (V1-V10 + Amount + Class)
cols_heat = [f"V{i}" for i in range(1, 11)] + ["Amount", "Class"]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[cols_heat].corr(), cmap="coolwarm", center=0,
            annot=False, linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap (V1-V10, Amount, Class)")
plt.tight_layout()
plt.savefig(f"{SHOT_DIR}/correlation_heatmap.png", dpi=100)
plt.close()
print("Saved correlation_heatmap.png")

#  3. Feature Scaling 
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])

feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
X = df[feature_cols]
y = df["Class"]

joblib.dump(scaler,       SCALER_PATH)
joblib.dump(feature_cols, FEAT_PATH)
print("Saved scaler.pkl and feature_cols.pkl")

#  4. Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

#  5. Train Models 
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
}

results = {}
for name, clf in models.items():
    print(f"\nTraining {name} ")
    clf.fit(X_train, y_train)
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    results[name] = {
        "model":     clf,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }
    r = results[name]
    print(f"  Accuracy : {r['accuracy']:.4f}")
    print(f"  Precision: {r['precision']:.4f}")
    print(f"  Recall   : {r['recall']:.4f}")
    print(f"  F1       : {r['f1']:.4f}")
    print(f"  ROC-AUC  : {r['roc_auc']:.4f}")

# Best model = Random Forest
best = results["Random Forest"]
joblib.dump(best["model"], MODEL_PATH)
print("\nSaved model.pkl  (Random Forest)")

#  6. Visualizations 
# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"], ax=ax)
ax.set_title("Confusion Matrix - Random Forest")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{SHOT_DIR}/confusion_matrix.png", dpi=100)
plt.close()
print("Saved confusion_matrix.png")

# ROC Curve (both models)
fig, ax = plt.subplots(figsize=(7, 5))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
plt.tight_layout()
plt.savefig(f"{SHOT_DIR}/roc_curve.png", dpi=100)
plt.close()
print("Saved roc_curve.png")

# Feature Importance
fi = pd.Series(
    best["model"].feature_importances_, index=feature_cols
).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 5))
fi.sort_values().plot(kind="barh", color="#2196F3", ax=ax)
ax.set_title("Top 15 Feature Importances - Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{SHOT_DIR}/feature_importance.png", dpi=100)
plt.close()
print("Saved feature_importance.png")

print("\nAll done. Artifacts saved.")
