"""
train.py — Random Forest Classifier for Heart Disease Prediction
Day 6 Project
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

#  Paths 
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "screenshots"), exist_ok=True)

#  Global plot style 
plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#C9D1D9",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "text.color":       "#C9D1D9",
    "grid.color":       "#21262D",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
})


#  Step 1 — Load Dataset 
print("=" * 55)
print("  RANDOM FOREST — HEART DISEASE PREDICTION")
print("=" * 55)

df = pd.read_csv(os.path.join(BASE, "dataset", "heart.csv"))
print(f"\n[1] Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} cols")

#  Step 2 — EDA 
print("\n[2] EDA")
print(df.describe().to_string())

# Target Distribution
fig, ax = plt.subplots(figsize=(8, 5))
counts = df["target"].value_counts()
ax.bar(["No Heart Disease", "Heart Disease"], [counts[0], counts[1]],
       color=["#238636", "#F85149"], width=0.5, edgecolor="none")
ax.set_title("Target Distribution", fontsize=16, fontweight="bold",
             color="#E6EDF3", pad=15)
ax.set_ylabel("Count", fontsize=12)
ax.set_ylim(0, max(counts) * 1.2)
for i, (val, label) in enumerate(zip([counts[0], counts[1]], ["No Disease", "Disease"])):
    ax.text(i, val + 2, str(val), ha="center", fontsize=13,
            fontweight="bold", color="#E6EDF3")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "screenshots", "target_distribution.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("      screenshots/target_distribution.png")

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdYlGn", center=0, annot=True, fmt=".2f",
            linewidths=0.5, linecolor="#0D1117", ax=ax,
            annot_kws={"size": 8, "color": "#E6EDF3"},
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold",
             color="#E6EDF3", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "screenshots", "correlation_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("      screenshots/correlation_heatmap.png")

#  Step 3 — Prepare Features 
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[3] Train/Test split  →  {len(X_train)} train / {len(X_test)} test")

#  Step 4 — Train Random Forest 
print("\n[4] Training Random Forest (n_estimators=100) …")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"      Random Forest Accuracy: {rf_acc * 100:.2f}%")

#  Step 5 — Train Decision Tree (for comparison) 
print("\n[5] Training Decision Tree …")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc  = accuracy_score(y_test, dt_pred)
print(f"      Decision Tree Accuracy : {dt_acc * 100:.2f}%")

#  Step 6 — Evaluate 
print("\n[6] Classification Report — Random Forest")
print(classification_report(y_test, rf_pred,
                             target_names=["No Disease", "Disease"]))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=2, linecolor="#0D1117",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            annot_kws={"size": 18, "fontweight": "bold", "color": "#0D1117"},
            cbar_kws={"shrink": 0.8})
ax.set_title("Confusion Matrix — Random Forest", fontsize=14, fontweight="bold",
             color="#E6EDF3", pad=15)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "screenshots", "confusion_matrix.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("screenshots/confusion_matrix.png")

#  Step 7 — Feature Importance 
importance = rf_model.feature_importances_
feat_df = (pd.DataFrame({"feature": X.columns, "importance": importance})
             .sort_values("importance", ascending=True))

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(feat_df)))
bars = ax.barh(feat_df["feature"], feat_df["importance"],
               color=colors, edgecolor="none")
ax.set_title("Feature Importance — Random Forest", fontsize=14,
             fontweight="bold", color="#E6EDF3", pad=15)
ax.set_xlabel("Importance Score", fontsize=11)
for bar, val in zip(bars, feat_df["importance"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, color="#8B949E")
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "screenshots", "feature_importance.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("      screenshots/feature_importance.png")

#  Step 8 — Model Comparison 
fig, ax = plt.subplots(figsize=(8, 5))
models     = ["Decision Tree", "Random Forest"]
accuracies = [dt_acc * 100, rf_acc * 100]
bars = ax.bar(models, accuracies, color=["#388BFD", "#3FB950"],
              width=0.4, edgecolor="none")
ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight="bold",
             color="#E6EDF3", pad=15)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, 110)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{acc:.1f}%", ha="center", fontsize=16,
            fontweight="bold", color="#E6EDF3")
ax.axhline(y=90, color="#F85149", linestyle="--", alpha=0.4, linewidth=1)
ax.text(1.25, 91, "90% threshold", fontsize=9, color="#F85149", alpha=0.7)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "screenshots", "model_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("      screenshots/model_comparison.png")

#  Step 9 — Save Model 
with open(os.path.join(BASE, "model.pkl"), "wb") as f:
    pickle.dump(rf_model, f)
with open(os.path.join(BASE, "feature_names.pkl"), "wb") as f:
    pickle.dump(list(X.columns), f)

print("\n[9] Model saved  →  model.pkl  |  feature_names.pkl")
print("\n" + "=" * 55)
print(f"  ✅  DONE  |  RF: {rf_acc*100:.2f}%  |  DT: {dt_acc*100:.2f}%")
print("=" * 55)
