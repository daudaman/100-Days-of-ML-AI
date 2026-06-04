import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ── load data ─────────────────────────────────────────────────────────────────

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map({
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
})

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["species_name"].value_counts())
print("\nFirst few rows:")
print(df.head())

# save the dataset so someone can inspect it later
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/iris.csv", index=False)
print("\nDataset saved to dataset/iris.csv")


# ── quick EDA ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Iris Dataset – Feature Distributions", fontsize=15, y=1.01)

colors = {"Iris-setosa": "#4CAF50", "Iris-versicolor": "#2196F3", "Iris-virginica": "#FF5722"}

features = iris.feature_names
for ax, feature in zip(axes.flatten(), features):
    for species, color in colors.items():
        subset = df[df["species_name"] == species]
        ax.hist(subset[feature], alpha=0.65, label=species, color=color, bins=15)
    ax.set_title(feature)
    ax.set_xlabel("cm")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

plt.tight_layout()
os.makedirs("screenshots", exist_ok=True)
plt.savefig("screenshots/feature_distributions.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: screenshots/feature_distributions.png")


# pairplot is the classic go-to for iris — keeps it here
pair_df = df[iris.feature_names + ["species_name"]].copy()
pair_df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in pair_df.columns]

fig2, axes2 = plt.subplots(4, 4, figsize=(14, 14))
feature_cols = [c for c in pair_df.columns if c != "species_name"]

for i, feat_y in enumerate(feature_cols):
    for j, feat_x in enumerate(feature_cols):
        ax = axes2[i][j]
        if i == j:
            for sp, color in colors.items():
                subset = pair_df[pair_df["species_name"] == sp]
                ax.hist(subset[feat_x], color=color, alpha=0.6, bins=12)
        else:
            for sp, color in colors.items():
                subset = pair_df[pair_df["species_name"] == sp]
                ax.scatter(subset[feat_x], subset[feat_y], c=color, alpha=0.6, s=20, label=sp)
        if i == len(feature_cols) - 1:
            ax.set_xlabel(feat_x, fontsize=7)
        if j == 0:
            ax.set_ylabel(feat_y, fontsize=7)
        ax.tick_params(labelsize=6)

# add a legend manually on the last subplot
handles = [plt.Line2D([0], [0], marker='o', color='w',
           markerfacecolor=c, markersize=8, label=sp)
           for sp, c in colors.items()]
axes2[0][-1].legend(handles=handles, loc="upper right", fontsize=7)

fig2.suptitle("Iris – Pairplot", fontsize=15)
plt.tight_layout()
plt.savefig("screenshots/pairplot.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: screenshots/pairplot.png")


# ── split & scale ──────────────────────────────────────────────────────────────

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ── model ──────────────────────────────────────────────────────────────────────

# RandomForest works well here without much tuning.
# n_estimators=100 is usually enough; more trees don't help much on this dataset.
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# cross-val to get a honest accuracy estimate
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\n5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ── evaluation ────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig3, ax3 = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=ax3)
ax3.set_title("Confusion Matrix – Test Set")
ax3.set_ylabel("Actual")
ax3.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: screenshots/confusion_matrix.png")


# feature importance
importances = model.feature_importances_
feat_names  = [f.replace(" (cm)", "") for f in iris.feature_names]
sorted_idx  = np.argsort(importances)[::-1]

fig4, ax4 = plt.subplots(figsize=(7, 4))
bars = ax4.bar(
    [feat_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color=["#4CAF50", "#2196F3", "#FF5722", "#9C27B0"]
)
ax4.set_title("Feature Importance – Random Forest")
ax4.set_ylabel("Importance")
plt.tight_layout()
plt.savefig("screenshots/feature_importance.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: screenshots/feature_importance.png")


# ── save model & scaler ───────────────────────────────────────────────────────

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel saved to model.pkl")
print("Scaler saved to scaler.pkl")
print("\nDone. Run app.py to start the web app.")
