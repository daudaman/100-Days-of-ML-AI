import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

os.makedirs("dataset", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

df.to_csv(
    "dataset/cancer.csv",
    index=False
)

print("Shape:", df.shape)

print("\nTarget Distribution:")
print(df["target"].value_counts())


plt.figure(figsize=(7, 5))

counts = df["target"].value_counts().sort_index()

plt.bar(
    ["Malignant", "Benign"],
    counts.values,
    color=["#F44336", "#4CAF50"]
)

plt.title("Target Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "screenshots/target_distribution.png",
    dpi=130,
    bbox_inches="tight"
)

plt.close()


feature_cols = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points"
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8)
)

eda_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness"
]

for ax, feature in zip(axes.flatten(), eda_features):

    ax.hist(
        df[df["target"] == 0][feature],
        bins=20,
        alpha=0.6,
        color="#F44336",
        label="Malignant"
    )

    ax.hist(
        df[df["target"] == 1][feature],
        bins=20,
        alpha=0.6,
        color="#4CAF50",
        label="Benign"
    )

    ax.set_title(feature)
    ax.legend(fontsize=8)

plt.suptitle(
    "Feature Distributions",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    "screenshots/feature_distributions.png",
    dpi=130,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(10, 8))

corr = df[
    feature_cols + ["target"]
].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "screenshots/correlation.png",
    dpi=130,
    bbox_inches="tight"
)

plt.close()


X = df[feature_cols]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


scores = []

for k in range(1, 21):

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    preds = model.predict(
        X_test_scaled
    )

    acc = accuracy_score(
        y_test,
        preds
    )

    scores.append(acc)


best_k = np.argmax(scores) + 1

print("\nBest K =", best_k)


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 21),
    scores,
    marker="o"
)

plt.title("K vs Accuracy")
plt.xlabel("K Value")
plt.ylabel("Accuracy")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "screenshots/accuracy_plot.png",
    dpi=130,
    bbox_inches="tight"
)

plt.close()


best_model = KNeighborsClassifier(
    n_neighbors=best_k
)

best_model.fit(
    X_train_scaled,
    y_train
)

preds = best_model.predict(
    X_test_scaled
)

acc = accuracy_score(
    y_test,
    preds
)

print("\nAccuracy:", round(acc, 4))

print(
    classification_report(
        y_test,
        preds,
        target_names=[
            "Malignant",
            "Benign"
        ]
    )
)


cm = confusion_matrix(
    y_test,
    preds
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Malignant",
        "Benign"
    ],
    yticklabels=[
        "Malignant",
        "Benign"
    ]
)

plt.title("Confusion Matrix")

plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.tight_layout()

plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=130,
    bbox_inches="tight"
)

plt.close()


with open("model.pkl", "wb") as f:
    pickle.dump(
        best_model,
        f
    )

with open("scaler.pkl", "wb") as f:
    pickle.dump(
        scaler,
        f
    )

with open("feature_cols.pkl", "wb") as f:
    pickle.dump(
        feature_cols,
        f
    )

print("\nModel saved.")
print("Done. Run app.py to start the web app.")