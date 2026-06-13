import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


os.makedirs("dataset", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)


def generate_dataset(n=2000):
    np.random.seed(42)

    radius = np.random.uniform(6, 28, n)
    texture = np.random.uniform(9, 40, n)
    perimeter = radius * 6.5 + np.random.normal(0, 5, n)
    area = np.pi * radius**2 * 0.3 + np.random.normal(0, 20, n)
    smoothness = np.random.uniform(0.05, 0.16, n)
    compactness = np.random.uniform(0.02, 0.35, n)
    concavity = np.random.uniform(0.0, 0.43, n)
    symmetry = np.random.uniform(0.1, 0.3, n)
    fractal_dim = np.random.uniform(0.05, 0.1, n)

    malignant_score = (
        radius * 0.15
        + texture * 0.05
        + compactness * 3.0
        + concavity * 4.0
        + smoothness * 8.0
        + np.random.normal(0, 0.5, n)
    )

    threshold = np.percentile(malignant_score, 63)
    label = (malignant_score >= threshold).astype(int)

    df = pd.DataFrame({
        "radius": radius.round(2),
        "texture": texture.round(2),
        "perimeter": perimeter.round(2),
        "area": area.round(2),
        "smoothness": smoothness.round(4),
        "compactness": compactness.round(4),
        "concavity": concavity.round(4),
        "symmetry": symmetry.round(4),
        "fractal_dimension": fractal_dim.round(5),
        "diagnosis": label
    })

    return df


if not os.path.exists("dataset/cancer.csv"):
    df = generate_dataset()
    df.to_csv("dataset/cancer.csv", index=False)
    print("Dataset created.")

df = pd.read_csv("dataset/cancer.csv")

print("Shape:", df.shape)
print("\nClass distribution:")
print(df["diagnosis"].value_counts())

feature_cols = ["radius", "texture", "perimeter", "area", "smoothness",
                "compactness", "concavity", "symmetry", "fractal_dimension"]

X = df[feature_cols]
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTrying different SVM kernels...")
for kernel in ["linear", "rbf", "poly"]:
    svm = SVC(kernel=kernel, random_state=42)
    scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
    print(f"  {kernel}: {scores.mean():.4f} +/- {scores.std():.4f}")

model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)
acc = accuracy_score(y_test, preds)
print(f"\nTest Accuracy (RBF kernel): {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["Benign", "Malignant"]))


plt.figure(figsize=(7, 5))
counts = df["diagnosis"].value_counts()
plt.bar(["Benign", "Malignant"], counts.values, color=["#4CAF50", "#F44336"])
plt.title("Tumor Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("screenshots/distribution.png", dpi=130, bbox_inches="tight")
plt.close()

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix - SVM (RBF)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig("screenshots/correlation.png", dpi=130, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, feature in zip(axes.flatten(), ["radius", "concavity", "compactness", "texture"]):
    ax.hist(df[df["diagnosis"] == 0][feature], alpha=0.6, label="Benign", color="#4CAF50", bins=25)
    ax.hist(df[df["diagnosis"] == 1][feature], alpha=0.6, label="Malignant", color="#F44336", bins=25)
    ax.set_title(feature)
    ax.legend(fontsize=8)
plt.suptitle("Feature Distributions by Diagnosis", fontsize=13)
plt.tight_layout()
plt.savefig("screenshots/feature_distributions.png", dpi=130, bbox_inches="tight")
plt.close()

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("\nModel saved. Run app.py to start the web app.")
