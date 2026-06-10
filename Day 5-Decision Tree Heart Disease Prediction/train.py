import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.makedirs("screenshots", exist_ok=True)

df = pd.read_csv("dataset/heart.csv")
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

#  EDA 
# Target Distribution
plt.figure(figsize=(6, 4))
colors = ['#2ecc71', '#e74c3c']
counts = df['target'].value_counts().sort_index()
bars = plt.bar(['No Disease', 'Heart Disease'], counts.values,
               color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.title('Target Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('screenshots/target_distribution.png', dpi=150)
plt.close()
print(" target_distribution.png saved")

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt='.2f',
            linewidths=0.5, annot_kws={'size': 7})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('screenshots/correlation_heatmap.png', dpi=150)
plt.close()
print(" correlation_heatmap.png saved")

#  Features / Labels 
X = df.drop("target", axis=1)
y = df["target"]

#  Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Model 
model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Also test entropy
model_e = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
model_e.fit(X_train, y_train)
print(f"Gini accuracy   : {accuracy_score(y_test, model.predict(X_test)):.4f}")
print(f"Entropy accuracy: {accuracy_score(y_test, model_e.predict(X_test)):.4f}")

#  Predictions & Evaluation ─
pred = model.predict(X_test)
acc  = accuracy_score(y_test, pred)
print(f"\nFinal Accuracy: {acc:.4f}")
print(classification_report(y_test, pred))

# Confusion Matrix
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease'])
plt.title(f'Confusion Matrix  |  Accuracy: {acc:.1%}', fontsize=13, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('screenshots/confusion_matrix.png', dpi=150)
plt.close()
print(" confusion_matrix.png saved")

#  Feature Importance 
importance = model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values('Importance')
plt.figure(figsize=(8, 6))
colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_df)))
plt.barh(feat_df['Feature'], feat_df['Importance'], color=colors_bar)
plt.title('Feature Importance - Decision Tree', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('screenshots/feature_importance.png', dpi=150)
plt.close()
print(" feature_importance.png saved")

#  Tree Visualization 
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=list(X.columns),
          class_names=['No Disease', 'Heart Disease'],
          filled=True, rounded=True, fontsize=8)
plt.title('Decision Tree — Heart Disease Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('screenshots/decision_tree.png', dpi=120)
plt.close()
print(" decision_tree.png saved")

#  Save Model 
pickle.dump(model,          open('model.pkl',        'wb'))
pickle.dump(list(X.columns), open('feature_names.pkl', 'wb'))
print("\n model.pkl and feature_names.pkl saved")
print(f"\n Final Accuracy: {acc:.2%}")
