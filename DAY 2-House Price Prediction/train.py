import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import os

os.makedirs("dataset", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

if not os.path.exists("dataset/housing.csv"):
    import numpy as np
    np.random.seed(42)
    n = 2000

    bedrooms = np.random.randint(1, 7, n)
    bathrooms = np.clip(bedrooms - np.random.randint(0, 2, n), 1, 5)
    sqft = np.random.randint(500, 5000, n)
    age = np.random.randint(0, 60, n)
    garage = np.random.randint(0, 4, n)
    floors = np.random.randint(1, 4, n)
    location_score = np.random.uniform(1, 10, n)
    distance_city = np.random.uniform(1, 50, n)

    price = (
        sqft * 120
        + bedrooms * 8000
        + bathrooms * 5000
        + garage * 7000
        - age * 600
        + location_score * 12000
        - distance_city * 800
        + floors * 4000
        + np.random.normal(0, 18000, n)
    )
    price = np.clip(price, 50000, 1200000)

    df = pd.DataFrame({
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft,
        "age_years": age,
        "garage_spaces": garage,
        "floors": floors,
        "location_score": location_score.round(1),
        "distance_to_city_km": distance_city.round(1),
        "price": price.round(0).astype(int)
    })
    df.to_csv("dataset/housing.csv", index=False)
    print("Dataset created.")

df = pd.read_csv("dataset/housing.csv")

print("Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nBasic stats:")
print(df.describe().round(2))


fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=40, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[i].set_title(col, fontsize=10)

plt.suptitle("Feature Distributions", fontsize=13)
plt.tight_layout()
os.makedirs("screenshots", exist_ok=True)
plt.savefig("screenshots/distributions.png", dpi=130, bbox_inches="tight")
plt.close()


plt.figure(figsize=(9, 7))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("screenshots/correlation.png", dpi=130, bbox_inches="tight")
plt.close()


feature_cols = ["bedrooms", "bathrooms", "sqft_living", "age_years",
                "garage_spaces", "floors", "location_score", "distance_to_city_km"]

X = df[feature_cols]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n{name}")
    print(f"  MAE:  ${mae:,.0f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  R2:   {r2:.4f}")


best_model = models["Gradient Boosting"]
best_preds = best_model.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_preds, alpha=0.3, color="#FF5722", s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", linewidth=1)
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted - Gradient Boosting")
plt.tight_layout()
plt.savefig("screenshots/actual_vs_predicted.png", dpi=130, bbox_inches="tight")
plt.close()


feat_importance = best_model.feature_importances_
sorted_idx = np.argsort(feat_importance)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(
    [feature_cols[i] for i in sorted_idx],
    feat_importance[sorted_idx],
    color="#4CAF50"
)
plt.title("Feature Importance - Gradient Boosting")
plt.ylabel("Importance")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("screenshots/feature_importance.png", dpi=130, bbox_inches="tight")
plt.close()


with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("\nModel saved to model.pkl")
print("Done. Run app.py to start the web app.")
