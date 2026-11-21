import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

baseline_pred = np.full(len(y_test), y_train.mean())

results["Baseline (Mean)"] = [
    mean_absolute_error(y_test, baseline_pred),
    np.sqrt(mean_squared_error(y_test, baseline_pred)),
    r2_score(y_test, baseline_pred)
]

predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    results[name] = [
        mean_absolute_error(y_test, y_pred),
        np.sqrt(mean_squared_error(y_test, y_pred)),
        r2_score(y_test, y_pred)
    ]

metrics_table = pd.DataFrame(
    results,
    index=["MAE", "RMSE", "R2"]
).T

print("Regression Metrics:")
print(metrics_table)

plt.figure(figsize=(6, 5))
plt.scatter(y_test, predictions["Linear Regression"], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Predicted vs Actual")
plt.grid()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(y_test, predictions["Random Forest"], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest: Predicted vs Actual")
plt.grid()
plt.show()

for name, y_pred in predictions.items():
    residuals = y_test - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"{name}: Residual Plot")
    plt.grid()
    plt.show()