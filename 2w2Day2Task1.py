import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

learning_rates = [0.3, 0.1, 0.03]
n_estimators = [100, 300, 1000]

results = []
best_model = None
best_accuracy = 0

for lr, trees in zip(learning_rates, n_estimators):
    model = XGBClassifier(
        learning_rate=lr,
        n_estimators=trees,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Learning Rate: {lr}")
    print(f"Best Iteration: {model.best_iteration}")

    results.append({
        "learning_rate": lr,
        "best_iteration": model.best_iteration,
        "test_accuracy": accuracy
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

results_df = pd.DataFrame(results)

print("\nComparison Table")
print(results_df.to_string(index=False))

importance = pd.Series(
    best_model.feature_importances_,
    index=data.feature_names
).sort_values(ascending=False)

print("\nFeature Importance")
print(importance)

importance.head(10).sort_values().plot(kind="barh")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()