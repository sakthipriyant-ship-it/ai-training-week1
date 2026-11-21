import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    weights=[0.95, 0.05],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42)
}

results = {}
probabilities = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    probabilities[name] = proba
    predictions = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, predictions)
    results[name] = [auc, f1]

    print(name, "AUC:", auc)

plt.figure(figsize=(8, 6))

for name, proba in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))

for name, proba in probabilities.items():
    precision, recall, _ = precision_recall_curve(y_test, proba)
    plt.plot(recall, precision, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

metrics_table = pd.DataFrame(
    results,
    index=["AUC", "F1"]
).T

print("\nMetrics Table:")
print(metrics_table)

def best_threshold(y_true, proba, min_recall=0.80):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    valid = np.where(recall[:-1] >= min_recall)[0]

    if len(valid) == 0:
        return None, None

    best_idx = valid[np.argmax(precision[valid])]
    return thresholds[best_idx], precision[best_idx]

print("\nThreshold with at least 80% recall:")

for name, proba in probabilities.items():
    threshold, precision = best_threshold(y_test, proba, 0.80)

    if threshold is not None:
        predictions = (proba >= threshold).astype(int)
        print(name)
        print("Threshold:", threshold)
        print("Recall:", recall_score(y_test, predictions))
        print("Precision:", precision_score(y_test, predictions))