
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

single_tree_accuracy = accuracy_score(y_test, single_tree.predict(X_test))

rf_model = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

print(f"Single tree test accuracy: {single_tree_accuracy:.2f}")
print(f"Random forest test accuracy: {rf_accuracy:.2f}")
print(f"OOB score: {rf_model.oob_score_:.2f}")

n_estimators_values = [10, 50, 100, 300]
accuracies = []

for n in n_estimators_values:
    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(accuracy)

plt.plot(n_estimators_values, accuracies, marker="o")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Test Accuracy")
plt.title("Random Forest Accuracy")
plt.show()

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=data.feature_names
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind="bar")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
