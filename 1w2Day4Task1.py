from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    weights=[0.95, 0.05],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_majority = [0] * len(y_test)

print("Majority-class model:")
print("Accuracy:", accuracy_score(y_test, y_majority))
print("Precision:", precision_score(y_test, y_majority, zero_division=0))
print("Recall:", recall_score(y_test, y_majority, zero_division=0))
print("F1:", f1_score(y_test, y_majority, zero_division=0))

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nLogistic Regression:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

balanced_model = LogisticRegression(class_weight="balanced")
balanced_model.fit(X_train, y_train)
y_balanced = balanced_model.predict(X_test)

print("\nLogistic Regression with balanced weights:")
print("Accuracy:", accuracy_score(y_test, y_balanced))
print("Precision:", precision_score(y_test, y_balanced, zero_division=0))
print("Recall:", recall_score(y_test, y_balanced, zero_division=0))
print("F1:", f1_score(y_test, y_balanced, zero_division=0))