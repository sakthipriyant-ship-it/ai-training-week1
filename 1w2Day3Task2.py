import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.DataFrame({
    "age": [25, 30, 40, 35, 50],
    "income": [30000, 45000, 60000, 80000, 100000]
})

standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)

print("StandardScaler")
print(pd.DataFrame(standard_scaled, columns=df.columns))
print("Mean:", standard_scaled.mean(axis=0))
print("Standard Deviation:", standard_scaled.std(axis=0))

minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(df)

print("\nMinMaxScaler")
print(pd.DataFrame(minmax_scaled, columns=df.columns))
print("Minimum:", minmax_scaled.min(axis=0))
print("Maximum:", minmax_scaled.max(axis=0))

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_accuracy = accuracy_score(y_test, knn.predict(X_test))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier()
knn_scaled.fit(X_train_scaled, y_train)
knn_scaled_accuracy = accuracy_score(y_test, knn_scaled.predict(X_test_scaled))

print("\nKNN Accuracy")
print(f"KNN without scaling: {knn_accuracy:.2f}")
print(f"KNN with scaling: {knn_scaled_accuracy:.2f}")

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_accuracy = accuracy_score(y_test, tree.predict(X_test))

tree_scaled = DecisionTreeClassifier(random_state=42)
tree_scaled.fit(X_train_scaled, y_train)
tree_scaled_accuracy = accuracy_score(y_test, tree_scaled.predict(X_test_scaled))

print("\nDecision Tree Accuracy")
print(f"Without scaling: {tree_accuracy:.2f}")
print(f"With scaling: {tree_scaled_accuracy:.2f}")