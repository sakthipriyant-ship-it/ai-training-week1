from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Supervised learning: Logistic Regression
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("F1 (macro):", round(f1_score(y_test, y_pred, average="macro"), 2))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Unsupervised learning: KMeans uses features only, not species labels
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
print("\nKMeans cluster sizes:", cluster_sizes.tolist())

# Compare clusters with the actual species
comparison = pd.crosstab(
    pd.Series(cluster_labels, name="KMeans cluster"),
    pd.Series(iris.target_names[y], name="Actual species")
)
#print("\nClusters compared with actual species:")
#print(comparison)