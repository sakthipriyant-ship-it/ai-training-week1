
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, random_state=7)

inertia_values = []
silhouette_values = []
k_values = range(2, 9)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=7, n_init=10)
    labels = model.fit_predict(X)

    inertia_values.append(model.inertia_)
    silhouette_values.append(silhouette_score(X, labels))

plt.plot(k_values, inertia_values, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

plt.plot(k_values, silhouette_values, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.show()

best_k = list(k_values)[np.argmax(silhouette_values)]
best_score = max(silhouette_values)

print(f"Best k by silhouette: {best_k} (score: {best_score:.4f})")

final_model = KMeans(n_clusters=best_k, random_state=7, n_init=10)
final_labels = final_model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap="viridis", s=30)
plt.scatter(
    final_model.cluster_centers_[:, 0],
    final_model.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.legend()
plt.show()

np.random.seed(7)

mall_data = np.vstack([
    np.random.normal([25, 25], [5, 5], size=(50, 2)),
    np.random.normal([25, 75], [5, 5], size=(50, 2)),
    np.random.normal([75, 25], [5, 5], size=(50, 2)),
    np.random.normal([75, 75], [5, 5], size=(50, 2)),
    np.random.normal([50, 50], [8, 8], size=(50, 2))
])

mall_data[:, 0] = np.clip(mall_data[:, 0], 1, 100)
mall_data[:, 1] = np.clip(mall_data[:, 1], 1, 100)

mall_model = KMeans(n_clusters=5, random_state=7, n_init=10)
mall_labels = mall_model.fit_predict(mall_data)

plt.scatter(mall_data[:, 0], mall_data[:, 1], c=mall_labels, cmap="viridis", s=30)
plt.scatter(
    mall_model.cluster_centers_[:, 0],
    mall_model.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids"
)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Mall Customer Segmentation")
plt.legend()
plt.show()

print("\nMall Customer Segments")

for i, center in enumerate(mall_model.cluster_centers_):
    income, spending = center

    if income < 40 and spending < 40:
        description = "Low income, low spending customers"
    elif income < 40 and spending >= 40:
        description = "Low income, high spending customers"
    elif income >= 60 and spending < 40:
        description = "High income, low spending customers"
    elif income >= 60 and spending >= 60:
        description = "High income, high spending customers"
    else:
        description = "Average income and spending customers"

    print(f"Cluster {i}: {description}")
