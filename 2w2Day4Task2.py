import time
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(random_state=42)

grid_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 20],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    rf, grid_params, cv=5, scoring="f1", n_jobs=-1
)

start = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start

print("Grid Search")
print("Best Parameters:", grid_search.best_params_)
print("Best F1:", grid_search.best_score_)
print("Time:", round(grid_time, 2), "seconds")

grid_results = pd.DataFrame(grid_search.cv_results_)
print(grid_results[["params", "mean_test_score", "std_test_score", "rank_test_score"]])

random_params = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_leaf": [1, 2, 4, 6, 8, 10]
}

random_search = RandomizedSearchCV(
    rf, random_params, n_iter=20, cv=5, scoring="f1",
    random_state=42, n_jobs=-1
)

start = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - start

print("\nRandomized Search")
print("Best Parameters:", random_search.best_params_)
print("Best F1:", random_search.best_score_)
print("Time:", round(random_time, 2), "seconds")

random_results = pd.DataFrame(random_search.cv_results_)
print(random_results[["params", "mean_test_score", "std_test_score", "rank_test_score"]])

print("\nComparison")
print(f"Grid Search: Best F1 = {grid_search.best_score_:.4f}, Time = {grid_time:.2f} seconds")
print(f"Randomized Search: Best F1 = {random_search.best_score_:.4f}, Time = {random_time:.2f} seconds")

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nTest F1:", f1_score(y_test, y_pred))