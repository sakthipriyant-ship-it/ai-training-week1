import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, mean_absolute_error,
    mean_squared_error, r2_score
)

from xgboost import XGBClassifier

np.random.seed(42)

n = 500

df = pd.DataFrame({
    "age": np.random.randint(21, 60, n),
    "income": np.random.randint(20000, 120000, n),
    "loan_amount": np.random.randint(5000, 50000, n),
    "credit_score": np.random.randint(300, 850, n),
    "employment_type": np.random.choice(
        ["Salaried", "Self-employed", "Unemployed"], n
    ),
    "marital_status": np.random.choice(
        ["Single", "Married", "Divorced"], n
    ),
    "education": np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"], n
    ),
    "application_date": pd.date_range(
        "2023-01-01", periods=n, freq="D"
    ).astype(str)
})

df["loan_status"] = (
    (df["loan_amount"] > df["income"] * 0.4) |
    (df["credit_score"] < 550)
).astype(int)

df.loc[np.random.choice(n, 25), "income"] = np.nan
df.loc[np.random.choice(n, 20), "credit_score"] = np.nan
df.loc[np.random.choice(n, 20), "employment_type"] = np.nan

# Classification
X = df[["age", "income", "loan_amount", "credit_score"]]
y = df["loan_status"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

def create_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

# GridSearchCV - Random Forest
rf_pipeline = create_pipeline(
    RandomForestClassifier(random_state=42)
)

rf_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_leaf": [1, 2]
}

rf_search = GridSearchCV(
    rf_pipeline,
    rf_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

print("Random Forest Best Parameters:")
print(rf_search.best_params_)
print("Random Forest Best CV Score:", rf_search.best_score_)

# RandomizedSearchCV - XGBoost
xgb_pipeline = create_pipeline(
    XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
)

xgb_params = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [3, 5, 7, 10],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb_pipeline,
    xgb_params,
    n_iter=15,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)

print("\nXGBoost Best Parameters:")
print(xgb_search.best_params_)
print("XGBoost Best CV Score:", xgb_search.best_score_)

# Classifiers
models = {
    "Logistic Regression": create_pipeline(
        LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": create_pipeline(
        DecisionTreeClassifier(max_depth=5, random_state=42)
    ),
    "Random Forest": rf_search.best_estimator_,
    "XGBoost": xgb_search.best_estimator_
}

results = []
roc_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)
    auc = roc_auc_score(y_val, probabilities)

    results.append([name, accuracy, precision, recall, f1, auc])

    print(f"\n{name}")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    roc_models[name] = probabilities

# Results Table
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
)

print("\nResults Table:")
print(results_df.round(4))

# ROC Curves
plt.figure(figsize=(8, 6))

for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
    probabilities = roc_models[name]
    fpr, tpr, _ = roc_curve(y_val, probabilities)
    auc = roc_auc_score(y_val, probabilities)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# Decision Threshold - At Least 75% Recall
best_model_name = results_df.sort_values(
    ["F1", "ROC-AUC"], ascending=False
).iloc[0]["Model"]

best_model = models[best_model_name]
best_probabilities = best_model.predict_proba(X_val)[:, 1]

precision_values, recall_values, thresholds = precision_recall_curve(
    y_val, best_probabilities
)

valid_indices = np.where(recall_values[:-1] >= 0.75)[0]

if len(valid_indices) > 0:
    best_threshold_index = valid_indices[
        np.argmax(precision_values[valid_indices])
    ]

    threshold = thresholds[best_threshold_index]
    threshold_predictions = (best_probabilities >= threshold).astype(int)

    print("\nDecision Threshold")
    print("Best Model:", best_model_name)
    print("Threshold:", threshold)
    print("Recall:", recall_score(y_val, threshold_predictions))
    print("Precision:", precision_score(y_val, threshold_predictions))
else:
    print("\nNo threshold achieved 75% recall.")

# Regression - Loan Amount
X_reg = df[["age", "income", "credit_score"]]
y_reg = df["loan_amount"]

X_reg_train, X_reg_temp, y_reg_train, y_reg_temp = train_test_split(
    X_reg, y_reg, test_size=0.40, random_state=42
)

X_reg_val, X_reg_test, y_reg_val, y_reg_test = train_test_split(
    X_reg_temp, y_reg_temp, test_size=0.50, random_state=42
)

regression_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

regression_model.fit(X_reg_train, y_reg_train)

reg_predictions = regression_model.predict(X_reg_val)

reg_mae = mean_absolute_error(y_reg_val, reg_predictions)
reg_rmse = np.sqrt(mean_squared_error(y_reg_val, reg_predictions))
reg_r2 = r2_score(y_reg_val, reg_predictions)

# Mean Baseline
baseline_predictions = np.full(
    len(y_reg_val), y_reg_train.mean()
)

baseline_mae = mean_absolute_error(
    y_reg_val, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_reg_val, baseline_predictions)
)

baseline_r2 = r2_score(
    y_reg_val, baseline_predictions
)

print("\nRegression Comparison")
print("Model MAE:", reg_mae)
print("Model RMSE:", reg_rmse)
print("Model R2:", reg_r2)

print("\nMean Baseline")
print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)
print("Baseline R2:", baseline_r2)

# Final Test Evaluation - Best Classifier Only
final_X = pd.concat([X_train, X_val])
final_y = pd.concat([y_train, y_val])

final_model = models[best_model_name]
final_model.fit(final_X, final_y)

test_predictions = final_model.predict(X_test)
test_probabilities = final_model.predict_proba(X_test)[:, 1]

print("\nFinal Test Evaluation")
print("Best Model:", best_model_name)
print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Precision:", precision_score(y_test, test_predictions, zero_division=0))
print("Recall:", recall_score(y_test, test_predictions, zero_division=0))
print("F1:", f1_score(y_test, test_predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, test_probabilities))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))