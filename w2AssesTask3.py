import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score

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

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

def create_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        oob_score=True
    )
}

results = []

for name, model in models.items():
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_val, y_val)
    results.append([name, score])

    print(f"{name}: Validation Score = {score:.4f}")

    if name == "Random Forest":
        print("OOB Score:", pipeline.named_steps["model"].oob_score_)

    if name == "Logistic Regression":
        probabilities = pipeline.predict_proba(X_val.iloc[:5])
        predictions = pipeline.predict(X_val.iloc[:5])

        print("\nLogistic Regression Predict Proba:")
        print(probabilities)

        print("Predictions:")
        print(predictions)

# XGBoost with Early Stopping
xgb_pipeline = create_pipeline(
    XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20
    )
)

xgb_pipeline.named_steps["imputer"].fit(X_train)
X_train_imputed = xgb_pipeline.named_steps["imputer"].transform(X_train)
X_val_imputed = xgb_pipeline.named_steps["imputer"].transform(X_val)

xgb_pipeline.named_steps["scaler"].fit(X_train_imputed)
X_train_scaled = xgb_pipeline.named_steps["scaler"].transform(X_train_imputed)
X_val_scaled = xgb_pipeline.named_steps["scaler"].transform(X_val_imputed)

xgb_pipeline.fit(
    X_train,
    y_train,
    model__eval_set=[(X_val_scaled, y_val)],
    model__verbose=False
)

xgb_score = xgb_pipeline.score(X_val, y_val)
results.append(["XGBoost", xgb_score])

print("XGBoost Validation Score:", xgb_score)

# Results Table
results_df = pd.DataFrame(
    results,
    columns=["Model", "Validation Score"]
)

print("\nClassification Results:")
print(results_df)

# Feature Importance
rf_model = create_pipeline(
    RandomForestClassifier(n_estimators=200, random_state=42)
)

rf_model.fit(X_train, y_train)

rf_importance = pd.Series(
    rf_model.named_steps["model"].feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nRandom Forest Top 5 Features:")
print(rf_importance.head(5))

xgb_importance = pd.Series(
    xgb_pipeline.named_steps["model"].feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nXGBoost Top 5 Features:")
print(xgb_importance.head(5))

# Regression
X_reg = df[["age", "income", "credit_score"]]
y_reg = df["loan_amount"]

X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0)
}

regression_results = []
coefficients = {}

for name, model in regression_models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_reg_train, y_reg_train)

    score = pipeline.score(X_reg_val, y_reg_val)
    regression_results.append([name, score])

    coefficients[name] = pipeline.named_steps["model"].coef_

    print(f"\n{name}: R2 = {score:.4f}")

# Coefficient Comparison
coef_df = pd.DataFrame(
    coefficients,
    index=X_reg.columns
)

print("\nRegression Coefficients:")
print(coef_df)

lasso_zero_count = np.sum(
    coefficients["Lasso"] == 0
)

print("Lasso Zero Coefficients:", lasso_zero_count)

# K-Means
cluster_data = df[["age", "income", "credit_score"]]

cluster_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cluster_scaled = cluster_pipeline.fit_transform(cluster_data)

inertia_values = []
silhouette_values = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cluster_scaled)

    inertia_values.append(kmeans.inertia_)
    silhouette_values.append(silhouette_score(cluster_scaled, labels))

    print(
        f"K={k}, Inertia={kmeans.inertia_:.2f}, "
        f"Silhouette={silhouette_values[-1]:.4f}"
    )

# Elbow Plot
plt.plot(range(2, 9), inertia_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.show()

# Silhouette Plot
plt.plot(range(2, 9), silhouette_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Scores")
plt.show()

# Customer Segments
final_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = final_kmeans.fit_predict(cluster_scaled)

print("\nCustomer Segments:")
print(df.groupby("cluster")[["age", "income", "credit_score", "loan_amount"]].mean().round(2))