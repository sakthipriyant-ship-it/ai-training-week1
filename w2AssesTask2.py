import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# Synthetic Dataset
n = 500

df = pd.DataFrame({
    "age": np.random.randint(21, 60, n),
    "income": np.random.randint(20000, 120000, n),
    "loan_amount": np.random.randint(5000, 50000, n),
    "credit_score": np.random.randint(300, 850, n),
    "employment_type": np.random.choice(["Salaried", "Self-employed", "Unemployed"], n),
    "marital_status": np.random.choice(["Single", "Married", "Divorced"], n),
    "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n),
    "application_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str)
})

df["loan_status"] = (
    (df["loan_amount"] > df["income"] * 0.4) |
    (df["credit_score"] < 550)
).astype(int)

df.loc[np.random.choice(n, 25), "income"] = np.nan
df.loc[np.random.choice(n, 20), "credit_score"] = np.nan
df.loc[np.random.choice(n, 20), "employment_type"] = np.nan

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# Feature Engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["application_date"] = pd.to_datetime(X["application_date"])
        X["application_year"] = X["application_date"].dt.year
        X["application_month"] = X["application_date"].dt.month
        X["application_day"] = X["application_date"].dt.day

        X["income_loan_ratio"] = X["income"] / (X["loan_amount"] + 1)
        X["log_loan_amount"] = np.log1p(X["loan_amount"])

        X = X.drop(columns=["application_date"])

        return X

numeric_cols = [
    "age", "income", "loan_amount", "credit_score",
    "application_year", "application_month", "application_day",
    "income_loan_ratio", "log_loan_amount"
]

categorical_cols = ["employment_type", "marital_status"]

ordinal_cols = ["education"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols),

    ("ord", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["High School", "Bachelor", "Master", "PhD"]]
        ))
    ]), ordinal_cols)
])

pipeline = Pipeline([
    ("features", FeatureEngineer()),
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# Cross Validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

print(f"Pipeline CV accuracy: {cv_scores.mean():.2f}")

# KNN Imputer
knn_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler())
    ]), numeric_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols),

    ("ord", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["High School", "Bachelor", "Master", "PhD"]]
        ))
    ]), ordinal_cols)
])

knn_pipeline = Pipeline([
    ("features", FeatureEngineer()),
    ("preprocessor", knn_preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

knn_scores = cross_val_score(knn_pipeline, X, y, cv=5, scoring="accuracy")

print(f"KNN Imputer CV accuracy: {knn_scores.mean():.2f}")

# Fit and Save Pipeline
pipeline.fit(X, y)

joblib.dump(pipeline, "loan_pipeline.pkl")

# Load Pipeline
loaded_pipeline = joblib.load("loan_pipeline.pkl")

# Predict 3 New Raw Rows
new_rows = pd.DataFrame([
    {
        "age": 25, "income": 40000, "loan_amount": 10000,
        "credit_score": 720, "employment_type": "Salaried",
        "marital_status": "Single", "education": "Bachelor",
        "application_date": "2025-01-15"
    },
    {
        "age": 35, "income": 60000, "loan_amount": 30000,
        "credit_score": 650, "employment_type": "Self-employed",
        "marital_status": "Married", "education": "Master",
        "application_date": "2025-05-20"
    },
    {
        "age": 45, "income": 80000, "loan_amount": 20000,
        "credit_score": 780, "employment_type": "Salaried",
        "marital_status": "Married", "education": "PhD",
        "application_date": "2025-09-10"
    }
])

predictions = loaded_pipeline.predict(new_rows)

print("Predictions for 3 new rows:", predictions)