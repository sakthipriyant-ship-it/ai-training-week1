import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

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

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# Train / Validation / Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Decision Tree Overfitting
X_train_num = X_train.select_dtypes(include="number").fillna(0)
X_val_num = X_val.select_dtypes(include="number").fillna(0)

depths = [1, 3, 5, 10, None]
train_scores = []
val_scores = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train_num, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_num))
    val_acc = accuracy_score(y_val, model.predict(X_val_num))

    train_scores.append(train_acc)
    val_scores.append(val_acc)

    print(f"Depth {depth}: train {train_acc:.2f} val {val_acc:.2f}")

plt.plot(["1", "3", "5", "10", "None"], train_scores, marker="o", label="Train")
plt.plot(["1", "3", "5", "10", "None"], val_scores, marker="o", label="Validation")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree - Overfitting")
plt.legend()
plt.show()

# Logistic Regression Cross Validation
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

cv_scores = cross_val_score(
    logistic_model, X, y, cv=5, scoring="accuracy"
)

print(f"CV: {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}")

# Data Leakage - Incorrect Scaling
numeric_X = X.select_dtypes(include="number").fillna(0)

scaler_wrong = StandardScaler()
X_scaled_wrong = scaler_wrong.fit_transform(numeric_X)

print("\nIncorrect Scaling: Fit on entire dataset")

# Correct Scaling
scaler_correct = StandardScaler()

X_train_scaled = scaler_correct.fit_transform(
    X_train.select_dtypes(include="number").fillna(0)
)

X_val_scaled = scaler_correct.transform(
    X_val.select_dtypes(include="number").fillna(0)
)

X_test_scaled = scaler_correct.transform(
    X_test.select_dtypes(include="number").fillna(0)
)

print("Correct Scaling: Fit on training data only")

# Leakage Explanation (4 Lines)
print("\nData Leakage:")
print("1. Incorrect scaling uses the entire dataset.")
print("2. Validation and test data influence the scaler.")
print("3. Correct scaling fits only on training data.")
print("4. Validation and test data are transformed without fitting.")

# Supervised vs Unsupervised
print("\nProblem Classification:")
print("Spam Detection: Supervised")
print("Customer Segmentation: Unsupervised")
print("House Price Prediction: Supervised")
print("Anomaly Detection: Unsupervised")

# ML Workflow (7 Steps)
print("\nML Workflow:")
print("1. Collect Data")
print("2. Clean Data")
print("3. Split Data")
print("4. Feature Engineering")
print("5. Train Model")
print("6. Evaluate Model")
print("7. Deploy Model")