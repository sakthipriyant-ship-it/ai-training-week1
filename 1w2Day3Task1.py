import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

df = pd.DataFrame({
    "age": [25, None, 40, 35, None],
    "income": [30000, 45000, None, 80000, 52000],
    "city": ["Chennai", "Mumbai", None, "Chennai", "Delhi"],
    "size": ["S", "L", "M", "S", "L"]
})

print("Original DataFrame")
print(df)

age_imputer = SimpleImputer(strategy="median")
df["age"] = age_imputer.fit_transform(df[["age"]])

print("\nAfter Age Imputation")
print(df)

city_imputer = SimpleImputer(strategy="most_frequent")
df["city"] = city_imputer.fit_transform(df[["city"]]).ravel()

print("\nAfter City Imputation")
print(df)

knn_imputer = KNNImputer(n_neighbors=2)
knn_data = pd.DataFrame(
    knn_imputer.fit_transform(df[["age", "income"]]),
    columns=["age", "income"]
)

print("\nKNN Imputation")
print(knn_data)

onehot_encoder = OneHotEncoder(sparse_output=False)
city_encoded = onehot_encoder.fit_transform(df[["city"]])

city_df = pd.DataFrame(
    city_encoded,
    columns=onehot_encoder.get_feature_names_out(["city"])
)

df = pd.concat([df.drop(columns=["city"]), city_df], axis=1)

print("\nAfter City OneHot Encoding")
print(df)

ordinal_encoder = OrdinalEncoder(categories=[["S", "M", "L"]])
df["size"] = ordinal_encoder.fit_transform(df[["size"]]).ravel()

print("\nAfter Size Ordinal Encoding")
print(df)

label_encoder = LabelEncoder()

target = ["Yes", "No", "Yes", "No", "Yes"]
target_encoded = label_encoder.fit_transform(target)

print("\nLabel Encoded Target")
print(target_encoded)

print("\nFinal Encoded DataFrame")
print(df)