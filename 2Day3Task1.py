import pandas as pd
data="""order_id,customer_id,amount
1,101,500
2,102,
3,101,700
3,101,700
4,103,300
5,104,900
"""
with open("orders.csv","w") as f:
    f.write(data)
df=pd.read_csv("orders.csv")
missing=df["amount"].isna().sum()
duplicates=df.duplicated().sum()

print(f"Missing: amount {missing} | Duplicates: {duplicates}")
df=df.drop_duplicates()
med=df["amount"].median()
print("Median used:",med)

df["amount"]=pd.to_numeric(df["amount"])
df["amount"]=df["amount"].fillna(med)
df["order_id"]=df["order_id"].astype(str)

res=df.groupby("customer_id").agg(
    total=("amount","sum"),
    orders=("order_id","count")
)
print(res)