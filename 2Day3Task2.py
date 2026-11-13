import pandas as pd

data="""customer_id,name,city
101, Arun, Chennai
102, Priya, Mumbai
103, Rahul, Chennai
105, Divya, Bangalore
"""
with open("customers.csv","w") as f:
    f.write(data)

cdf=pd.read_csv("customers.csv")

odf=pd.read_csv("orders.csv")
odf=odf.drop_duplicates()
odf["amount"]=odf["amount"].fillna(odf["amount"].median()).astype(int)
odf["order_id"]=odf["order_id"].astype(str)

ij=pd.merge(odf,cdf,on="customer_id",how="inner")
print(f"Inner join rows: {len(ij)} (customer 104 dropped)")

lj=pd.merge(cdf,odf,on="customer_id",how="left")
print("No orders:",", ".join(lj.loc[lj["order_id"].isna(),"name"]))

oj=pd.merge(odf,cdf,on="customer_id",how="outer")
for i in oj.loc[oj["name"].isna(),"customer_id"]:
    print("Order without customer: customer_id",i)