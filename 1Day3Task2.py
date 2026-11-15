import pandas as pd
data="""id,name,department,salary,city
1,Arun,Sales,30000,Chennai
2,Priya,HR,45000,Chennai
3,Rahul,IT,60000,Bangalore
4,Divya,IT,75000,Chennai
5,Karthik,Finance,90000,Mumbai
6,Meena,Sales,52000,Bangalore
7,Suresh,IT,68000,Chennai
8,Anitha,HR,48000,Mumbai
"""
with open("employees.csv","w") as f:
    f.write(data)
df=pd.read_csv("employees.csv")
chennai = df.loc[df["city"]=="Chennai","name"]
print("Chennai:",", ".join(chennai))
hs=df[df["salary"]>50000]
print("Salary above 50000:",len(hs),"employees")
it=df[df["department"]=="IT"]
print("IT department:",", ".join(it["name"]))