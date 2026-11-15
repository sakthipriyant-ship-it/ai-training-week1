import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("employees.csv")

#Bar chart
avg=df.groupby("department")["salary"].mean()
bar=plt.bar(avg.index,avg.values)
plt.title("Average Salary per Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
for b in bar:
    plt.text(
        b.get_x()+b.get_width()/2,
        b.get_height(),
        round(b.get_height(),2),
        ha="center",
        va="bottom"
    )
plt.show()


#Histogram

plt.hist(df["salary"],bins=5)

plt.title("Hist Salary")
plt.xlabel("Salary")
plt.ylabel("Bin frequency")

plt.show()

#Scatter plot
for department in df["department"].unique():
    data=df[df["department"]==department]

    plt.scatter(
        data["id"],
        data["salary"],
        label=department
        )

    plt.title("Id vs Salary")
    plt.xlabel("ID")
    plt.ylabel("Salary")

plt.legend()
plt.show()

#Line Chart

months=["Jan","Feb","Mar","Apr","May","Jun"]
sales=[12,15,14,18,22,25]

plt.plot(months,sales,marker="*")

plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("sale")

plt.show()