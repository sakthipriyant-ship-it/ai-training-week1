import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May"]
sales = [10000, 15000, 12000, 18000, 22000]

plt.bar(months, sales)

plt.title("Monthly Sales Report")
plt.xlabel("Months")
plt.ylabel("Sales")

plt.show()
