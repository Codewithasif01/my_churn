import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/customers.csv")
df["churn_flag"] = (df["churn"] == "Yes").astype(int)

rate = df.groupby("contract_type")["churn_flag"].mean() * 100
print(rate)

rate.plot(kind="bar")
plt.ylabel("Churn rate (%)")
plt.title("Churn rate by contract type")
plt.tight_layout()
plt.show()