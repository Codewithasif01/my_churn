import pandas as pd

df = pd.read_csv("data/customers.csv")

print("Shape:", df.shape)
print(df.head())
df.info()
print(df.isna().sum())
print(df["churn"].value_counts())