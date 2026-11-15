import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("data/customers.csv")

# 1) X = inputs, y = jo predict karna hai
X = df.drop(columns=["customer_id", "churn"])
y = (df["churn"] == "Yes").astype(int)

# 2) Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train churn rate:", round(y_train.mean(), 3))
print("Test churn rate:", round(y_test.mean(), 3))

# 3) Columns ke do group
numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "num_support_tickets"]
categorical_cols = ["gender", "contract_type", "internet_service",
                    "online_security", "payment_method", "paperless_billing"]

# 4) Har group ke liye "recipe"
numeric_pipe = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

# 5) Train se seekho (fit), phir dono par lagao (transform)
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)
print("Columns pehle:", X_train.shape[1], "-> baad mein:", X_train_ready.shape[1])