import joblib
import pandas as pd

model = joblib.load("churn_model.joblib")

new_customers = pd.DataFrame([
    {   # naya customer, month-to-month, fiber, electronic check
        "age": 29, "gender": "Male", "tenure_months": 3,
        "contract_type": "Month-to-month", "internet_service": "Fiber optic",
        "online_security": "No", "payment_method": "Electronic check",
        "paperless_billing": "Yes", "monthly_charges": 95.5,
        "total_charges": 286.5, "num_support_tickets": 4,
    },
    {   # purana customer, 2 saal ka contract
        "age": 54, "gender": "Female", "tenure_months": 60,
        "contract_type": "Two year", "internet_service": "DSL",
        "online_security": "Yes", "payment_method": "Credit card",
        "paperless_billing": "No", "monthly_charges": 48.0,
        "total_charges": 2880.0, "num_support_tickets": 0,
    },
])

proba = model.predict_proba(new_customers)[:, 1]

for i, p in enumerate(proba, start=1):
    risk = "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.4 else "LOW"
    print(f"Customer {i}: churn probability = {p:.1%} -> {risk}")