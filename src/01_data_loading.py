import pandas as pd
import numpy as np


df = pd.read_csv('../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"Данные успешно загружены")
print(f"Размер датасета: {df.shape[0]:,} строк и {df.shape[1]} колонок")
print(df.info())

yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
df[yes_no_cols] = df[yes_no_cols].replace({'Yes': 1, 'No': 0}).astype('int8')
categorical_cols = df.select_dtypes(include='string').columns
df[categorical_cols] = df[categorical_cols].astype('category')
df['SeniorCitizen'] = df['SeniorCitizen'].astype('int8')
str_cols = 'customerID'
df[str_cols] = df[str_cols].astype('string')
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["tenure_group"] = np.select(
    [
        df["tenure"] == 0,
        df["tenure"] <= 6,
        df["tenure"] <= 12,
        df["tenure"] <= 24,
        df["tenure"] <= 36,
    ],
    [
        "New customer",
        "1-6 months",
        "7-12 months",
        "13-24 months",
        "25-36 months",
    ],
    default="37+ months"
)

# customer type
df["customer_type"] = np.where(df["TotalCharges"] == 0, "New", "Existing")

# avg monthly per tenure month
df["avg_monthly_per_tenure_month"] = (
    df["MonthlyCharges"] / df["tenure"].replace(0, np.nan)
).round(2)

# flags
df["fiber_optic_flag"] = (df["InternetService"] == "Fiber optic").astype(int)
df["electronic_check_flag"] = (df["PaymentMethod"] == "Electronic check").astype(int)
df["month_to_month_flag"] = (df["Contract"] == "Month-to-month").astype(int)

df.to_csv('../data/processed/telco_churn.csv', index=False)
df.to_parquet('../data/processed/telco_churn.parquet', index=False)