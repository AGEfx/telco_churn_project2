-- =============================================
-- Script: 07_create_main_view.sql
-- Project: telco-churn-project
-- Author: Artyom K. (Junior Data Analyst)
-- Date: 2026-03-27
-- Description: Базовая аналитическая витрина для проекта
-- =============================================

CREATE OR REPLACE VIEW clean.vw_churn_analysis AS
SELECT 
    "customerID",
    "gender",
    "Partner",
    "Dependents",
    tenure,
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
    CASE 
        WHEN tenure = 0 THEN 'New customer'
        WHEN tenure <= 6 THEN '1-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 36 THEN '25-36 months'
        ELSE '37+ months'
    END AS tenure_group,
    CASE 
        WHEN "TotalCharges" = 0 THEN 'New'
        ELSE 'Existing'
    END AS customer_type,
    ROUND("MonthlyCharges" / NULLIF(tenure, 0), 2) AS avg_monthly_per_tenure_month,
    CASE WHEN "InternetService" = 'Fiber optic' THEN 1 ELSE 0 END AS fiber_optic_flag,
    CASE WHEN "PaymentMethod" = 'Electronic check' THEN 1 ELSE 0 END AS electronic_check_flag,
    CASE WHEN "Contract" = 'Month-to-month' THEN 1 ELSE 0 END AS month_to_month_flag,
    "SeniorCitizen"
FROM clean.wa_fn_usec;

select * from clean.vw_churn_analysis vca 