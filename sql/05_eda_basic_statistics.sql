-- =============================================
-- Script: 05_eda_basic_statistics.sql
-- Project: telco-churn-project
-- Author: Artyom K. (Junior Data Analyst)
-- Date: 2026-03-27
-- Description: Базовый Exploratory Data Analysis (EDA)
-- =============================================

-- Общая информация о таблице
SELECT 
    COUNT(*) AS total_customers,
    COUNT(CASE WHEN "Churn" = 'Yes' THEN 1 END) AS total_churned,
    ROUND(COUNT(CASE WHEN "Churn" = 'Yes' THEN 1 END)::NUMERIC / COUNT(*) * 100, 2) AS churn_rate_percent,
    MIN(tenure) AS min_tenure,
    MAX(tenure) AS max_tenure,
    ROUND(AVG(tenure), 2) AS avg_tenure
FROM clean.wa_fn_usec;

-- Основные финансовые метрики
SELECT 
    ROUND(AVG("MonthlyCharges"), 2) AS avg_monthly_charges,
    ROUND(MIN("MonthlyCharges"), 2) AS min_monthly_charges,
    ROUND(MAX("MonthlyCharges"), 2) AS max_monthly_charges,
    ROUND(AVG("TotalCharges"), 2) AS avg_total_charges,
    ROUND(SUM("TotalCharges"), 2) AS total_revenue
FROM clean.wa_fn_usec;

-- Сравнение клиентов, которые ушли и остались
SELECT 
    "Churn",
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    ROUND(AVG("MonthlyCharges"), 2) AS avg_monthly_charges,
    ROUND(AVG("TotalCharges"), 2) AS avg_total_charges,
    ROUND(AVG(tenure), 2) AS avg_tenure_months,
    ROUND(SUM("TotalCharges"), 2) AS total_revenue
FROM clean.wa_fn_usec
GROUP BY "Churn";

-- Проверка на NULL
SELECT 
    COUNT(*) AS total_rows,
    COUNT(CASE WHEN "MonthlyCharges" IS NULL THEN 1 END) AS null_monthlycharges,
    COUNT(CASE WHEN "TotalCharges" IS NULL THEN 1 END) AS null_totalcharges,
    COUNT(CASE WHEN tenure IS NULL THEN 1 END) AS null_tenure,
    COUNT(CASE WHEN "gender" IS NULL THEN 1 END) AS null_gender,
    COUNT(CASE WHEN "SeniorCitizen" IS NULL THEN 1 END) AS null_seniorcitizen,
    COUNT(CASE WHEN "Partner" IS NULL THEN 1 END) AS null_partner,
    COUNT(CASE WHEN "Dependents" IS NULL THEN 1 END) AS null_dependents,
    COUNT(CASE WHEN "PhoneService" IS NULL THEN 1 END) AS null_phoneservice,
    COUNT(CASE WHEN "MultipleLines" IS NULL THEN 1 END) AS null_multiplelines,
    COUNT(CASE WHEN "InternetService" IS NULL THEN 1 END) AS null_internetservice,
    COUNT(CASE WHEN "OnlineSecurity" IS NULL THEN 1 END) AS null_onlinesecurity,
    COUNT(CASE WHEN "OnlineBackup" IS NULL THEN 1 END) AS null_onlinebackup,
    COUNT(CASE WHEN "DeviceProtection" IS NULL THEN 1 END) AS null_deviceprotection,
    COUNT(CASE WHEN "TechSupport" IS NULL THEN 1 END) AS null_techsupport,
    COUNT(CASE WHEN "StreamingTV" IS NULL THEN 1 END) AS null_streamingtv,
    COUNT(CASE WHEN "StreamingMovies" IS NULL THEN 1 END) AS null_streamingmovies,
    COUNT(CASE WHEN "Contract" IS NULL THEN 1 END) AS null_contract,
    COUNT(CASE WHEN "PaperlessBilling" IS NULL THEN 1 END) AS null_paperlessbilling,
    COUNT(CASE WHEN "PaymentMethod" IS NULL THEN 1 END) AS null_paymentmethod,
    COUNT(CASE WHEN "Churn" IS NULL THEN 1 END) AS null_churn
FROM clean.wa_fn_usec;