-- =============================================
-- Script: 02_raw_create_table.sql
-- Project: telco-churn-project
-- Author: Artyom K. (Junior Data Analyst)
-- Date: 2026-03-27
-- Description: Создание таблицы для исходных данных
-- =============================================

-- Удаление существующей таблицы raw.wa_fn_usec
DROP TABLE IF EXISTS raw.wa_fn_usec;

-- Создание raw-таблицы перед импортом данных
CREATE TABLE raw.wa_fn_usec (
    "customerID"        VARCHAR(50),
    "gender"            VARCHAR(50),
    "SeniorCitizen"     INTEGER,
    "Partner"           VARCHAR(50),
    "Dependents"        VARCHAR(50),
    "tenure"            INTEGER,
    "PhoneService"      VARCHAR(50),
    "MultipleLines"     VARCHAR(50),
    "InternetService"   VARCHAR(50),
    "OnlineSecurity"    VARCHAR(50),
    "OnlineBackup"      VARCHAR(50),
    "DeviceProtection"  VARCHAR(50),
    "TechSupport"       VARCHAR(50),
    "StreamingTV"       VARCHAR(50),
    "StreamingMovies"   VARCHAR(50),
    "Contract"          VARCHAR(50),
    "PaperlessBilling"  VARCHAR(50),
    "PaymentMethod"     VARCHAR(50),
    "MonthlyCharges"    NUMERIC(10,2),
    "TotalCharges"      NUMERIC(12,2),
    "Churn"             VARCHAR(50)
);

COMMENT ON TABLE raw.wa_fn_usec IS 'Raw data as imported from CSV';