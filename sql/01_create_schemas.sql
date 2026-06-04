-- =============================================
-- Script: 01_create_schemas.sql
-- Project: telco-churn-project
-- Author: Artyom K. (Junior Data Analyst)
-- Date: 2026-03-27
-- Description: Создание схем для хранения данных
-- =============================================

-- Создание схем raw и clean
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;

-- Проверка статусов схем
SELECT 
    schema_name,
    'exists' as status
FROM information_schema.schemata 
WHERE schema_name IN ('raw', 'clean');