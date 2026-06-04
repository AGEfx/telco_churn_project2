-- =============================================
-- Script: 04_clean_data.sql
-- Project: telco-churn-project
-- Author: Artyom K. (Junior Data Analyst)
-- Date: 2026-03-27
-- Description: Очистка данных
-- =============================================

-- Проверка raw-данных на NULL-значения
SELECT *
FROM raw.wa_fn_usec
WHERE NOT (wa_fn_usec IS NOT NULL);

-- Проверка raw-данных на дубликаты
SELECT "customerID", COUNT(*) AS cnt
FROM raw.wa_fn_usec
GROUP BY "customerID"
HAVING COUNT(*) > 1;

-- Удаление существующей таблицы clean.wa_fn_usec
DROP TABLE IF EXISTS clean.wa_fn_usec;

-- Создание clean-таблицы и дублирование данных из raw
CREATE TABLE clean.wa_fn_usec AS
SELECT *
FROM raw.wa_fn_usec;

-- Обновление NULL-значений 
UPDATE clean.wa_fn_usec
SET "TotalCharges" = COALESCE("TotalCharges", 0);

-- Основная информация после предобработки
SELECT 
    COUNT(*) AS total_rows_in_clean,
    MIN(tenure) AS min_tenure,
    MAX(tenure) AS max_tenure
FROM clean.wa_fn_usec;

-- Вывод первых 20 строк
SELECT * FROM clean.wa_fn_usec LIMIT 20;