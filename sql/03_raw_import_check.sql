-- =============================================
-- Script: 03_raw_import_check.sql
-- Project: telco-churn-project
-- Author: Artyom K. (Junior Data Analyst)
-- Date: 2026-03-27
-- Description: Импорт сырых данных в схему raw и проверка
-- =============================================

-- Копирование данных из папки контейнера
COPY raw.wa_fn_usec
FROM '/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv'
WITH (FORMAT CSV, HEADER, DELIMITER ',', NULL ' ', ENCODING 'UTF8');

-- Общее количество загруженных строк
SELECT COUNT(*) AS total_rows FROM raw.wa_fn_usec;

-- Первые 10 строк для проверки
SELECT * FROM raw.wa_fn_usec LIMIT 10;

-- Проверка структуры таблицы
SELECT 
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_schema = 'raw' 
  AND table_name = 'wa_fn_usec'
ORDER BY ordinal_position;