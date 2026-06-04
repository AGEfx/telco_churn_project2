# Telco Customer Churn Prediction

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Docker](https://img.shields.io/badge/Docker-2CA5E0?logo=docker&logoColor=white)](https://hub.docker.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Used-success)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretable-blue)](https://shap.readthedocs.io/)
![Built with Streamlit](https://img.shields.io/badge/-Built%20with%20Streamlit-orange?logo=streamlit&logoColor=white)

**Прогнозирование оттока клиентов телекоммуникационной компании с помощью машинного обучения и интерпретируемого AI.**

---

## 🎯 О проекте

Этот проект представляет собой **полноценный end-to-end ML-кейс**: от разведочного анализа данных до построения высокоинтерпретируемой модели прогнозирования оттока клиентов телеком-оператора.

**Основная цель** — не только построить точную модель, но и понять *почему* клиенты уходят, чтобы предложить конкретные рекомендации по удержанию.

### Ключевые результаты

- **ROC-AUC** на тесте: **0.8549**
- **F1-score**: **0.6508** (после тюнинга порога вероятности)
- **Recall**: **0.8021** — модель хорошо ловит уходящих клиентов
- Лучшая модель: **XGBoost Classifier** + Bayesian Optimization
- Глубокий **SHAP-анализ** для интерпретируемости

**Топ-3 самых важных признака (по SHAP):**
1. **Contract** (тип контракта)
2. **avg_monthly_per_tenure_month** (средний чек за месяц лояльности)
3. **high_risk_combination** (комбинированный риск-фактор)

---

## 📊 Основные insights из EDA

- Уровень оттока — **26.54%** (1869 клиентов из 7043)
- Более половины клиентов (55%) используют помесячный контракт — именно они уходят чаще всего
- Средний срок пользования услугами — 32.4 месяца
- Клиенты с высокими ежемесячными платежами и отсутствием услуг безопасности/поддержки — в зоне повышенного риска

**Бизнес-выводы и рекомендации** находятся в презентации (`Telco Customer Churn Analysis presentation.pdf`).

---

## 🛠 Технологический стек
 
- **Python и Machine Learning**: Python 3.8+, Pandas, NumPy, SciPy
- **Моделирование**: XGBoost, Scikit-learn
- **Оптимизация**: Bayesian Optimization (`bayes_opt`)
- **Интерпретируемость**: SHAP (TreeExplainer)
- **Визуализация**: Plotly Express, Matplotlib
- **Дополнительно**:  
  JSON, Joblib — сохранение моделей  
  Docker — контейнеризация и воспроизводимость
      
---

## 📁 Структура проекта

```text
telco-churn-project/
├── data/                  # raw + processed данные
├── src/                   # 01_data_loading → 04_churn_modeling
├── dashboards/            # Интерактивные отчёты и визуализации
├── models/                # Сохранённые модели (joblib)
├── utils/                 # Вспомогательные скрипты
├── Dockerfile
├── requirements.txt
└── README.md
```
---

## 🚀 Как запустить проект

### Требования

- Python 3.11+  **или** Docker Desktop
- Git

### Способ 1 — Python (без Docker)

#### 1. Клонируйте репозиторий

```bash
git clone https://github.com/tema_kiselevv/telco_churn_project2.git
cd telco_churn_project2
```

#### 2. Скачайте датасет и положите файл в `data/raw` 
Ссылка: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data  
Файл должен находиться по пути:  
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

#### 3. Создайте и активируйте виртуальное окружение

Рекомендуемая версия python для данного проекта - python 3.11.9

```console
py -3.11 -m venv venv       # вместо "3.11" напишите вашу версию python
```
```console
# Linux/macOS
source venv/bin/activate
```
```console
# Windows
venv\Scripts\activate
```

#### 4. Установите зависимости

```bash
pip install -r requirements.txt
```

#### 5. Запустите дашборд

```bash
streamlit run app.py
```

Открой в браузере: [http://localhost:8501](http://localhost:8501)

#### Остановка

Нажми **Ctrl + C** в терминале.

### Способ 2 — Docker

#### 1. Клонируйте репозиторий

```bash
git clone https://github.com/tema_kiselevv/telco_churn_project2.git
cd telco_churn_project2
```

#### 2. Соберите образ

```bash
docker build -t telco_churn_project2 .
```

#### 3. Запустите контейнер

```bash
docker run -p 8501:8501 telco_churn_project2
```

Открой в браузере: [http://localhost:8501](http://localhost:8501)

#### Остановка

Нажми **Ctrl + C** в терминале.

---

## 📈 Методология

* Предобработка данных (SQL + Pandas)
* Разведочный анализ (EDA) + статистические тесты
* Feature Engineering (включая сильные комбинированные признаки)
* Валидация: StratifiedKFold + Train/Test split
* Моделирование → XGBoost с Bayesian Optimization
* Тюнинг порога для максимизации F1-score
* Интерпретация с помощью SHAP (глобальная + локальная)

---

📌 Ключевые особенности проекта

* Воспроизводимость через Docker
* Реализованный фронтенд
* Высокая интерпретируемость модели (SHAP)
* Бизнес-ориентированные insights и рекомендации
* Чистая, модульная структура проекта

---

## Дополнительные материалы

* Презентация — Telco Customer Churn Analysis presentation.pdf  
* Jupyter notebooks — подробный разбор каждого этапа
* SQL-скрипты — создание схем, очистка и аналитические витрины

**Хотите увидеть проект в действии?  
Открывайте ноутбуки или презентацию — там вся ценность.  
⭐ Если проект был полезен — ставьте звезду!**  

## Автор  
Киселев Артём — Junior Data Analyst  
GitHub: Tema Kiselev (temakiselevv) | Telegram: @tema_kiselev

*Проект создан как демонстрация сильных навыков в data analysis, end-to-end ML, feature engineering и интерпретируемом машинном обучении.*
