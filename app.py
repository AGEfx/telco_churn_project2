"""
Telco Churn Analytics — Streamlit Dashboard
Запуск: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from pathlib import Path
from glob import glob

# НАСТРОЙКА
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Тёмная тема через CSS
st.markdown("""
<style>
    .main { background-color: #0a0f1e; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 3px solid #6366f1;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background: #1e293b;
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# СТИЛЬ ГРАФИКОВ
CHART_STYLE = dict(
    template="plotly_dark",
    paper_bgcolor="#1e293b",
    plot_bgcolor="#1e293b",
    font=dict(color="#ffffff"),
    title_font=dict(color="#ffffff"),
    legend=dict(font=dict(color="#ffffff")),
    xaxis=dict(
        tickfont=dict(color="#ffffff"),
        title=dict(font=dict(color="#ffffff")),  
    ),
    yaxis=dict(
        tickfont=dict(color="#ffffff"),
        title=dict(font=dict(color="#ffffff")),  
    ),
)

# ПУТИ
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# ЗАГРУЗКА ДАННЫХ
@st.cache_data
def load_data():
    path = DATA_DIR / "data_featured.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        # Fallback: ищем telco_churn.parquet
        path2 = DATA_DIR / "telco_churn.parquet"
        if path2.exists():
            df = pd.read_parquet(path2)
        else:
            st.error("❌ Файл данных не найден. Ожидается: data/processed/data_featured.parquet")
            st.stop()

    # Нормализуем колонку Churn
    if "Churn" in df.columns:
        if df["Churn"].dtype == object:
            df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
        else:
            df["Churn_bin"] = df["Churn"].astype(int)
            df["Churn"] = df["Churn_bin"].map({1: "Yes", 0: "No"})

    return df


@st.cache_data
def load_model_infos():
    """Загружает все JSON с метриками моделей."""
    infos = []
    for path in sorted(glob(str(MODEL_DIR / "model_info_*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
                data["_file"] = Path(path).name
                infos.append(data)
        except Exception:
            pass
    return infos


@st.cache_resource
def load_best_model():
    pkl_path = MODEL_DIR / "xgboost_churn_model_20260420.pkl"
    if pkl_path.exists():
        return joblib.load(pkl_path)
    # Берём последний best
    pkls = sorted(glob(str(MODEL_DIR / "xgboost_churn_best_*.pkl")))
    if pkls:
        return joblib.load(pkls[-1])
    return None


def load_best_params():
    path = MODEL_DIR / "best_params.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ЗАГРУЖАЕМ
df = load_data()
model_infos = load_model_infos()
model = load_best_model()
best_params = load_best_params()

churn_col = "Churn_bin" if "Churn_bin" in df.columns else "Churn"
total = len(df)
churned = int(df[churn_col].sum()) if churn_col == "Churn_bin" else (df["Churn"] == "Yes").sum()
retained = total - churned
churn_rate = churned / total * 100

# САЙДБАР
with st.sidebar:
    st.markdown("## 📡 Telco Churn")
    st.markdown("---")

    st.markdown("### 🔢 Данные")
    st.metric("Клиентов", f"{total:,}")
    st.metric("Churn Rate", f"{churn_rate:.2f}%")
    st.metric("Ушли", f"{churned:,}")
    st.metric("Остались", f"{retained:,}")

    st.markdown("---")
    st.markdown("### 👾 Модель")
    st.metric("Статус", "Загружена ✅" if model else "Не найдена ❌")
    st.metric("Версий", str(len(model_infos)))

    if best_params:
        st.markdown("### ⚙️ Best Params")
        for k, v in list(best_params.items())[:5]:
            st.text(f"{k}: {round(float(v), 4) if isinstance(v, (int, float)) else v}")

    st.markdown("---")
    st.caption("Telco Churn Analytics · 2026")

# ЗАГОЛОВОК
st.markdown("# 📡 Telco Churn Analytics Dashboard")
st.markdown("Прогнозирование оттока клиентов · XGBoost · Bayesian Optimization · SHAP")
st.markdown("---")

# ВКЛАДКИ
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Обзор",
    "🔍 EDA",
    "👾 Модель",
    "🔮 SHAP / Важность",
    "🎯 Предсказание",
])


# TAB 1 — ОБЗОР
with tab1:
    st.subheader("Ключевые метрики")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Всего клиентов", f"{total:,}")
    c2.metric("📉 Churn Rate", f"{churn_rate:.2f}%", delta=f"-{retained:,} остались", delta_color="inverse")
    if "MonthlyCharges" in df.columns:
        c3.metric("💳 Ср. платёж/мес", f"${df['MonthlyCharges'].mean():.2f}")
    if "tenure" in df.columns:
        c4.metric("📅 Ср. tenure", f"{df['tenure'].mean():.1f} мес")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Pie churn
        fig = px.pie(
            values=[retained, churned],
            names=["Остались", "Ушли"],
            color_discrete_sequence=["#10b981", "#ef4444"],
            title="Соотношение Churn / Retained",
            hole=0.4,
        )
        fig.update_layout(**CHART_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Churn по типу контракта
        if "Contract" in df.columns:
            grp = df.groupby("Contract").apply(
                lambda x: pd.Series({
                    "total": len(x),
                    "churned": (x["Churn"] == "Yes").sum() if "Churn" in x.columns else x[churn_col].sum(),
                })
            ).reset_index()
            grp["rate"] = grp["churned"] / grp["total"] * 100

            fig2 = px.bar(
                grp, x="Contract", y="rate",
                color="rate",
                color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                title="Churn Rate по типу контракта (%)",
                labels={"rate": "Churn Rate %"},
                text=grp["rate"].round(1).astype(str) + "%",
            )
            fig2.update_layout(**CHART_STYLE, coloraxis_showscale=False)
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

    # Таблица топ-клиентов с риском
    st.markdown("### 📋 Данные (первые 100 строк)")
    display_cols = [c for c in ["customerID", "Contract", "tenure", "MonthlyCharges", "TotalCharges", "Churn"] if c in df.columns]
    st.dataframe(df[display_cols].head(100), use_container_width=True)



# TAB 2 — EDA
with tab2:
    st.subheader("Исследовательский анализ данных")

    eda_choice = st.radio(
        "Выбери раздел анализа:",
        ["Tenure", "Платежи", "Контракт и оплата", "Услуги", "Демография", "Корреляции"],
        horizontal=True,
    )

    churn_label = "Churn" if "Churn" in df.columns else churn_col

    if eda_choice == "Tenure" and "tenure" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df, x="tenure", color=churn_label,
                nbins=72, barmode="overlay",
                color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
                title="Распределение tenure по Churn",
                labels={"tenure": "Месяцы"},
                opacity=0.75,
            )
            fig.update_layout(**CHART_STYLE)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Churn rate по группам tenure
            bins = [0, 6, 12, 24, 36, 100]
            labels = ["1-6 мес", "7-12 мес", "13-24 мес", "25-36 мес", "37+ мес"]
            df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)
            grp = df.groupby("tenure_group", observed=True).apply(
                lambda x: (x["Churn"] == "Yes").mean() * 100 if "Churn" in x.columns else x[churn_col].mean() * 100
            ).reset_index(name="churn_rate")
            fig2 = px.bar(
                grp, x="tenure_group", y="churn_rate",
                color="churn_rate", color_continuous_scale=["#10b981", "#ef4444"],
                title="Churn Rate по группам tenure (%)",
                labels={"churn_rate": "Churn Rate %", "tenure_group": "Группа"},
                text=grp["churn_rate"].round(1).astype(str) + "%",
            )
            fig2.update_layout(**CHART_STYLE, coloraxis_showscale=False)
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

    elif eda_choice == "Платежи" and "MonthlyCharges" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                df, x=churn_label, y="MonthlyCharges",
                color=churn_label,
                color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
                title="MonthlyCharges: Churn vs Retained",
            )
            fig.update_layout(**CHART_STYLE)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "TotalCharges" in df.columns:
                fig2 = px.scatter(
                    df.sample(min(2000, len(df))),
                    x="tenure", y="MonthlyCharges",
                    color=churn_label,
                    color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
                    title="tenure vs MonthlyCharges",
                    opacity=0.6,
                )
                fig2.update_layout(**CHART_STYLE)
                st.plotly_chart(fig2, use_container_width=True)

    elif eda_choice == "Контракт и оплата":
        cats = [c for c in ["Contract", "PaymentMethod", "InternetService"] if c in df.columns]
        for cat in cats:
            grp = df.groupby(cat).apply(
                lambda x: pd.Series({
                    "Churn Rate %": (x["Churn"] == "Yes").mean() * 100 if "Churn" in x.columns else x[churn_col].mean() * 100,
                    "Кол-во": len(x),
                })
            ).reset_index()
            fig = px.bar(
                grp, x=cat, y="Churn Rate %",
                color="Churn Rate %",
                color_continuous_scale=["#10b981", "#ef4444"],
                title=f"Churn Rate по {cat}",
                text=grp["Churn Rate %"].round(1).astype(str) + "%",
            )
            fig.update_layout(**CHART_STYLE, coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    elif eda_choice == "Услуги":
        service_cols = [c for c in [
            "OnlineSecurity", "TechSupport", "OnlineBackup",
            "DeviceProtection", "StreamingTV", "StreamingMovies"
        ] if c in df.columns]

        rows = []
        for col in service_cols:
            for val in df[col].dropna().unique():
                subset = df[df[col] == val]
                rate = (subset["Churn"] == "Yes").mean() * 100 if "Churn" in df.columns else subset[churn_col].mean() * 100
                rows.append({"Услуга": col, "Значение": str(val), "Churn Rate %": round(rate, 1)})

        svc_df = pd.DataFrame(rows)
        fig = px.bar(
            svc_df, x="Услуга", y="Churn Rate %", color="Значение",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Churn Rate по наличию услуг",
        )
        fig.update_layout(**CHART_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_choice == "Демография":
        demo_cols = [c for c in ["gender", "SeniorCitizen", "Partner", "Dependents"] if c in df.columns]
        cols = st.columns(len(demo_cols))
        for i, col in enumerate(demo_cols):
            grp = df.groupby(col).apply(
                lambda x: (x["Churn"] == "Yes").mean() * 100 if "Churn" in x.columns else x[churn_col].mean() * 100
            ).reset_index(name="Churn Rate %")
            fig = px.bar(grp, x=col, y="Churn Rate %", title=f"Churn по {col}",
                         color="Churn Rate %", color_continuous_scale=["#10b981", "#ef4444"],
                         text=grp["Churn Rate %"].round(1).astype(str) + "%")
            fig.update_layout(**CHART_STYLE,
                               coloraxis_showscale=False, height=300)
            fig.update_traces(textposition="outside")
            cols[i].plotly_chart(fig, use_container_width=True)

    elif eda_choice == "Корреляции":
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Корреляционная матрица числовых признаков",
                aspect="auto",
            )
            fig.update_layout(**CHART_STYLE,
                               height=600)
            st.plotly_chart(fig, use_container_width=True)


# TAB 3 — МОДЕЛЬ
with tab3:
    st.subheader("👾 Результаты моделирования XGBoost")

    if model_infos:
        # Берём последний запуск
        latest = model_infos[-1]

        # Пробуем достать метрики
        metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        found_metrics = {k: latest.get(k) for k in metrics_keys if k in latest}

        # Возможно вложены в подсловарь
        if not found_metrics:
            for sub in ["metrics", "test_metrics", "results", "scores"]:
                if sub in latest and isinstance(latest[sub], dict):
                    found_metrics = {k: latest[sub].get(k) for k in metrics_keys if k in latest[sub]}
                    break

        if found_metrics:
            cols = st.columns(len(found_metrics))
            labels = {"accuracy": "Accuracy", "precision": "Precision",
                      "recall": "Recall", "f1": "F1-Score", "roc_auc": "ROC-AUC"}
            for i, (k, v) in enumerate(found_metrics.items()):
                if v is not None:
                    cols[i].metric(labels.get(k, k), f"{float(v):.3f}")

        st.markdown("---")

        # График эволюции метрик по версиям
        if len(model_infos) > 1:
            rows = []
            for info in model_infos:
                row = {"файл": info.get("_file", "?")}
                for k in metrics_keys:
                    v = info.get(k)
                    if v is None:
                        for sub in ["metrics", "test_metrics", "results", "scores"]:
                            if sub in info and isinstance(info[sub], dict):
                                v = info[sub].get(k)
                                break
                    row[k] = float(v) if v is not None else None
                rows.append(row)

            evol_df = pd.DataFrame(rows).dropna(subset=["roc_auc", "f1"], how="all")
            if not evol_df.empty:
                fig = go.Figure()
                for metric, color in [("roc_auc", "#6366f1"), ("f1", "#10b981"),
                                       ("precision", "#f59e0b"), ("recall", "#ef4444")]:
                    if metric in evol_df.columns and evol_df[metric].notna().any():
                        fig.add_trace(go.Scatter(
                            x=evol_df["файл"], y=evol_df[metric],
                            name=metric.upper(), line=dict(color=color, width=2),
                            mode="lines+markers",
                        ))
                fig.update_layout(
                    **CHART_STYLE,
                    title=dict(text="Эволюция метрик по версиям модели", font=dict(color="#ffffff")),
                    xaxis_tickangle=-45, height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📄 Полный JSON последней модели")
        with st.expander("Показать / скрыть"):
            st.json(latest)

    else:
        st.info("JSON-файлы с метриками моделей не найдены в папке `models/`.")

    # Best params
    if best_params:
        st.markdown("### ⚙️ Лучшие гиперпараметры (Bayesian Optimization)")
        params_df = pd.DataFrame([
            {"Параметр": k, "Значение": round(float(v), 5) if isinstance(v, (int, float)) else v}
            for k, v in best_params.items()
        ])
        st.dataframe(params_df, use_container_width=True)



# TAB 4 — SHAP / ВАЖНОСТЬ ПРИЗНАКОВ
with tab4:
    st.subheader("🔮 Важность признаков")

    if model is not None:
        # Feature importance из модели
        try:
            fi = model.feature_importances_
            feat_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else [f"f{i}" for i in range(len(fi))]

            fi_df = pd.DataFrame({"feature": feat_names, "importance": fi})
            fi_df = fi_df.sort_values("importance", ascending=True).tail(20)

            fig = px.bar(
                fi_df, x="importance", y="feature", orientation="h",
                title="Топ-20 признаков по важности (XGBoost feature_importances_)",
                color="importance", color_continuous_scale="Viridis",
                labels={"importance": "Важность", "feature": "Признак"},
            )
            fig.update_layout(
                **CHART_STYLE,
                height=600, coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)


            # SHAP
            st.markdown("### 🧠 SHAP Analysis")
            try:
                import shap

                if hasattr(model, "feature_names_in_"):
                    known_features = list(model.feature_names_in_)
                    available = [c for c in known_features if c in df.columns]
                    missing = [c for c in known_features if c not in df.columns]

                    X_sample = df[available].dropna().sample(min(200, len(df)), random_state=42).copy()

                    for col in missing:
                        X_sample[col] = 0

                    X_sample = X_sample[known_features]

                    booster = model.get_booster()
                    feature_types = booster.feature_types

                    # Категории точно как в обучающих данных
                    cat_categories = {
                        "MultipleLines":      ["No", "No phone service", "Yes"],
                        "InternetService":    ["DSL", "Fiber optic", "No"],
                        "OnlineSecurity":     ["No", "No internet service", "Yes"],
                        "OnlineBackup":       ["No", "No internet service", "Yes"],
                        "DeviceProtection":   ["No", "No internet service", "Yes"],
                        "TechSupport":        ["No", "No internet service", "Yes"],
                        "StreamingTV":        ["No", "No internet service", "Yes"],
                        "StreamingMovies":    ["No", "No internet service", "Yes"],
                        "Contract":           ["Month-to-month", "One year", "Two year"],
                        "PaymentMethod":      ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
                        "tenure_group":       ["1-6 мес", "7-12 мес", "13-24 мес", "25-36 мес", "37+ мес"],
                        "customer_type":      sorted(df["customer_type"].dropna().unique().tolist()) if "customer_type" in df.columns else [],
                    }

                    for i, col in enumerate(X_sample.columns):
                        col_type = feature_types[i] if feature_types else None
                        if col_type == "c":
                            cats = cat_categories.get(col)
                            if cats:
                                X_sample[col] = pd.Categorical(
                                    X_sample[col].astype(str),
                                    categories=cats
                                )
                            else:
                                # Если категории неизвестны — берём из df
                                unique_cats = sorted(df[col].dropna().astype(str).unique().tolist())
                                X_sample[col] = pd.Categorical(
                                    X_sample[col].astype(str),
                                    categories=unique_cats
                                )
                        else:
                            X_sample[col] = pd.to_numeric(X_sample[col], errors="coerce").fillna(0)

                if not X_sample.empty:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)

                    if isinstance(shap_values, list):
                        vals = shap_values[1]
                    elif shap_values.ndim == 3:
                        vals = shap_values[:, :, 1]
                    else:
                        vals = shap_values

                    shap_importance = np.abs(vals).mean(axis=0)
                    shap_df = pd.DataFrame({
                        "feature": X_sample.columns,
                        "shap_importance": shap_importance,
                    }).sort_values("shap_importance", ascending=True).tail(15)

                    fig2 = px.bar(
                        shap_df, x="shap_importance", y="feature", orientation="h",
                        title="SHAP Mean |value| — влияние признаков на предсказание",
                        color="shap_importance", color_continuous_scale="Plasma",
                        labels={"shap_importance": "Mean |SHAP|", "feature": "Признак"},
                    )
                    fig2.update_layout(**CHART_STYLE, height=500, coloraxis_showscale=False)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Нет подходящих признаков для SHAP.")
            except ImportError:
                st.info("💡 Установи `shap` для SHAP-анализа: `pip install shap`")
            except Exception as e:
                st.warning(f"SHAP не удалось выполнить: {e}")

        except AttributeError:
            st.warning("Модель не имеет атрибута feature_importances_.")
    else:
        st.error("Модель не загружена. Убедись, что в папке `models/` есть `.pkl` файл.")


# TAB 5 — ПРЕДСКАЗАНИЕ
# TAB 5 — ПРЕДСКАЗАНИЕ
with tab5:
    st.subheader("🎯 Предсказание оттока для нового клиента")

    if model is None:
        st.error("Модель не найдена. Предсказание недоступно.")
    else:
        # Создаем две под-вкладки
        tab_single, tab_batch = st.tabs(["👤 Индивидуальный скоринг", "📁 Пакетный скоринг"])

        with tab_single:
            st.markdown("Заполни параметры клиента и получи прогноз вероятности оттока.")

            col1, col2, col3 = st.columns(3)
    
            with col1:
                tenure = st.slider("Tenure (месяцы)", 0, 72, 12)
                monthly = st.slider("MonthlyCharges ($)", 20.0, 120.0, 65.0, step=0.5)
                total_charges = st.slider("TotalCharges ($)", 0.0, 9000.0, float(monthly * tenure), step=10.0)
    
            with col2:
                contract = st.selectbox("Тип контракта", ["Month-to-month", "One year", "Two year"])
                internet = st.selectbox("Интернет", ["Fiber optic", "DSL", "No"])
                payment = st.selectbox("Способ оплаты", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
            with col3:
                senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Да" if x else "Нет")
                partner = st.selectbox("Partner", ["Yes", "No"])
                security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
                tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
    
            st.markdown("---")
    
            if st.button("🔮 Предсказать", type="primary", use_container_width=True):
    
                # Категории точно как в обучающих данных
                cat_categories = {
                    "MultipleLines":      ["No", "No phone service", "Yes"],
                    "InternetService":    ["DSL", "Fiber optic", "No"],
                    "OnlineSecurity":     ["No", "No internet service", "Yes"],
                    "OnlineBackup":       ["No", "No internet service", "Yes"],
                    "DeviceProtection":   ["No", "No internet service", "Yes"],
                    "TechSupport":        ["No", "No internet service", "Yes"],
                    "StreamingTV":        ["No", "No internet service", "Yes"],
                    "StreamingMovies":    ["No", "No internet service", "Yes"],
                    "Contract":           ["Month-to-month", "One year", "Two year"],
                    "PaymentMethod":      ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
                    "tenure_group":       ["1-6 мес", "7-12 мес", "13-24 мес", "25-36 мес", "37+ мес"],
                    "customer_type":      sorted(df["customer_type"].dropna().unique().tolist()) if "customer_type" in df.columns else [],
                }
    
                # Собираем строку с нулями по всем признакам модели
                input_row = {f: 0 for f in model.feature_names_in_}
    
                # Заполняем числовые признаки из формы
                input_row["tenure"] = tenure
                input_row["MonthlyCharges"] = monthly
                input_row["TotalCharges"] = total_charges
                input_row["SeniorCitizen"] = senior
                input_row["fiber_optic_flag"] = 1 if internet == "Fiber optic" else 0
                input_row["electronic_check_flag"] = 1 if payment == "Electronic check" else 0
                input_row["month_to_month_flag"] = 1 if contract == "Month-to-month" else 0
    
                # Категориальные признаки
                if "Contract" in input_row:
                    input_row["Contract"] = contract
                if "InternetService" in input_row:
                    input_row["InternetService"] = internet
                if "PaymentMethod" in input_row:
                    input_row["PaymentMethod"] = payment
                if "OnlineSecurity" in input_row:
                    input_row["OnlineSecurity"] = "No" if security == "No" else "Yes"
                if "TechSupport" in input_row:
                    input_row["TechSupport"] = "No" if tech_support == "No" else "Yes"
                if "SeniorCitizen" in input_row:
                    input_row["SeniorCitizen"] = senior
                if "Partner" in input_row:
                    input_row["Partner"] = partner
        
                # Строим DataFrame
                X_pred = pd.DataFrame([input_row])[list(model.feature_names_in_)]
    
                booster = model.get_booster()
                feature_types = booster.feature_types
    
                for i, col in enumerate(X_pred.columns):
                    col_type = feature_types[i] if feature_types else None
                    if col_type == "c":
                        cats = cat_categories.get(col)
                        if cats:
                            X_pred[col] = pd.Categorical(
                                X_pred[col].astype(str),
                                categories=cats
                            )
                        else:
                            unique_cats = sorted(df[col].dropna().astype(str).unique().tolist())
                            X_pred[col] = pd.Categorical(
                                X_pred[col].astype(str),
                                categories=unique_cats
                            )
                    else:
                        X_pred[col] = pd.to_numeric(X_pred[col], errors="coerce").fillna(0)
    
                try:
                    proba = model.predict_proba(X_pred)[0][1]
                    pred = int(proba >= 0.4)
    
                    risk_color = "#ef4444" if proba >= 0.5 else "#f59e0b" if proba >= 0.3 else "#10b981"
                    risk_label = "🔴 Высокий риск" if proba >= 0.5 else "🟡 Средний риск" if proba >= 0.3 else "🟢 Низкий риск"
    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("Вероятность оттока", f"{proba:.1%}")
                    col_r2.metric("Предсказание", "Уйдёт ❌" if pred else "Останется ✅")
                    col_r3.metric("Уровень риска", risk_label)
    
                    st.progress(float(proba), text=f"Churn probability: {proba:.1%}")
    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=proba * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Вероятность оттока (%)", "font": {"color": "#f1f5f9"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                            "bar": {"color": risk_color},
                            "bgcolor": "#1e293b",
                            "steps": [
                                {"range": [0, 30],  "color": "rgba(16, 185, 129, 0.2)"},   # зелёный
                                {"range": [30, 50], "color": "rgba(245, 158, 11, 0.2)"},   # жёлтый
                                {"range": [50, 100],"color": "rgba(239, 68, 68, 0.2)"},    # красный
                            ],
                            "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": 40},
                        },
                        number={"suffix": "%", "font": {"color": risk_color}},
                    ))
                    gauge_style = {k: v for k, v in CHART_STYLE.items()  if k not in ("title", "title_font")}
    
                    fig.update_layout(
                        **gauge_style,
                        height=300,
                        margin=dict(t=60, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
                except Exception as e:
                    st.error(f"Ошибка предсказания: {e}")

        # (здесь заканчивается код индивидуального скоринга Артёма)

        with tab_batch:
            st.subheader("Массовая оценка базы клиентов")
            st.write("Загрузите датасет в формате `.csv` для получения предсказаний по всем абонентам сразу.")
            
            # Виджет для загрузки файла
            uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    df_batch = pd.read_csv(uploaded_file)
                    st.info(f"Файл загружен. Количество записей: {len(df_batch)}")
                    
                    with st.expander("Посмотреть исходные данные"):
                        st.dataframe(df_batch.head())
                    
                    if st.button("Запустить скоринг", type="primary"):
                        with st.spinner("Модель анализирует данные..."):
                            
                            # Инференс
                            probabilities = model.predict_proba(df_batch)[:, 1]
                            predictions = model.predict(df_batch)
                            
                            # Формируем итоговый датафрейм
                            df_result = df_batch.copy()
                            df_result['Churn_Probability_%'] = (probabilities * 100).round(2)
                            df_result['Churn_Prediction'] = ['Уйдет' if p == 1 else 'Останется' for p in predictions]
                            
                            st.success("✅ Скоринг успешно завершен!")
                            st.dataframe(df_result[['Churn_Probability_%', 'Churn_Prediction']].head(10))
                            
                            # Виджет скачивания
                            csv_data = df_result.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Скачать результаты (.csv)",
                                data=csv_data,
                                file_name='churn_predictions_results.csv',
                                mime='text/csv'
                            )
                except Exception as e:
                    st.error(f"Произошла ошибка при обработке файла: {e}")
                    st.write("Убедитесь, что загружаемый датафрейм содержит все необходимые колонки.")
