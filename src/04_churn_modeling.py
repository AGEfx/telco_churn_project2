import sys
import importlib

sys.path.append('../utils')
import visualization
importlib.reload(visualization)
from visualization import save_to_html

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, precision_recall_curve)
import xgboost as xgb
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import json
import joblib
import shap

import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('../data/processed/data_featured.parquet')

print(f"Исходный размер данных: {df.shape}")

# Preprocessing

columns_to_drop = ['Churn', 'customerID', 'TotalCharges', 'gender']

y = df['Churn'].astype(int)
X = df.drop(columns=columns_to_drop, errors='ignore')

# Разделяем признаки
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['int8', 'int64', 'float64']).columns.tolist()

print(f"Категориальные признаки: {cat_cols}")
print(f"Числовые признаки: {num_cols}")

# Кодируем категориальные признаки
for col in cat_cols:
    X[col] = X[col].astype('category')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Bayesian optimization

def xgb_evaluate(max_depth, learning_rate, n_estimators, subsample, colsample_bytree, 
                 min_child_weight, gamma, scale_pos_weight):
    
    params = {
        'max_depth': int(max_depth),
        'learning_rate': float(learning_rate),
        'n_estimators': int(n_estimators),
        'subsample': float(subsample),
        'colsample_bytree': float(colsample_bytree),
        'min_child_weight': float(min_child_weight),
        'gamma': float(gamma),
        'scale_pos_weight': float(scale_pos_weight),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'verbosity': 0,
        'enable_categorical': True,     
        'use_label_encoder': False
    }
    
    model = xgb.XGBClassifier(**params)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr, verbose=False)
        
        y_pred_val = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred_val))
    
    return np.mean(scores)

pbounds = {
    'max_depth': (4, 10),
    'learning_rate': (0.02, 0.15),
    'n_estimators': (200, 900),
    'subsample': (0.7, 0.95),
    'colsample_bytree': (0.7, 0.95),
    'min_child_weight': (1, 8),
    'gamma': (0, 3),
    'scale_pos_weight': (1.5, 4.5)     
}


optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(
    init_points=10,
    n_iter=25
)

# Извлекаем лучшие параметры
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['scale_pos_weight'] = round(best_params['scale_pos_weight'], 2)

print("\nЛучшие параметры:")
print(json.dumps(best_params, indent=2))

# Final model

final_params = {
    'max_depth': best_params['max_depth'],
    'learning_rate': best_params['learning_rate'],
    'n_estimators': best_params['n_estimators'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'min_child_weight': best_params['min_child_weight'],
    'gamma': best_params['gamma'],
    'scale_pos_weight': best_params['scale_pos_weight'],
    'objective': 'binary:logistic',
    'random_state': 42,
    'enable_categorical': True
}

final_model = xgb.XGBClassifier(**final_params)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

print("Результаты модели")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_proba):.4f}")

# Model Evaluation + Threshold Tuning

modeling_figs = {}
modeling_tables = {}

y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred_default = final_model.predict(X_test)

# Метрики при стандартном пороге 0.5
metrics_df = pd.DataFrame([{
    'Threshold': 0.5,
    'Accuracy': round(accuracy_score(y_test, y_pred_default), 4),
    'Precision': round(precision_score(y_test, y_pred_default), 4),
    'Recall': round(recall_score(y_test, y_pred_default), 4),
    'F1-score': round(f1_score(y_test, y_pred_default), 4),
    'ROC-AUC': round(roc_auc_score(y_test, y_pred_proba), 4)
}])

# Подбор оптимального порога по F1-score
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

y_pred_opt = (y_pred_proba >= optimal_threshold).astype(int)

opt_metrics_df = pd.DataFrame([{
    'Threshold': round(optimal_threshold, 4),
    'Accuracy': round(accuracy_score(y_test, y_pred_opt), 4),
    'Precision': round(precision_score(y_test, y_pred_opt), 4),
    'Recall': round(recall_score(y_test, y_pred_opt), 4),
    'F1-score': round(f1_score(y_test, y_pred_opt), 4),
    'ROC-AUC': round(roc_auc_score(y_test, y_pred_proba), 4)
}])

metrics_df = (pd.concat([metrics_df, opt_metrics_df], ignore_index=True).style
              .format({
                  'Threshold': '{:.4f}', 
                  'Accuracy': '{:.4f}', 
                  'Precision': '{:.4f}', 
                  'Recall': '{:.4f}', 
                  'F1-score': '{:.4f}' , 
                  'ROC-AUC': '{:.4f}'
                  })
              .set_caption("Метрики при пороге 0.5 и оптимальном пороге"))
modeling_tables["Метрики при пороге 0.5 и оптимальном пороге"] = metrics_df

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_opt)

fig_cm = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale='Blues',
    labels=dict(x="Predicted", y="Actual", color="Count"),
    title=f"Confusion Matrix (Threshold = {optimal_threshold:.3f})"
)
fig_cm.update_layout(width=600, height=500)
modeling_figs["Confusion Matrix"] = fig_cm

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

fig_roc = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700,
    height=500
)
fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
modeling_figs["ROC Curve"] = fig_roc

# Feature Importance

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

top15 = importance_df.head(15)

fig_imp = px.bar(
    top15,
    x='importance',
    y='feature',
    orientation='h',
    title='Top 15 Most Important Features (XGBoost - Gain)',
    labels={'importance': 'Feature Importance (Gain)', 'feature': 'Feature'},
    color='importance',
    color_continuous_scale='Viridis',
    text='importance'
)

fig_imp.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig_imp.update_layout(
    height=680,
    width=950,
    yaxis=dict(autorange="reversed"), 
    xaxis_title="Importance Score",
    title_font_size=18
)

modeling_figs["Feature Importance"] = fig_imp

save_to_html(
    figures_dict=modeling_figs, 
    tables_dict=modeling_tables, 
    filename="modeling_results.html", 
    title="Model Performance Dashboard", 
    dashboard_dir="modeling"
)

# Сохранение лучшей модели

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

model_path = f'../models/xgboost_churn_best_{timestamp}.pkl'
joblib.dump(final_model, model_path)

model_info = {
    "date": timestamp,
    "best_params": best_params,
    "optimal_threshold": float(optimal_threshold),
    "metrics": {
        "accuracy": float(accuracy_score(y_test, y_pred_opt)),
        "precision": float(precision_score(y_test, y_pred_opt)),
        "recall": float(recall_score(y_test, y_pred_opt)),
        "f1_score": float(f1_score(y_test, y_pred_opt)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba))
    }
}

with open(f'../models/model_info_{timestamp}.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"Модель успешно сохранена в '{model_path}'")

# SHAP Analysis

shap_figs = {}
shap_tables = {}

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

# Summary Plot (Beeswarm) 
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.title("SHAP Summary Plot — Влияние признаков на предсказание оттока")
plt.tight_layout()
plt.savefig('../dashboards/modeling/shap_summary_plot.png', dpi=150, bbox_inches='tight')

# Среднее абсолютное влияние признака
shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
mean_shap = np.abs(shap_df).mean().sort_values(ascending=False).head(15).reset_index()
mean_shap.columns = ['feature', 'mean_shap']

# График Топ-15 признаков по среднему влиянию (SHAP)
fig_shap = px.bar(
    mean_shap,
    x='mean_shap',
    y='feature',
    orientation='h',
    title='Топ-15 признаков по среднему влиянию (SHAP)',
    labels={'mean_shap': 'Среднее абсолютное SHAP значение', 'feature': 'Признак'},
    color='mean_shap',
    color_continuous_scale='RdYlBu_r'
)

fig_shap.update_layout(height=650, width=950, yaxis=dict(autorange="reversed"))
shap_figs["Топ-15 признаков по среднему влиянию (SHAP)"] = fig_shap

# Waterfall Plot
high_risk_idx = np.argsort(y_pred_proba)[-1] 

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[high_risk_idx],
        feature_names=X_test.columns.tolist()
    ),
    max_display=15,
    show=False
)
plt.title("Пример объяснения предсказания для одного клиента (SHAP Waterfall Plot)")
plt.savefig('../dashboards/modeling/shap_waterfall_plot.png', dpi=150, bbox_inches='tight')

# Топ-10 признаков в табличном виде
shap_tables["Топ-10 самых важных признаков по SHAP"] = mean_shap.head(10).style.format({'mean_shap': '{:.4f}'})

save_to_html(
    figures_dict=shap_figs, 
    tables_dict=shap_tables, 
    filename="shap.html", 
    title="SHAP Analysis", 
    dashboard_dir="modeling"
)


