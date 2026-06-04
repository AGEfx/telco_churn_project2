import sys
import importlib

sys.path.append('../utils')
import visualization
importlib.reload(visualization)
from visualization import save_to_html

import pandas as pd
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')


df = pd.read_parquet('../data/processed/telco_churn.parquet')

print(f"{df.shape[0]:,} строк и {df.shape[1]} колонок")
df.info()

# Новые флаги
df['high_monthly_flag'] = (df['MonthlyCharges'] > 80).astype(int)
df = df.rename(columns={'SeniorCitizen': 'senior_citizen_flag'})

# Количество дополнительных услуг у клиента
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']

df['num_additional_services'] = df[service_cols].apply(
    lambda x: (x == 'Yes').sum(), axis=1
)

# Флаг наличия хотя бы одной дополнительной услуги
df['has_additional_services'] = (df['num_additional_services'] > 0).astype(int)

# Комбинированный рисковый признак
df['high_risk_combination'] = (
    ((df['tenure_group'] == '1-6 months') & (df['high_monthly_flag'] == 1)) | 
    ((df['tenure_group'] == '7-12 months') & (df['high_monthly_flag'] == 1)) | 
    ((df['tenure_group'] == '1-6 months') & (df['fiber_optic_flag'] == 1)) | 
    ((df['tenure_group'] == '1-6 months') & (df['senior_citizen_flag'] == 1)) | 
    ((df['tenure_group'] == '1-6 months') & (df['electronic_check_flag'] == 1)) | 
    ((df['tenure_group'] == '1-6 months') & (df['TechSupport'] == 'No')) | 
    ((df['tenure_group'] == '1-6 months') & (df['OnlineSecurity'] == 'No')) | 
    ((df['month_to_month_flag'] == 1) & (df['senior_citizen_flag'] == 1)) | 
    ((df['month_to_month_flag'] == 1) & (df['fiber_optic_flag'] == 1)) | 
    ((df['electronic_check_flag'] == 1) & (df['OnlineBackup'] == 'No')) | 
    ((df['tenure_group'] == 1) & (df['electronic_check_flag'] == 1)) | 
    ((df['electronic_check_flag'] == 1) & (df['high_monthly_flag'] == 1)) | 
    ((df['fiber_optic_flag'] == 1) & (df['high_monthly_flag'] == 1)) 
).astype(int)

# Проверка новых признаков

# Словари графиков и таблиц для сохранения в html-файл
features_figs = {}
features_tables = {}

new_features = [ 
    'avg_monthly_per_tenure_month',
    'num_additional_services',
    'has_additional_services',
    'high_monthly_flag',
    'fiber_optic_flag',
    'electronic_check_flag',
    'month_to_month_flag',
    'senior_citizen_flag',
    'high_risk_combination'
]

features_tables['Первые 5 строк новых признаков'] = df[['customerID'] + new_features].head(5)

for col in ['has_additional_services', 'high_risk_combination']:
    freq = df[col].value_counts()
    percent = df[col].value_counts(normalize=True) * 100
    table = pd.concat([freq, percent.round(2)], axis=1, keys=['Count', 'Percent (%)'])

    fig = px.pie(df, names=col, title=f'Распределение {col}',
             color_discrete_sequence=px.colors.qualitative.Set2,
             hole=0.4)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(height=500)

    features_figs[f'Распределение {col}'] = fig

# Связь новых признаков с Churn 

for col in new_features:
    if col == 'avg_monthly_per_tenure_month':
        fig = px.histogram(df, x=col, color='Churn', 
                        nbins=50,
                        title=f'Распределение {col} по Churn',
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        barmode='overlay')
        fig.update_layout(height=500)

        features_figs[f'Распределение {col} по Churn'] = fig
        features_tables[f'Описание {col} по Churn'] = df.groupby('Churn')[col].describe().reset_index()

    else:
        churn_rate = (df.groupby(col, observed=False)['Churn']
                        .mean()
                        .mul(100)           
                        .round(2)
                        .reset_index(name='Churn_Rate'))
        
        churn_rate = churn_rate.sort_values('Churn_Rate', ascending=False)
        churn_rate_display = churn_rate[[col, 'Churn_Rate']]
        
        fig = px.bar(
            churn_rate,
            x=col,
            y='Churn_Rate',
            title=f'Процент оттока (Churn = 1) по {col}',
            color='Churn_Rate',                    
            color_continuous_scale='RdYlGn_r',       
            text='Churn_Rate',
            labels={'Churn_Rate': 'Процент ушедших (%)', col: col}        
        )
        
        fig.update_traces(texttemplate='%{y:.2f}%')

        features_figs[f'Распределение {col} по Churn'] = fig

# Сохранение графиков и таблиц в html-файл
save_to_html(
    figures_dict=features_figs, 
    tables_dict=features_tables, 
    filename="features.html", 
    title="Uni and Bivariate Analysis of New Features",
    dashboard_dir="feature_engineering"
)

df.to_csv('../data/processed/data_featured.csv', index=False)
df.to_parquet('../data/processed/data_featured.parquet', index=False)