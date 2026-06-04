import sys
import importlib

sys.path.append('../utils')
import visualization
importlib.reload(visualization)
from visualization import save_to_html

import pandas as pd
import plotly.express as px
import itertools
from scipy.stats import ttest_ind, pointbiserialr

df = pd.read_parquet('../data/processed/telco_churn.parquet')

# Univariate Analysis
# Распределение целевой переменной Churn

# Словарь графиков для сохранения в html-файл
univariate_figs = {}

total_customers = df['Churn'].count()
churn_yes = (df['Churn'] == 1).sum()
churn_no = (df['Churn'] == 0).sum()

fig = px.pie(df, names='Churn', title='Распределение оттока клиентов (Churn)',
             color_discrete_sequence=px.colors.qualitative.Set2,
             hole=0.4)
fig.update_traces(textinfo='percent+label')
fig.update_layout(height=500)

print(f"Всего клиентов: {total_customers}")
print(f"Оставшихся (Churn = 1): {churn_no} - {round(churn_no / total_customers * 100.0, 2)}%") 
print(f"Ушедших (Churn = 0): {churn_yes} - {round(churn_yes / total_customers * 100.0, 2)}%") 

univariate_figs["Распределение Churn"] = fig

# Общее распределение числовых переменных

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

for col in numerical_cols:
    fig = px.histogram(df, x=col,
                        marginal='box',
                        nbins=50,
                        title=f'Общее распределение {col}',
                        color_discrete_sequence=['blue'],
                        barmode='overlay')
    fig.update_layout(height=500)

    univariate_figs[f"Распределение {col}"] = fig

    # Таблицы и графики частот для категориальных признаков

categorical_cols = [col for col in df.columns if col not in numerical_cols 
                    and col not in ['customerID', 'Churn', 'customer_type', 
                                    'avg_monthly_per_tenure_month', 'fiber_optic_flag', 
                                    'electronic_check_flag', 'month_to_month_flag']]

for col in categorical_cols:
    freq = df[col].value_counts().head(10)
    percent = df[col].value_counts(normalize=True).head(10) * 100
    table = pd.concat([freq, percent.round(2)], axis=1, keys=['Count', 'Percent (%)'])

    fig = px.pie(df, names=col, title=f'Распределение {col}',
             color_discrete_sequence=px.colors.qualitative.Set2,
             hole=0.4)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(height=500)

    univariate_figs[f"Распределение {col}"] = fig

# Сохранение графиков в html-файл
save_to_html(
    figures_dict=univariate_figs,
    filename="eda_univariate.html",
    title="Univariate Analysis",
    dashboard_dir="eda"
)

# Bivariate Analysis
# Распределение числовых переменных с разделением по Churn

# Словари графиков и таблиц для сохранения в html-файл
bivariate_figs = {}
bivariate_tables = {}

for col in numerical_cols:
    fig = px.histogram(df, x=col, color='Churn', 
                        marginal='box',
                        nbins=50,
                        title=f'Распределение {col} по Churn',
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        barmode='overlay')
    fig.update_layout(height=500)

    print(f'Описание {col} по Churn:')
    table = df.groupby('Churn')[col].describe().reset_index()

    bivariate_figs[f"Распределение {col} по Churn"] = fig
    bivariate_tables[f"Описание {col} по Churn"] = table

# Статистические тесты: влияние числовых признаков на Churn

results = []

for col in numerical_cols:
    group_yes = df[df['Churn'] == 1][col]
    group_no  = df[df['Churn'] == 0][col]
    
    # t-тест
    t_stat, p_value_t = ttest_ind(group_yes, group_no, equal_var=False, nan_policy='omit')
    
    # Point-Biserial Correlation
    corr, p_value_pb = pointbiserialr(df['Churn'], df[col])
    
    # Интерпретация силы связи
    if abs(corr) < 0.1:
        strength = "очень слабая"
    elif abs(corr) < 0.3:
        strength = "слабая"
    elif abs(corr) < 0.5:
        strength = "средняя"
    else:
        strength = "сильная"
    
    results.append({
        'Признак': col,
        'Mean (No Churn)': round(group_no.mean(), 2),
        'Mean (Yes Churn)': round(group_yes.mean(), 2),
        'Разница средних': round(group_yes.mean() - group_no.mean(), 2),
        't-статистика': round(t_stat, 4),
        'p-value (t-test)': f"{p_value_t:.6f}",
        'Корреляция (Point-Biserial)': round(corr, 4),
        'Сила связи': strength,
        'p-value (корр.)': f"{p_value_pb:.6f}"
    })

# Вывод результатов в виде таблицы
stats_df = pd.DataFrame(results)

bivariate_tables["Статистические тесты влияния числовых признаков на отток"] = stats_df

# Анализ оттока (Churn = 1) по категориям

results = []

for col in categorical_cols:
    is_numeric = False
    if pd.api.types.is_numeric_dtype(df[col]):
        is_numeric = True
        df[col] = df[col].astype('category')

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

    bivariate_figs[f"Churn Rate по {col}"] = fig

    for _, row in churn_rate_display.iterrows():
            results.append({
                'category': col,
                'value': f"{row[col]}",
                'churn_rate': row['Churn_Rate']
            })

    if is_numeric:
        df[col] = df[col].astype('int8')

top_category = pd.DataFrame(results)
top10 = top_category.sort_values('churn_rate', ascending=False).head(10).reset_index(drop=True)

bivariate_tables["ТОП-10 самых рискованных категориальных признаков"] = top10[['category', 'value', 'churn_rate']].style.format({'churn_rate': '{:.2f}%'})

# Сохранение графиков и таблиц в html-файл
save_to_html(
     figures_dict=bivariate_figs,
     tables_dict=bivariate_tables,
     filename="eda_bivariate.html",
     title="Bivariate Analysis",
     dashboard_dir="eda")


# Multivariate Analysis
# Анализ комбинаций MonthlyCharges + Категориальные признаки

# Словари графиков и таблиц для сохранения в html-файл
multivariate_figs = {}
multivariate_tables = {}

cat_cols = ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport', 
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'SeniorCitizen', 'tenure_group']
results = []

df['MonthlyCharges_bin'] = pd.qcut(df['MonthlyCharges'], q=4, duplicates='drop', 
                                labels=['Low', 'Medium-Low', 'Medium-High', 'High'])  

for cat_col in cat_cols:
    rate = (df.groupby([cat_col, 'MonthlyCharges_bin'], observed=False)['Churn'].mean()                 
        .mul(100)                  
        .round(2)
        .reset_index(name='Churn_Rate'))
    
    if cat_col != 'tenure_group':
        for _, row in rate.iterrows():
            results.append({
                'combination': f"{row[cat_col]} x {row['MonthlyCharges_bin']}",
                'feature_pair': f"{cat_col} x MonthlyCharges",
                'churn_rate': row['Churn_Rate']
            })

    
    if cat_col == 'tenure_group':
        rate = rate.sort_values('Churn_Rate', ascending=False).reset_index(drop=True)

        fig = px.bar(rate.head(20), 
                    x=cat_col, 
                    y='Churn_Rate', 
                    color='MonthlyCharges_bin',
                    barmode='group',
                    title=f'Churn Rate: {cat_col} x Уровень MonthlyCharges',
                    text='Churn_Rate',
                    labels={'MonthlyCharges_bin': 'Уровень MonthlyCharges'})
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=550, xaxis_tickangle=45)
        fig.show()

        multivariate_figs[f"Churn Rate по {cat_col} x MonthlyCharges"] = fig

        table = rate.head(10).style.format({'Churn_Rate': '{:.2f}%'})
        multivariate_tables[f"ТОП-10 рискованных комбинаций tenure_group x MonthlyCharges"] = rate.head(10).style.format({'Churn_Rate': '{:.2f}%'})


top_combinations = pd.DataFrame(results)
top10 = top_combinations.sort_values('churn_rate', ascending=False).head(10).reset_index(drop=True)
multivariate_tables["ТОП-10 рискованных комбинаций c MonthlyCharges (исключая tenure_group)"] = top10[['feature_pair', 'combination', 'churn_rate']].style.format({'churn_rate': '{:.2f}%'})


df.drop(columns=['MonthlyCharges_bin'], inplace=True, errors='ignore')

# Multivariate Analysis
# Анализ комбинаций категориальных признаков

combinations = list(itertools.combinations(cat_cols, 2))
results_with_tenure = []
results_no_tenure = []

for col1, col2 in combinations:
    rate = (df.groupby([col1, col2], observed=False)['Churn']
        .mean()                 
        .mul(100)                  
        .round(2)
        .reset_index(name='Churn_Rate'))
    
    rate = rate.sort_values('Churn_Rate', ascending=False)
    
    
    if col1 == 'tenure_group' or col2 == 'tenure_group':
        for _, row in rate.iterrows():
            results_with_tenure.append({
                'col1' : col1,
                'col2' : col2,
                'feature_pair' : f"{col1} x {col2}",
                'combination': f"{row[col1]} x {row[col2]}",
                'churn_rate': row['Churn_Rate']
            })

    if col1 != 'tenure_group' and col2 != 'tenure_group':
        for _, row in rate.iterrows():
            results_no_tenure.append({
                'col1' : col1,
                'col2' : col2,
                'feature_pair' : f"{col1} x {col2}",
                'combination': f"{row[col1]} x {row[col2]}",
                'churn_rate': row['Churn_Rate']
            })


top_combinations_with_tenure = pd.DataFrame(results_with_tenure)
top10_with_tenure = top_combinations_with_tenure.sort_values('churn_rate', ascending=False).head(10).reset_index(drop=True)
multivariate_tables["ТОП-10 самых рискованных комбинаций (включая tenure_group)"] = top10_with_tenure[['feature_pair', 'combination', 'churn_rate']].style.format({'churn_rate': '{:.2f}%'})

top_combinations_no_tenure = pd.DataFrame(results_no_tenure)
top10_no_tenure = top_combinations_no_tenure.sort_values('churn_rate', ascending=False).head(10).reset_index(drop=True)
multivariate_tables["ТОП-10 самых рискованных комбинаций (исключая tenure_group)"] = top10_no_tenure[['feature_pair', 'combination', 'churn_rate']].style.format({'churn_rate': '{:.2f}%'})

# Сохранение графиков и таблиц в html-файл
save_to_html(
    figures_dict=multivariate_figs,
    tables_dict=multivariate_tables,
    filename="eda_multivariate.html",
    title="Multivariate Analysis",
    dashboard_dir="eda"
)

