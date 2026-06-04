import os


def save_to_html(figures_dict=None, tables_dict=None, filename="report.html", 
                 title="Отчёт", dashboard_dir="eda"):
    """
    Сохраняет в один HTML-файл графики и/или таблицы.
    
    Параметры:
        figures_dict (dict): { 'Название графика': fig, ... } — опционально
        tables_dict (dict):  { 'Название таблицы': pd.DataFrame, ... } — опционально
        filename (str): имя файла
        title (str): заголовок страницы
        dashboard_dir (str): подпапка внутри ../dashboards/
    """

    folder_dir = f"../dashboards/{dashboard_dir}"
    os.makedirs(folder_dir, exist_ok=True)
    filepath = f"{folder_dir}/{filename}"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"""<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #f9f9f9; }}
        h1 {{ text-align: center; color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 40px; }}
        .plot-container {{ margin-bottom: 50px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>{title}</h1>
""")
        if filename == "shap.html":
            f.write(f"""
                    <h2>Графики</h2>
        <img src="shap_summary_plot.png" alt="SHAP Summary Plot" style="max-width: 100%; height: auto; display: block; margin: 30px 0; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
        <img src="shap_waterfall_plot.png" alt="SHAP Waterfall Plot" style="max-width: 100%; height: auto; display: block; margin: 30px 0; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
""")
        if figures_dict:
            if filename != "shap.html":
                f.write('<h2>Графики</h2>')
            for fig in figures_dict.values():
                f.write(f'    <div class="plot-container">')
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write('    </div>\n')

        if tables_dict:
            f.write('<h2>Таблицы</h2>')
            for name, df in tables_dict.items():
                f.write(f'    <div class="table-container">')
                f.write(f'        <h3>{name}</h3>')
                f.write(df.to_html(index=False, classes='table', escape=False))
                f.write('    </div>\n')

        f.write("</body></html>")

    return f"Сохранено в {filepath}"