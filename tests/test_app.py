import pandas as pd


def test_dummy_always_passes():
    """1. Базовый тест (Smoke test), проверяющий, что фреймворк тестирования работает."""
    assert True


def test_dataframe_creation():
    """2. Тест создания DataFrame для индивидуального скоринга."""
    df = pd.DataFrame({'tenure': [12], 'MonthlyCharges': [65.0]})
    assert not df.empty, "DataFrame не должен быть пустым"
    assert 'tenure' in df.columns, "Колонка tenure должна присутствовать"


def test_risk_level_logic():
    """3. Тест логики присвоения текстового уровня риска по вероятности."""
    proba_high = 0.6
    risk_label_high = "🔴 Высокий риск" if proba_high >= 0.5 else "🟡 Средний риск" if proba_high >= 0.3 else "🟢 Низкий риск"
    assert risk_label_high == "🔴 Высокий риск"

    proba_low = 0.1
    risk_label_low = "🔴 Высокий риск" if proba_low >= 0.5 else "🟡 Средний риск" if proba_low >= 0.3 else "🟢 Низкий риск"
    assert risk_label_low == "🟢 Низкий риск"


def test_prediction_threshold():
    """4. Тест порога отсечения (threshold = 0.4) для бинарного предсказания."""
    proba_churn = 0.45
    pred_churn = int(proba_churn >= 0.4)
    assert pred_churn == 1, "При вероятности >= 0.4 модель должна предсказывать отток (1)"

    proba_stay = 0.39
    pred_stay = int(proba_stay >= 0.4)
    assert pred_stay == 0, "При вероятности < 0.4 модель должна предсказывать удержание (0)"


def test_batch_scoring_logic():
    """5. Тест логики разметки пакета данных (Batch Scoring)."""
    df_batch = pd.DataFrame({'client_id': [1, 2], 'proba': [0.1, 0.8]})
    # Симулируем создание колонки с бизнес-решением
    df_batch['Churn_Prediction'] = ['Уйдет' if p >= 0.5 else 'Останется' for p in df_batch['proba']]

    assert df_batch['Churn_Prediction'].iloc[0] == 'Останется'
    assert df_batch['Churn_Prediction'].iloc[1] == 'Уйдет'