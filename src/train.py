import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from features import load_and_clean, build_features, FEATURE_COLS, TARGET_COL


def train(data_path: str, model_path: str, split_date: str = '2001-10-01'):
    df = load_and_clean(data_path)
    df = build_features(df)

    train = df[df['Datetime'] < split_date]
    test  = df[df['Datetime'] >= split_date]

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test,  y_test  = test[FEATURE_COLS],  test[TARGET_COL]

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 64,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_test,  label=y_test, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100)
        ]
    )

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mape  = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100

    print(f"MAE: {mae:,.0f} MW | RMSE: {rmse:,.0f} MW | MAPE: {mape:.2f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    return model, {'mae': mae, 'rmse': rmse, 'mape': mape}


if __name__ == '__main__':
    train(
        data_path='data/raw/PJM_Load_hourly.csv',
        model_path='models/lgbm_energy.txt'
    )
