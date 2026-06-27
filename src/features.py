import pandas as pd
import numpy as np


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    df = df.rename(columns={'PJM_Load_MW': 'energy_mw'})
    df = df.set_index('Datetime')
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df = df.reindex(full_range)
    df['energy_mw'] = df['energy_mw'].interpolate(method='linear')
    df.index.name = 'Datetime'
    return df.reset_index()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    df['quarter'] = df['Datetime'].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['lag_24'] = df['energy_mw'].shift(24)
    df['lag_48'] = df['energy_mw'].shift(48)
    df['lag_168'] = df['energy_mw'].shift(168)
    df['rolling_mean_24'] = df['energy_mw'].shift(1).rolling(24).mean()
    df['rolling_mean_168'] = df['energy_mw'].shift(1).rolling(168).mean()
    df['rolling_std_24'] = df['energy_mw'].shift(1).rolling(24).std()
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_week'] = np.sin(2 * np.pi * df['Datetime'].dt.dayofyear / 7)
    df['cos_week'] = np.cos(2 * np.pi * df['Datetime'].dt.dayofyear / 7)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    return df.dropna().reset_index(drop=True)


FEATURE_COLS = [
    'hour', 'dayofweek', 'month', 'quarter', 'is_weekend',
    'lag_24', 'lag_48', 'lag_168',
    'rolling_mean_24', 'rolling_mean_168', 'rolling_std_24',
    'sin_hour', 'cos_hour', 'sin_week', 'cos_week', 'sin_month', 'cos_month'
]

TARGET_COL = 'energy_mw'
