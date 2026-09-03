"""
features.py
Common feature engineering for market/financial prediction tasks.
Since it's "hedge-fund-grade data" with a live leaderboard, these are the features
that give the fastest signal-to-effort payoff. Comment out what doesn't apply
once you see the real columns.
"""

import pandas as pd
import numpy as np


def add_lag_features(df: pd.DataFrame, col: str, lags=(1, 2, 3, 5, 10), group_col=None):
    """Lagged values of a numeric column. group_col = e.g. an asset/ticker id if data has multiple series."""
    for lag in lags:
        if group_col:
            df[f"{col}_lag{lag}"] = df.groupby(group_col)[col].shift(lag)
        else:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows=(3, 5, 10, 20), group_col=None):
    """Rolling mean/std/min/max — cheap signal, almost always helps."""
    grouped = df.groupby(group_col)[col] if group_col else df[col]
    for w in windows:
        roll = grouped.rolling(w, min_periods=1)
        df[f"{col}_roll{w}_mean"] = roll.mean().reset_index(level=0, drop=True) if group_col else roll.mean()
        df[f"{col}_roll{w}_std"] = roll.std().reset_index(level=0, drop=True) if group_col else roll.std()
    return df


def add_return_features(df: pd.DataFrame, price_col: str, group_col=None):
    """Simple and log returns — standard for any price/level series."""
    if group_col:
        df[f"{price_col}_ret1"] = df.groupby(group_col)[price_col].pct_change(1)
        df[f"{price_col}_logret1"] = np.log(df[price_col] / df.groupby(group_col)[price_col].shift(1))
    else:
        df[f"{price_col}_ret1"] = df[price_col].pct_change(1)
        df[f"{price_col}_logret1"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def add_momentum_features(df: pd.DataFrame, col: str, windows=(5, 10, 20), group_col=None):
    """Momentum = value now vs value N periods ago. Cheap and often predictive."""
    for w in windows:
        if group_col:
            df[f"{col}_mom{w}"] = df[col] - df.groupby(group_col)[col].shift(w)
        else:
            df[f"{col}_mom{w}"] = df[col] - df[col].shift(w)
    return df


def add_time_features(df: pd.DataFrame, time_col: str):
    """Hour/day/weekday cyclical features — useful if there's any intraday or weekly pattern."""
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def build_feature_set(df: pd.DataFrame, numeric_cols, time_col=None, group_col=None):
    """
    One-call convenience wrapper. Pass the numeric columns you want engineered.
    Run this, then drop rows with NaN from lag/rolling warm-up before training.
    """
    for col in numeric_cols:
        df = add_lag_features(df, col, group_col=group_col)
        df = add_rolling_features(df, col, group_col=group_col)
        df = add_momentum_features(df, col, group_col=group_col)

    if time_col and time_col in df.columns:
        df = add_time_features(df, time_col)

    return df
