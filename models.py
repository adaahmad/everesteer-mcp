"""
models.py
Three candidate models to have ready for the staking mechanic:
  1. baseline  - fast, interpretable, good fallback (Ridge / Logistic)
  2. tree      - usually your best single model (LightGBM if installed, else sklearn GBM)
  3. ensemble  - blend of the two, often the safest stake

is_classification: set True if predicting direction/class, False if predicting a continuous value.
"""

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def get_baseline(is_classification: bool):
    return LogisticRegression(max_iter=1000) if is_classification else Ridge(alpha=1.0)


def get_tree_model(is_classification: bool):
    if HAS_LGBM:
        params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8)
        return lgb.LGBMClassifier(**params) if is_classification else lgb.LGBMRegressor(**params)
    else:
        params = dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8)
        return GradientBoostingClassifier(**params) if is_classification else GradientBoostingRegressor(**params)


class EnsembleModel:
    """Simple weighted average blend of baseline + tree. Weight toward whichever validates better."""

    def __init__(self, is_classification: bool, weight_tree=0.7):
        self.is_classification = is_classification
        self.weight_tree = weight_tree
        self.baseline = get_baseline(is_classification)
        self.tree = get_tree_model(is_classification)

    def fit(self, X, y):
        self.baseline.fit(X, y)
        self.tree.fit(X, y)
        return self

    def predict(self, X):
        if self.is_classification:
            p1 = self.baseline.predict_proba(X)[:, 1]
            p2 = self.tree.predict_proba(X)[:, 1]
            return self.weight_tree * p2 + (1 - self.weight_tree) * p1
        else:
            p1 = self.baseline.predict(X)
            p2 = self.tree.predict(X)
            return self.weight_tree * p2 + (1 - self.weight_tree) * p1


def evaluate(y_true, y_pred, is_classification: bool):
    """Quick scoring — swap in whatever metric Everesteer's leaderboard actually uses once you know it."""
    if is_classification:
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float("nan")
        return {"auc": auc}
    else:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        return {"rmse": rmse}


def walk_forward_validate(df, feature_cols, target_col, is_classification, n_splits=5):
    """
    Time-series safe validation — never shuffle. Expanding window.
    Returns per-fold scores for baseline, tree, and ensemble so you can pick what to stake.
    """
    n = len(df)
    fold_size = n // (n_splits + 1)
    results = {"baseline": [], "tree": [], "ensemble": []}

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        val_end = fold_size * (i + 1)
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]

        if len(val) == 0:
            continue

        X_train, y_train = train[feature_cols], train[target_col]
        X_val, y_val = val[feature_cols], val[target_col]

        baseline = get_baseline(is_classification).fit(X_train, y_train)
        tree = get_tree_model(is_classification).fit(X_train, y_train)
        ens = EnsembleModel(is_classification).fit(X_train, y_train)

        for name, model in [("baseline", baseline), ("tree", tree), ("ensemble", ens)]:
            if is_classification and name != "ensemble":
                pred = model.predict_proba(X_val)[:, 1]
            else:
                pred = model.predict(X_val)
            results[name].append(evaluate(y_val, pred, is_classification))

    return results
