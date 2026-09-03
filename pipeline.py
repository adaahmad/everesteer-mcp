"""
pipeline.py
End-to-end run: load -> engineer features -> validate -> train final model -> write submission.
This is the file you actually run during the hackathon. Edit the CONFIG block, then:

    python pipeline.py

Goal: under 2 minutes from "new data dropped" to "submission written", so you can
iterate across rounds fast.
"""

import time
import pandas as pd
from data_loader import load_data, quick_profile, TARGET_COL, TIME_COL, ID_COL, DROP_COLS
from features import build_feature_set
from models import get_baseline, get_tree_model, EnsembleModel, walk_forward_validate, evaluate

# ============================================================
# CONFIG — edit per round
# ============================================================
DATA_PATH = "data.csv"                 # wherever tonight's data drops
SUBMIT_PATH = "submission.csv"
IS_CLASSIFICATION = False              # True if predicting a class/direction, False if continuous
NUMERIC_FEATURE_COLS = []              # fill in from quick_profile() output
FINAL_MODEL = "ensemble"               # "baseline" | "tree" | "ensemble" — swap based on validation
# ============================================================


def run():
    t0 = time.time()

    df = load_data(DATA_PATH)
    print(f"[{time.time()-t0:.1f}s] loaded {df.shape}")

    df = build_feature_set(df, NUMERIC_FEATURE_COLS, time_col=TIME_COL)
    df = df.dropna().reset_index(drop=True)  # drop lag/rolling warm-up rows
    print(f"[{time.time()-t0:.1f}s] features built, {df.shape} after dropna")

    feature_cols = [
        c for c in df.columns
        if c not in [TARGET_COL, TIME_COL, ID_COL] + DROP_COLS
    ]

    # Split off a holdout for validation, train on the rest, time-ordered (no shuffling)
    split_idx = int(len(df) * 0.85)
    train_df, val_df = df.iloc[:split_idx], df.iloc[split_idx:]

    scores = walk_forward_validate(train_df, feature_cols, TARGET_COL, IS_CLASSIFICATION, n_splits=4)
    print(f"[{time.time()-t0:.1f}s] validation scores:")
    for name, folds in scores.items():
        print(f"  {name}: {folds}")

    # Train final model on ALL available data before predicting the real target
    X_all, y_all = df[feature_cols], df[TARGET_COL]

    if FINAL_MODEL == "baseline":
        model = get_baseline(IS_CLASSIFICATION).fit(X_all, y_all)
    elif FINAL_MODEL == "tree":
        model = get_tree_model(IS_CLASSIFICATION).fit(X_all, y_all)
    else:
        model = EnsembleModel(IS_CLASSIFICATION).fit(X_all, y_all)

    print(f"[{time.time()-t0:.1f}s] final model trained ({FINAL_MODEL})")

    # ---- Predict on whatever needs a submission ----
    # Replace this block once you know if there's a separate "to-predict" file
    # or if you're predicting the tail/holdout of this same file.
    to_predict = val_df
    X_pred = to_predict[feature_cols]

    if IS_CLASSIFICATION and FINAL_MODEL != "ensemble":
        preds = model.predict_proba(X_pred)[:, 1]
    else:
        preds = model.predict(X_pred)

    submission = pd.DataFrame({
        ID_COL: to_predict[ID_COL] if ID_COL in to_predict.columns else to_predict.index,
        "prediction": preds,
    })
    submission.to_csv(SUBMIT_PATH, index=False)
    print(f"[{time.time()-t0:.1f}s] submission written to {SUBMIT_PATH}, shape {submission.shape}")


if __name__ == "__main__":
    run()
