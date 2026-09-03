\# Everesteer — MCP Futures Market Prediction Platform



Built during the Everesteer hackathon (August 2026). A modular pipeline for predicting futures-market movement.



\## Results

\- +2.9% principal growth

\- 9th place out of 42 on the skill leaderboard

\- Key finding: plain Ridge regression outperformed LightGBM and ensemble methods on chronological holdout data



\## Tech Stack

Python, scikit-learn, LightGBM (with automatic sklearn GradientBoosting fallback), pandas



\## How it works

The pipeline runs in four stages: `data\_loader.py` loads and profiles incoming data (schema, missing values, target distribution), `features.py` engineers lag, rolling, momentum, return, and time-based features, `models.py` trains and compares a baseline (Ridge/Logistic), a tree model (LightGBM or sklearn GBM), and an ensemble blend across validation folds, and `pipeline.py` runs the full load → features → validate → train → submit sequence.



\## Data

Training/validation data omitted per competition data-sharing terms. Provide your own parquet files with the same schema to run the pipeline.



\## Setup

\\`\\`\\`

python pipeline.py

\\`\\`\\`

