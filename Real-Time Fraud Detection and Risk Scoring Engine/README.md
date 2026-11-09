# Real-Time Fraud Detection and Risk Scoring Engine

This project scores card transactions in real time and tells you how risky each one is. It is built for the hard version of the problem: only about 1 in 1000 transactions is fraud, and every answer has to come back in well under 50 milliseconds.

For every transaction the API returns:

* a risk score between 0 and 1
* a decision (approve, review or decline)
* the top reasons behind the score, using SHAP values
* how long each step took

## What is inside

```
div_project/
  data/
    synthetic_generator.py        creates the fake but realistic dataset
    raw/                          transactions and the drift batch (generated)
    processed/                    user features and the drift reference set (generated)
  src/
    config.py                     all paths, feature names and settings in one place
    features/
      engineering.py              feature logic shared by training and serving
      online_store.py             reads user features from Feast, or from a local table as a fallback
      materialize.py              registers Feast objects and loads the online store
      feast_repo/
        definitions.py            Feast entity, feature view and feature service
        feature_store.yaml        local setup using SQLite
        feature_store.redis.yaml  Docker setup using Redis
    models/
      train.py                    LightGBM plus Optuna, tuned for PR-AUC, tracked in MLflow
    inference/
      scorer.py                   loads the model, gets features, scores and explains
      main.py                     FastAPI app with the /score endpoint
    monitoring/
      drift.py                    Evidently drift report for a new batch
  notebooks/                      the full workflow, one notebook per step
  models/                         trained model and metadata (generated)
  reports/                        drift reports (generated)
  tests/                          API tests
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## How it works

```
  transaction  ->  FastAPI /score
                       |
                       |  1. look up user history by user_id
                       v
                 Feast online store  (SQLite locally, Redis in Docker)
                       |
                       |  2. join with the transaction fields, add ratio features
                       v
                 LightGBM model
                       |
                       |  3. one call returns both the score and exact SHAP values
                       v
                 score, decision, top reasons, timings
```

Training happens offline:

```
  synthetic_generator.py -> transactions + user features (parquet)
                         -> train.py (time based split, Optuna search, threshold by cost)
                         -> models/model.txt + models/metadata.json + MLflow run
                         -> materialize.py pushes user features into the online store
```

Monitoring runs on batches:

```
  new batch -> drift.py -> Evidently HTML report + a short JSON summary
```

## The main ideas, in plain words

**Class imbalance.** With 0.1% fraud, accuracy is useless because a model that always says "not fraud" is 99.9% accurate. So the model is tuned for PR-AUC, which only cares about how well fraud is ranked above normal traffic. Optuna also tunes `scale_pos_weight`, which tells LightGBM how much more a missed fraud should hurt than a false alarm.

**Business cost threshold.** A score is not a decision. The cutoff is picked on the validation set by minimising real cost: a missed fraud costs the transaction amount, and a false alarm costs a fixed review fee (5 by default, set with `REVIEW_COST`). Scores above the cutoff are declined, scores above half of it go to review.

**No time travel.** Data is split by time (70% train, 15% validation, 15% test), so the model is always tested on transactions that happen after the ones it learned from.

**Same features in training and serving.** Both sides use the same code in `src/features/engineering.py`, so the model sees the same numbers in production as it did in training.

**Fast explanations.** SHAP values come straight from LightGBM using `pred_contrib=True`. This is the exact TreeSHAP result, and it takes well under a millisecond, so there is no need for a separate explainer at request time. Values are in log odds: positive means the feature pushed the risk up.

**Drift.** `drift.py` compares a new batch against a sample of the training data. Evidently builds the visual report, and a KS test per column gives a quick machine readable summary in `reports/drift_summary.json`.

## Run it locally

You need Python 3.11 or newer.

```
pip install -r requirements.txt
python data/synthetic_generator.py
python -m src.models.train
python -m src.features.materialize
uvicorn src.inference.main:app --port 8000
```

Then open http://localhost:8000/docs and try `/score`, or call it directly:

```
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d "{\"transaction_id\":\"t1\",\"user_id\":42,\"amount\":1450,\"hour\":3,\"merchant_category\":7,\"is_foreign\":1,\"card_present\":0,\"device_change\":1,\"distance_from_home_km\":900,\"seconds_since_last_txn\":25}"
```

Drift check:

```
python data/synthetic_generator.py --drift
python -m src.monitoring.drift
```

Open `reports/drift_report.html` in a browser to see the result.

Tests:

```
pytest -q
```

MLflow runs:

```
mlflow ui --backend-store-uri ./mlruns
```

## Run with notebooks

Open the `notebooks` folder and run them in order. Each one does one step and shows the results.

1. `01_data_generation.ipynb` creates the data and shows the fraud rate and patterns
2. `02_feature_store.ipynb` sets up Feast and measures online lookup speed
3. `03_model_training.ipynb` trains the model and shows metrics, importance and the precision recall table
4. `04_realtime_inference.ipynb` calls the API 1000 times and reports p50, p95 and p99 latency
5. `05_drift_monitoring.ipynb` creates a shifted batch and builds the drift report

## Run with Docker

```
docker compose up --build
```

This starts Redis, runs the whole pipeline once (data, training, loading Redis, drift report), then starts the API on port 8000 and the MLflow UI on port 5000.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| FEATURE_BACKEND | feast | `feast` reads from the Feast online store, `local` reads the parquet file into memory |
| FEAST_CONFIG | feature_store.yaml | which Feast config to use |
| MLFLOW_TRACKING_URI | ./mlruns | where MLflow saves runs |
| REVIEW_COST | 5.0 | cost of one false alarm when choosing the threshold |

If Feast cannot be reached when the API starts, it logs a warning and falls back to the local table, so the API still works.

## API

| Method | Path | What it does |
|---|---|---|
| GET | /health | model version and feature source |
| GET | /model-info | features, thresholds and test metrics |
| POST | /score | scores one transaction |
| POST | /score/batch | scores up to 1000 transactions |

## Speed

Most of the time per request goes to the feature lookup. The model call and SHAP together take a fraction of a millisecond because the model is small and runs on one row. Notebook 4 prints the real numbers on your machine.
