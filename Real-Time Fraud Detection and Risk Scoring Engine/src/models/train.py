import argparse
import json
import os
import time

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from src.config import (
    CATEGORICAL_FEATURES,
    METADATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_FEATURES,
    MODEL_PATH,
    MODELS_DIR,
    REFERENCE_PATH,
    REVIEW_COST,
    TARGET,
    TIMESTAMP,
    TRANSACTIONS_PATH,
    USER_FEATURES,
    USER_FEATURES_PATH,
)
from src.features.engineering import join_user_features

THREADS = os.cpu_count() or 4


def load_dataset() -> pd.DataFrame:
    transactions = pd.read_parquet(TRANSACTIONS_PATH)
    users = pd.read_parquet(USER_FEATURES_PATH)
    return join_user_features(transactions, users).sort_values(TIMESTAMP).reset_index(drop=True)


def time_split(df: pd.DataFrame, train_share: float = 0.7, valid_share: float = 0.15):
    n = len(df)
    a, b = int(n * train_share), int(n * (train_share + valid_share))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def business_cost(y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray) -> float:
    missed_fraud = amounts[(y_true == 1) & (y_pred == 0)].sum()
    false_alarms = ((y_true == 0) & (y_pred == 1)).sum() * REVIEW_COST
    return float(missed_fraud + false_alarms)


def best_threshold(y_true: np.ndarray, scores: np.ndarray, amounts: np.ndarray) -> tuple[float, float]:
    grid = np.unique(np.quantile(scores, np.linspace(0.90, 0.99999, 400)))
    costs = [business_cost(y_true, (scores >= t).astype(int), amounts) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def base_params() -> dict:
    return {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "num_threads": THREADS,
        "bagging_freq": 1,
        "seed": 42,
    }


def make_objective(dtrain: lgb.Dataset, dvalid: lgb.Dataset, x_valid: pd.DataFrame, y_valid: np.ndarray):
    def objective(trial: optuna.Trial) -> float:
        params = {
            **base_params(),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 96),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 200.0, log=True),
        }
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        scores = booster.predict(x_valid, num_iteration=booster.best_iteration)
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        return float(average_precision_score(y_valid, scores))

    return objective


def evaluate(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, threshold: float, prefix: str) -> dict:
    pred = (scores >= threshold).astype(int)
    return {
        f"{prefix}_pr_auc": float(average_precision_score(y, scores)),
        f"{prefix}_roc_auc": float(roc_auc_score(y, scores)),
        f"{prefix}_precision": float(precision_score(y, pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y, pred, zero_division=0)),
        f"{prefix}_business_cost": business_cost(y, pred, amounts),
        f"{prefix}_alert_rate": float(pred.mean()),
    }


def train(n_trials: int = 25, timeout: int = 300) -> dict:
    started = time.perf_counter()
    df = load_dataset()
    train_df, valid_df, test_df = time_split(df)
    x_train, y_train = train_df[MODEL_FEATURES], train_df[TARGET].to_numpy()
    x_valid, y_valid = valid_df[MODEL_FEATURES], valid_df[TARGET].to_numpy()
    x_test, y_test = test_df[MODEL_FEATURES], test_df[TARGET].to_numpy()
    print(f"Rows train {len(train_df):,} valid {len(valid_df):,} test {len(test_df):,}")
    print(f"Fraud cases train {y_train.sum()} valid {y_valid.sum()} test {y_test.sum()}")

    dtrain = lgb.Dataset(x_train, y_train, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    dvalid = lgb.Dataset(x_valid, y_valid, reference=dtrain, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(dtrain, dvalid, x_valid, y_valid), n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_params = {**base_params(), **study.best_params}
    best_rounds = study.best_trial.user_attrs["best_iteration"]
    print(f"Best validation PR-AUC {study.best_value:.4f} after {len(study.trials)} trials")

    booster = lgb.train(best_params, dtrain, num_boost_round=best_rounds)
    valid_scores = booster.predict(x_valid)
    test_scores = booster.predict(x_test)
    threshold, _ = best_threshold(y_valid, valid_scores, valid_df["amount"].to_numpy())

    metrics = {
        **evaluate(y_valid, valid_scores, valid_df["amount"].to_numpy(), threshold, "valid"),
        **evaluate(y_test, test_scores, test_df["amount"].to_numpy(), threshold, "test"),
        "threshold": threshold,
        "best_rounds": best_rounds,
        "n_trials": len(study.trials),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH))
    importance = dict(zip(MODEL_FEATURES, booster.feature_importance("gain").round(2).tolist()))
    metadata = {
        "model_version": time.strftime("%Y%m%d%H%M%S"),
        "features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold": threshold,
        "review_threshold": threshold * 0.5,
        "feature_defaults": train_df[USER_FEATURES].median().astype(float).to_dict(),
        "params": best_params,
        "metrics": metrics,
        "feature_importance_gain": importance,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    reference = train_df.sample(n=min(20_000, len(train_df)), random_state=42)[MODEL_FEATURES].copy()
    reference["prediction"] = booster.predict(reference[MODEL_FEATURES])
    reference.to_parquet(REFERENCE_PATH, index=False)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"lgbm_{metadata['model_version']}") as run:
        mlflow.log_params({k: v for k, v in best_params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="artifacts")
        mlflow.log_artifact(str(METADATA_PATH), artifact_path="artifacts")
        mlflow.lightgbm.log_model(booster, "model")
        metadata["mlflow_run_id"] = run.info.run_id
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Test PR-AUC {metrics['test_pr_auc']:.4f} ROC-AUC {metrics['test_roc_auc']:.4f}")
    print(f"Threshold {threshold:.4f} precision {metrics['test_precision']:.3f} recall {metrics['test_recall']:.3f}")
    print(f"Saved model to {MODEL_PATH} in {time.perf_counter() - started:.1f}s")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LightGBM fraud model with Optuna and MLflow")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    train(args.trials, args.timeout)


if __name__ == "__main__":
    main()
