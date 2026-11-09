import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FEAST_REPO = ROOT / "src" / "features" / "feast_repo"

TRANSACTIONS_PATH = RAW_DIR / "transactions.parquet"
CURRENT_BATCH_PATH = RAW_DIR / "current_batch.parquet"
USER_FEATURES_PATH = PROCESSED_DIR / "user_features.parquet"
REFERENCE_PATH = PROCESSED_DIR / "reference.parquet"
MODEL_PATH = MODELS_DIR / "model.txt"
METADATA_PATH = MODELS_DIR / "metadata.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraud-detection")

FEATURE_BACKEND = os.getenv("FEATURE_BACKEND", "feast")
FEAST_CONFIG = os.getenv("FEAST_CONFIG", "feature_store.yaml")
FEATURE_VIEW = "user_features"

ENTITY_KEY = "user_id"
TARGET = "is_fraud"
TIMESTAMP = "event_timestamp"

TRANSACTION_FEATURES = [
    "amount",
    "hour",
    "merchant_category",
    "is_foreign",
    "card_present",
    "device_change",
    "distance_from_home_km",
    "seconds_since_last_txn",
]

USER_FEATURES = [
    "user_avg_amount_30d",
    "user_std_amount_30d",
    "user_txn_count_7d",
    "user_account_age_days",
    "user_chargebacks_1y",
    "user_distinct_merchants_30d",
    "user_night_txn_ratio",
]

DERIVED_FEATURES = ["amount_to_avg_ratio", "amount_zscore"]

MODEL_FEATURES = TRANSACTION_FEATURES + USER_FEATURES + DERIVED_FEATURES
CATEGORICAL_FEATURES = ["merchant_category"]

REVIEW_COST = float(os.getenv("REVIEW_COST", "5.0"))
