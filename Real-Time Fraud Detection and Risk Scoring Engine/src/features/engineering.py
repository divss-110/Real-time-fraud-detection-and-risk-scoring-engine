import numpy as np
import pandas as pd

from src.config import ENTITY_KEY, MODEL_FEATURES, USER_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_to_avg_ratio"] = df["amount"] / (df["user_avg_amount_30d"] + 1.0)
    df["amount_zscore"] = (df["amount"] - df["user_avg_amount_30d"]) / (df["user_std_amount_30d"] + 1.0)
    return df


def join_user_features(transactions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    df = transactions.merge(users[[ENTITY_KEY] + USER_FEATURES], on=ENTITY_KEY, how="left")
    return add_derived_features(df)


def build_vector(txn: dict, user: dict) -> np.ndarray:
    row = {**txn, **user}
    row["amount_to_avg_ratio"] = row["amount"] / (row["user_avg_amount_30d"] + 1.0)
    row["amount_zscore"] = (row["amount"] - row["user_avg_amount_30d"]) / (row["user_std_amount_30d"] + 1.0)
    return np.array([[float(row[name]) for name in MODEL_FEATURES]], dtype=np.float64)
