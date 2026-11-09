import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CURRENT_BATCH_PATH, PROCESSED_DIR, RAW_DIR, TRANSACTIONS_PATH, USER_FEATURES_PATH

START = pd.Timestamp("2026-01-01", tz="UTC")
RISKY_MERCHANTS = np.array([3, 7, 15])


def generate_users(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    avg = rng.lognormal(3.6, 0.7, n_users)
    users = pd.DataFrame(
        {
            "user_id": np.arange(n_users, dtype=np.int64),
            "user_avg_amount_30d": avg.round(2),
            "user_std_amount_30d": (avg * rng.uniform(0.2, 0.9, n_users)).round(2),
            "user_txn_count_7d": rng.poisson(12, n_users).astype(np.int64),
            "user_account_age_days": rng.integers(1, 3650, n_users, dtype=np.int64),
            "user_chargebacks_1y": rng.poisson(0.05, n_users).astype(np.int64),
            "user_distinct_merchants_30d": rng.integers(1, 40, n_users, dtype=np.int64),
            "user_night_txn_ratio": rng.beta(1.2, 8.0, n_users).round(4),
        }
    )
    users["event_timestamp"] = START
    users["created"] = START
    return users


def solve_intercept(logit: np.ndarray, target_rate: float) -> float:
    low, high = -30.0, 10.0
    for _ in range(60):
        mid = (low + high) / 2.0
        rate = (1.0 / (1.0 + np.exp(-(logit + mid)))).mean()
        if rate > target_rate:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def generate_transactions(
    users: pd.DataFrame,
    n: int,
    fraud_rate: float,
    rng: np.random.Generator,
    start: pd.Timestamp,
    days: int,
    drift: bool,
) -> pd.DataFrame:
    uid = rng.integers(0, len(users), n)
    avg = users["user_avg_amount_30d"].to_numpy()[uid]
    std = users["user_std_amount_30d"].to_numpy()[uid]
    age = users["user_account_age_days"].to_numpy()[uid]
    chargebacks = users["user_chargebacks_1y"].to_numpy()[uid]
    night_ratio = users["user_night_txn_ratio"].to_numpy()[uid]

    offsets = np.sort(rng.uniform(0, days * 86400, n))
    ts = start + pd.to_timedelta(offsets, unit="s")
    night = rng.random(n) < night_ratio
    hour = np.where(night, rng.integers(0, 6, n), rng.integers(6, 24, n))

    amount = np.abs(rng.normal(avg, std)) + 0.5
    foreign_rate, distance_scale = 0.04, 15.0
    if drift:
        amount = amount * 1.35
        foreign_rate, distance_scale = 0.09, 22.0

    is_foreign = (rng.random(n) < foreign_rate).astype(np.int64)
    card_present = (rng.random(n) < 0.65).astype(np.int64)
    device_change = (rng.random(n) < 0.03).astype(np.int64)
    merchant_category = rng.integers(0, 20, n, dtype=np.int64)
    distance = rng.exponential(distance_scale, n)
    since_last = rng.exponential(20000.0, n)

    ratio = np.clip(amount / (avg + 1.0), 0, 10)
    logit = (
        1.1 * np.log1p(ratio) * 2.0
        + 1.6 * is_foreign
        + 1.2 * (1 - card_present)
        + 2.2 * device_change
        + 0.5 * np.log1p(distance)
        + 1.1 * (hour < 6)
        + 1.0 * np.minimum(chargebacks, 3)
        + 1.0 * (age < 60)
        + 0.8 * np.isin(merchant_category, RISKY_MERCHANTS)
        + 1.3 * (since_last < 300)
    )
    intercept = solve_intercept(logit, fraud_rate)
    prob = 1.0 / (1.0 + np.exp(-(logit + intercept)))
    is_fraud = (rng.random(n) < prob).astype(np.int64)

    fraud_idx = is_fraud == 1
    amount[fraud_idx] = amount[fraud_idx] * rng.uniform(1.5, 4.0, fraud_idx.sum())
    since_last[fraud_idx] = since_last[fraud_idx] * rng.uniform(0.02, 0.3, fraud_idx.sum())

    return pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:09d}" for i in range(n)],
            "user_id": uid.astype(np.int64),
            "event_timestamp": ts,
            "amount": amount.round(2),
            "hour": hour.astype(np.int64),
            "merchant_category": merchant_category,
            "is_foreign": is_foreign,
            "card_present": card_present,
            "device_change": device_change,
            "distance_from_home_km": distance.round(3),
            "seconds_since_last_txn": since_last.round(1),
            "is_fraud": is_fraud,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic and highly imbalanced transaction dataset")
    parser.add_argument("--n-transactions", type=int, default=500_000)
    parser.add_argument("--n-users", type=int, default=20_000)
    parser.add_argument("--fraud-rate", type=float, default=0.001)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift", action="store_true", help="Create a shifted batch for drift monitoring")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.drift:
        users = pd.read_parquet(USER_FEATURES_PATH)
        n = min(args.n_transactions, 20_000)
        batch = generate_transactions(users, n, args.fraud_rate, rng, START + pd.Timedelta(days=args.days), 1, True)
        batch.to_parquet(CURRENT_BATCH_PATH, index=False)
        print(f"Drifted batch: {len(batch):,} rows written to {CURRENT_BATCH_PATH}")
        return

    users = generate_users(args.n_users, rng)
    transactions = generate_transactions(users, args.n_transactions, args.fraud_rate, rng, START, args.days, False)
    users.to_parquet(USER_FEATURES_PATH, index=False)
    transactions.to_parquet(TRANSACTIONS_PATH, index=False)
    print(f"Users: {len(users):,} rows written to {USER_FEATURES_PATH}")
    print(f"Transactions: {len(transactions):,} rows, fraud rate {transactions['is_fraud'].mean():.4%}")


if __name__ == "__main__":
    main()
