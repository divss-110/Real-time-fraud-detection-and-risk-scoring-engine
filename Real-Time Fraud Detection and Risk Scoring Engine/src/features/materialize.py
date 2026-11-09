import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
from feast import FeatureStore

from src.config import FEAST_CONFIG, FEAST_REPO, USER_FEATURES_PATH
from src.features.feast_repo import definitions


def get_store(config_name: str = FEAST_CONFIG) -> FeatureStore:
    return FeatureStore(repo_path=str(FEAST_REPO), fs_yaml_file=FEAST_REPO / config_name)


def apply_and_materialize(config_name: str = FEAST_CONFIG) -> FeatureStore:
    store = get_store(config_name)
    store.apply([definitions.user, definitions.user_features, definitions.fraud_scoring_v1])
    users = pd.read_parquet(USER_FEATURES_PATH, columns=["event_timestamp"])
    start = users["event_timestamp"].min().to_pydatetime() - timedelta(days=1)
    end = datetime.now(timezone.utc)
    store.materialize(start_date=start, end_date=end)
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Register Feast objects and load features into the online store")
    parser.add_argument("--config", default=FEAST_CONFIG, help="feature_store.yaml or feature_store.redis.yaml")
    args = parser.parse_args()
    store = apply_and_materialize(args.config)
    sample = store.get_online_features(
        features=["user_features:user_avg_amount_30d", "user_features:user_chargebacks_1y"],
        entity_rows=[{"user_id": 0}, {"user_id": 1}],
    ).to_dict()
    print(f"Online store ready using {args.config}")
    print(sample)


if __name__ == "__main__":
    main()
