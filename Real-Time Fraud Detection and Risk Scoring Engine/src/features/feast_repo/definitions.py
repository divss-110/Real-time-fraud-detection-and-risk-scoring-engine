from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

user = Entity(
    name="user",
    join_keys=["user_id"],
    value_type=ValueType.INT64,
    description="Card holder that makes transactions",
)

user_features_source = FileSource(
    name="user_features_source",
    path="../../../data/processed/user_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=3650),
    online=True,
    source=user_features_source,
    schema=[
        Field(name="user_avg_amount_30d", dtype=Float64),
        Field(name="user_std_amount_30d", dtype=Float64),
        Field(name="user_txn_count_7d", dtype=Int64),
        Field(name="user_account_age_days", dtype=Int64),
        Field(name="user_chargebacks_1y", dtype=Int64),
        Field(name="user_distinct_merchants_30d", dtype=Int64),
        Field(name="user_night_txn_ratio", dtype=Float64),
    ],
    tags={"team": "risk", "domain": "fraud"},
)

fraud_scoring_v1 = FeatureService(name="fraud_scoring_v1", features=[user_features])
