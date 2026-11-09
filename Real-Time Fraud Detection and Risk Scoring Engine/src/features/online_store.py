import logging
import os

import pandas as pd

from src.config import ENTITY_KEY, FEAST_CONFIG, FEAST_REPO, FEATURE_BACKEND, FEATURE_VIEW, USER_FEATURES, USER_FEATURES_PATH

logger = logging.getLogger(__name__)


class OnlineFeatureClient:
    def __init__(self, defaults: dict, backend: str | None = None, config_name: str | None = None):
        backend = backend or os.getenv("FEATURE_BACKEND", FEATURE_BACKEND)
        config_name = config_name or os.getenv("FEAST_CONFIG", FEAST_CONFIG)
        self.defaults = {name: float(defaults[name]) for name in USER_FEATURES}
        self.refs = [f"{FEATURE_VIEW}:{name}" for name in USER_FEATURES]
        self.backend = backend
        self.store = None
        self.table = {}
        if backend == "feast":
            try:
                from feast import FeatureStore

                self.store = FeatureStore(repo_path=str(FEAST_REPO), fs_yaml_file=FEAST_REPO / config_name)
                self.store.get_online_features(features=self.refs, entity_rows=[{ENTITY_KEY: 0}])
            except Exception as exc:
                logger.warning("Feast online store unavailable, falling back to local table: %s", exc)
                self.store = None
                self.backend = "local"
        if self.store is None:
            self._load_local_table()

    def _load_local_table(self) -> None:
        users = pd.read_parquet(USER_FEATURES_PATH, columns=[ENTITY_KEY] + USER_FEATURES)
        self.table = users.set_index(ENTITY_KEY).astype(float).to_dict("index")

    def get(self, user_id: int) -> tuple[dict, bool]:
        if self.store is not None:
            result = self.store.get_online_features(features=self.refs, entity_rows=[{ENTITY_KEY: user_id}]).to_dict()
            values = {name: result[name][0] for name in USER_FEATURES}
            if all(value is None for value in values.values()):
                return dict(self.defaults), False
            return {name: float(self.defaults[name] if value is None else value) for name, value in values.items()}, True
        row = self.table.get(user_id)
        if row is None:
            return dict(self.defaults), False
        return row, True
