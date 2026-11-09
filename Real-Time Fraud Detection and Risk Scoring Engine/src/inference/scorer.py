import json
import time

import lightgbm as lgb
import numpy as np

from src.config import METADATA_PATH, MODEL_PATH
from src.features.engineering import build_vector
from src.features.online_store import OnlineFeatureClient


class RiskScorer:
    def __init__(self, backend: str | None = None):
        self.metadata = json.loads(METADATA_PATH.read_text())
        self.booster = lgb.Booster(model_file=str(MODEL_PATH))
        self.features = self.metadata["features"]
        self.threshold = float(self.metadata["threshold"])
        self.review_threshold = float(self.metadata["review_threshold"])
        kwargs = {} if backend is None else {"backend": backend}
        self.feature_client = OnlineFeatureClient(self.metadata["feature_defaults"], **kwargs)

    def decide(self, score: float) -> tuple[str, str]:
        if score >= self.threshold:
            return "high", "decline"
        if score >= self.review_threshold:
            return "medium", "review"
        return "low", "approve"

    def score(self, txn: dict, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        user, found = self.feature_client.get(int(txn["user_id"]))
        t1 = time.perf_counter()
        vector = build_vector(txn, user)
        contrib = self.booster.predict(vector, pred_contrib=True)[0]
        t2 = time.perf_counter()

        base_value = float(contrib[-1])
        shap_values = contrib[:-1]
        score = float(1.0 / (1.0 + np.exp(-contrib.sum())))
        level, decision = self.decide(score)
        order = np.argsort(-np.abs(shap_values))[:top_k]
        reasons = [
            {
                "feature": self.features[i],
                "value": float(vector[0, i]),
                "shap": round(float(shap_values[i]), 5),
                "direction": "increases risk" if shap_values[i] > 0 else "decreases risk",
            }
            for i in order
        ]
        t3 = time.perf_counter()
        return {
            "transaction_id": txn["transaction_id"],
            "risk_score": round(score, 6),
            "risk_level": level,
            "decision": decision,
            "threshold": round(self.threshold, 6),
            "base_value": round(base_value, 6),
            "top_reasons": reasons,
            "shap_values": {name: round(float(v), 6) for name, v in zip(self.features, shap_values)},
            "user_found": found,
            "feature_source": self.feature_client.backend,
            "model_version": self.metadata["model_version"],
            "latency_ms": {
                "features": round((t1 - t0) * 1000, 3),
                "model_and_shap": round((t2 - t1) * 1000, 3),
                "total": round((t3 - t0) * 1000, 3),
            },
        }
