from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.scorer import RiskScorer


class Transaction(BaseModel):
    transaction_id: str = Field(examples=["txn_000000001"])
    user_id: int = Field(ge=0, examples=[42])
    amount: float = Field(gt=0, examples=[129.99])
    hour: int = Field(ge=0, le=23, examples=[2])
    merchant_category: int = Field(ge=0, le=19, examples=[7])
    is_foreign: int = Field(ge=0, le=1, examples=[1])
    card_present: int = Field(ge=0, le=1, examples=[0])
    device_change: int = Field(ge=0, le=1, examples=[1])
    distance_from_home_km: float = Field(ge=0, examples=[350.0])
    seconds_since_last_txn: float = Field(ge=0, examples=[45.0])


class Reason(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    risk_level: str
    decision: str
    threshold: float
    base_value: float
    top_reasons: list[Reason]
    shap_values: dict[str, float]
    user_found: bool
    feature_source: str
    model_version: str
    latency_ms: dict[str, float]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    scorer = RiskScorer()
    scorer.score(
        {
            "transaction_id": "warmup",
            "user_id": 0,
            "amount": 10.0,
            "hour": 12,
            "merchant_category": 1,
            "is_foreign": 0,
            "card_present": 1,
            "device_change": 0,
            "distance_from_home_km": 1.0,
            "seconds_since_last_txn": 3600.0,
        }
    )
    state["scorer"] = scorer
    yield
    state.clear()


app = FastAPI(title="Real-Time Fraud Detection and Risk Scoring Engine", version="1.0.0", lifespan=lifespan)


def get_scorer() -> RiskScorer:
    scorer = state.get("scorer")
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return scorer


@app.get("/health")
def health() -> dict:
    scorer = get_scorer()
    return {"status": "ok", "model_version": scorer.metadata["model_version"], "feature_source": scorer.feature_client.backend}


@app.get("/model-info")
def model_info() -> dict:
    scorer = get_scorer()
    meta = scorer.metadata
    return {
        "model_version": meta["model_version"],
        "features": meta["features"],
        "threshold": meta["threshold"],
        "review_threshold": meta["review_threshold"],
        "metrics": meta["metrics"],
        "mlflow_run_id": meta.get("mlflow_run_id"),
    }


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> dict:
    return get_scorer().score(txn.model_dump())


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(txns: list[Transaction]) -> list[dict]:
    if len(txns) > 1000:
        raise HTTPException(status_code=413, detail="Batch limit is 1000 transactions")
    scorer = get_scorer()
    return [scorer.score(t.model_dump()) for t in txns]
