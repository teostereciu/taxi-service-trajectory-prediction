from fastapi import FastAPI
from prometheus_client import generate_latest
from starlette.responses import Response

from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.model import NextNodeModel
from src.api.features import build_feature_row
from src.api.metrics import (
    REQUEST_COUNTER,
    ERROR_COUNTER,
    LATENCY_HISTOGRAM,
    NODE_PREDICTION_COUNTER,
)

app = FastAPI()

model = NextNodeModel()


@app.post("/predict_next_node")
def predict_next_node(req: PredictionRequest):
    start = time.time()
    REQUEST_COUNTER.inc()

    try:
        predictions = model.predict_top_k(req.dict(), k=3)

        for p in predictions:
            NODE_PREDICTION_COUNTER.labels(node=p["node"]).inc()

        return predictions

    except Exception:
        ERROR_COUNTER.inc()
        raise

    finally:
        LATENCY_HISTOGRAM.observe(time.time() - start)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.get("/health")
def health():
    return {"status": "ok"}