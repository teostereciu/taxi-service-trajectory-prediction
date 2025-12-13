from fastapi import FastAPI, HTTPException
from prometheus_client import generate_latest
from starlette.responses import Response

from src.api.schemas import PredictionRequest
from src.api.model import NextNodeModel
from src.api.graph import TransitionGraph
from src.api.metrics import (
    REQUEST_COUNTER,
    ERROR_COUNTER,
    LATENCY_HISTOGRAM,
    NODE_PREDICTION_COUNTER,
    MODEL_ACCURACY,
    MODEL_TOP3_ACCURACY,
    MODEL_INFO,
    MODEL_CONFIDENCE,
    MODEL_ENTROPY
)

from src.config import PATHS

app = FastAPI()
model = NextNodeModel(model_path=PATHS["model"])
graph = TransitionGraph(path=PATHS["graph_artefact"])


@app.on_event("startup")
def load_model_metadata():
    bundle = model.bundle  

    MODEL_ACCURACY.set(bundle["metrics"]["accuracy"])
    MODEL_TOP3_ACCURACY.set(bundle["metrics"]["top3_accuracy"])
    
    MODEL_INFO.info({
        "version": bundle["metadata"]["version"],
        "model": "logistic_regression",
        "top_k": str(bundle["metadata"]["top_k"]),
    })


@app.post("/predict_next_node")
def predict_next_node(req: PredictionRequest):
    REQUEST_COUNTER.inc()

    with LATENCY_HISTOGRAM.time():
        try:
            results = model.predict_top_k(req.dict(), k=3)
            predictions = results["predictions"]
            
            MODEL_CONFIDENCE.observe(results["confidence"])
            MODEL_ENTROPY.observe(results["entropy"])
            
            for p in predictions:
                NODE_PREDICTION_COUNTER.labels(node=p["node"]).inc()

            return predictions

        except Exception:
            ERROR_COUNTER.inc()
            raise


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.get("/graph_node_info/{node}")
def graph_node_info(node: str):
    summary = graph.node_summary(node)
    
    if summary["degree"] == 0:
        raise HTTPException(status_code=404, detail="Node not found in graph")
    
    return summary


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": model.bundle["metadata"]["version"],
        "top_k": model.bundle["metadata"]["top_k"],
    }