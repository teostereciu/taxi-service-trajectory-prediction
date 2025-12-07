from fastapi import FastAPI
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.model import NextNodeModel
from src.api.features import build_feature_row

app = FastAPI()

model = NextNodeModel()


@app.post("/predict_next_node")
def predict_next_node(req: PredictionRequest):
    features = req.dict()
    return model.predict_top_k(features, k=3)


@app.get("/health")
def health():
    return {"status": "ok"}