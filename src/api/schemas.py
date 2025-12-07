from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    prev_node: str
    curr_node: str
    hour: int
    day_of_week: int
    call_type: str


class NodePrediction(BaseModel):
    node: str
    probability: float


class PredictionResponse(BaseModel):
    top_predictions: List[NodePrediction]
