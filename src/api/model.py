from joblib import load
import numpy as np
import pandas as pd


class NextNodeModel:
    def __init__(self, model_path: str = "models/logreg.joblib"):
        self.pipeline = load(model_path)
        self.classes_ = self.pipeline.named_steps["model"].classes_

    def predict_top_k(self, features: dict, k: int = 3):
        """
        Predict top-k next nodes.

        features: dict with keys
          - prev_node
          - curr_node
          - hour
          - day_of_week
          - call_type
        """
        X = pd.DataFrame([features])

        probs = self.pipeline.predict_proba(X)[0]

        top_idx = np.argsort(probs)[::-1][:k]

        return [
            {
                "node": self.classes_[i],
                "probability": float(probs[i]),
            }
            for i in top_idx
        ]
