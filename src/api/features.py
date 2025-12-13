import numpy as np
import pandas as pd


def time_bucket(hour: int) -> str:
    if 5 <= hour < 10:
        return "morning"
    elif 10 <= hour < 16:
        return "afternoon"
    elif 16 <= hour < 22:
        return "evening"
    else:
        return "night"


def build_serving_features(
    prev_node: str,
    curr_node: str,
    call_type: str,
    hour: int,
    day_of_week: int,
    step_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Build a single-row feature frame compatible with the trained model.
    Mirrors the training-time feature columns.
    """
    return pd.DataFrame([{
        "prev_node": prev_node,
        "curr_node": curr_node,
        "call_type": call_type,
        "time_bucket": time_bucket(int(hour)),

        "hour_sin": float(np.sin(2 * np.pi * int(hour) / 24)),
        "hour_cos": float(np.cos(2 * np.pi * int(hour) / 24)),
        "dow_sin": float(np.sin(2 * np.pi * int(day_of_week) / 7)),
        "dow_cos": float(np.cos(2 * np.pi * int(day_of_week) / 7)),

        "is_weekend": int(int(day_of_week) >= 5),
        "step_frac": float(step_frac),
    }])