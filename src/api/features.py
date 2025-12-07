import pandas as pd
from datetime import datetime


def build_feature_row(
    prev_node: str,
    curr_node: str,
    timestamp: int,
    call_type: str
) -> pd.DataFrame:
    dt = datetime.utcfromtimestamp(timestamp)

    return pd.DataFrame([{
        "prev_node": prev_node,
        "curr_node": curr_node,
        "hour": dt.hour,
        "day_of_week": dt.weekday(),
        "call_type": call_type
    }])
