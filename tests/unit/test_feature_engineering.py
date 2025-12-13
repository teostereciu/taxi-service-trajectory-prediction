import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.feature_engineering import (
    split_sequence,
    time_bucket,
    explode_trip_into_transitions,
    build_feature_table,
)

from src.config import FEATURE_ENG, PATHS

def test_split_sequence_basic():
    assert split_sequence("a b c") == ["a", "b", "c"]


def test_split_sequence_empty_or_nan():
    assert split_sequence("") == []
    assert split_sequence(None) == []


def test_time_bucket_values():
    assert time_bucket(6) == "morning"
    assert time_bucket(12) == "afternoon"
    assert time_bucket(18) == "evening"
    assert time_bucket(2) == "night"


def test_explode_trip_into_transitions_basic():
    df = pd.DataFrame({
        "geohash_sequence": ["a b c d"],
        "hour": [10],
        "day_of_week": [2],
        "CALL_TYPE": ["A"],
    })

    out = explode_trip_into_transitions(df)

    # For sequence length 4 -> 2 transitions
    assert len(out) == 2

    assert set(out.columns) == {
        "prev_node",
        "curr_node",
        "call_type",
        "target_node",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "time_bucket",
        "is_weekend",
        "step_frac",
    }

    assert out.iloc[0]["prev_node"] == "a"
    assert out.iloc[0]["curr_node"] == "b"
    assert out.iloc[0]["target_node"] == "c"


def test_explode_trip_skips_short_sequences():
    df = pd.DataFrame({
        "geohash_sequence": ["a b"],
        "hour": [10],
        "day_of_week": [2],
        "CALL_TYPE": ["A"],
    })

    out = explode_trip_into_transitions(df)
    assert out.empty


@patch("src.feature_engineering.dd.read_parquet")
def test_build_feature_table_returns_paths(mock_read_parquet, tmp_path):
    mock_ddf = MagicMock()
    mock_transitions = MagicMock()

    mock_ddf.map_partitions.return_value = mock_transitions
    mock_transitions.random_split.return_value = (MagicMock(), MagicMock())

    mock_read_parquet.return_value = mock_ddf

    result = build_feature_table(
        input_dir=PATHS["preprocessed"],
        output_dir=tmp_path,
        train_frac=FEATURE_ENG["train_frac"],
        seed=FEATURE_ENG["seed"],
    )

    assert "train" in result
    assert "test" in result
    assert str(tmp_path / "train") == result["train"]
    assert str(tmp_path / "test") == result["test"]
