import pandas as pd

from src.data_preprocessing import (
    parse_polyline,
    coords_to_geohashes,
    simplify_sequence,
    is_valid_trip,
    extract_time_features,
)

def test_parse_polyline_valid():
    s = "[[1.0, 2.0], [3.0, 4.0]]"
    assert parse_polyline(s) == [[1.0, 2.0], [3.0, 4.0]]


def test_parse_polyline_invalid_returns_empty():
    assert parse_polyline("not-json") == []
    assert parse_polyline("") == []


def test_coords_to_geohashes_produces_strings():
    coords = [[-8.61, 41.15], [-8.62, 41.16]]
    hashes = coords_to_geohashes(coords, precision=5)

    assert len(hashes) == 2
    assert all(isinstance(h, str) for h in hashes)


def test_is_valid_trip_min_length():
    assert is_valid_trip(["a", "b"], min_length=2)
    assert not is_valid_trip(["a"], min_length=2)


def test_extract_time_features_adds_columns():
    df = pd.DataFrame({"TIMESTAMP": [1609459200]})  # 2021-01-01 00:00 UTC
    out = extract_time_features(df)

    assert "hour" in out.columns
    assert "day_of_week" in out.columns
    assert out.loc[0, "hour"] == 0
