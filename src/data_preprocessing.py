import json
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import geohash2 as geohash


def parse_polyline(polyline_str: str):
    """Parse POLYLINE string into list of [lon, lat] points."""
    try:
        if not polyline_str or polyline_str == "[]":
            return []
        return json.loads(polyline_str)
    except Exception:
        return []  # fail-safe: treat malformed as empty


def coords_to_geohashes(coords, precision=6):
    """Convert list of [lon, lat] points into geohashes."""
    if not coords:
        return []
    out = []
    for lon, lat in coords:   
        try:
            out.append(geohash.encode(lat, lon, precision=precision))
        except Exception:
            continue
    return out


def sequence_to_string(seq):
    """Convert geohash sequence to space-separated string."""
    if not seq:
        return ""
    return " ".join(seq)


def simplify_sequence(seq):
    """Remove consecutive duplicate geohashes."""
    if not seq:
        return []
    simplified = [seq[0]]
    for x in seq[1:]:
        if x != simplified[-1]:
            simplified.append(x)
    return simplified


def is_valid_trip(seq, min_length=2):
    """Return True if a trip has enough movement to be useful."""
    return seq is not None and len(seq) >= min_length


def extract_time_features(df: pd.DataFrame):
    """Extract hour and weekday from TIMESTAMP in a pandas partition."""
    dt = pd.to_datetime(df["TIMESTAMP"], unit="s")
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    return df


def preprocess_dataset(
    input_csv: str,
    output_parquet: str,
    geohash_precision: int = 6,
    min_trip_length: int = 2,
    blocksize: str = "default"
):
    """
    Full preprocessing pipeline:
    1. Load raw CSV with Dask
    2. Parse POLYLINE
    3. Convert coords to geohash
    4. Simplify sequences
    5. Filter invalid trips
    6. Extract time features
    7. Write Parquet output
    """

    df = dd.read_csv(
        input_csv,
        assume_missing=True,         
        dtype={
            "TRIP_ID": "object",
            "CALL_TYPE": "object",
            "ORIGIN_CALL": "float64",
            "ORIGIN_STAND": "float64",
            "TAXI_ID": "float64",
            "TIMESTAMP": "int64",
            "DAY_TYPE": "object",
            "MISSING_DATA": "object",
            "POLYLINE": "object"
        },
        blocksize=blocksize
    )

    df["coords"] = df["POLYLINE"].apply(parse_polyline, meta=("coords", "object"))

    df["geohash_sequence"] = df["coords"].apply(
        lambda seq: coords_to_geohashes(seq, precision=geohash_precision),
        meta=("geohash_sequence", "object")
    )

    df["geohash_sequence"] = df["geohash_sequence"].apply(
        simplify_sequence,
        meta=("geohash_sequence", "object")
    )

    df["is_valid"] = df["geohash_sequence"].apply(
        lambda seq: is_valid_trip(seq, min_trip_length),
        meta=("is_valid", "bool")
    )
    df = df[df["is_valid"]]

    df = df.map_partitions(extract_time_features)
    
    df["geohash_sequence_str"] = df["geohash_sequence"].apply(sequence_to_string, 
                                                              meta=("geohash_sequence_str", 
                                                                    "object")
                                                              )
    
    df = df.drop(columns=["geohash_sequence"])
    df = df.rename(columns={"geohash_sequence_str": "geohash_sequence"})
    
    split_quantile = 0.8
    split_ts = df["TIMESTAMP"].quantile(split_quantile).compute()
    df["split"] = df["TIMESTAMP"].apply(
        lambda ts: "train" if ts < split_ts else "test", 
        meta=("split", "object")
        )

    df_out = df[[
        "TRIP_ID",
        "CALL_TYPE",
        "TAXI_ID",
        "hour",
        "day_of_week",
        "geohash_sequence",
        "split"
    ]]

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    train_path = Path(output_parquet) / "train"
    test_path = Path(output_parquet) / "test"

    df_out[df_out["split"] == "train"].drop(columns=["split"]) \
        .to_parquet(train_path, write_index=False)

    df_out[df_out["split"] == "test"].drop(columns=["split"]) \
        .to_parquet(test_path, write_index=False)

    return output_parquet


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Taxi Dataset Preprocessing")
    parser.add_argument("--input", required=True, help="Path to raw CSV file")
    parser.add_argument("--output", required=True, help="Path for output Parquet")
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--min_trip_length", type=int, default=2)

    args = parser.parse_args()

    preprocess_dataset(
        input_csv=args.input,
        output_parquet=args.output,
        geohash_precision=args.precision,
        min_trip_length=args.min_trip_length
    )
