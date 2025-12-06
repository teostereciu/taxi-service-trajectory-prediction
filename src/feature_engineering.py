import dask.dataframe as dd
import pandas as pd
from pathlib import Path


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sequence-based features.
    Operates on a pandas partition.
    """

    seq_col = df["geohash_sequence"].fillna("")

    seqs = seq_col.str.split(" ")

    # Length of sequence
    df["seq_len"] = seqs.apply(lambda x: len(x) if x != [""] else 0)

    # Unique geohashes
    df["unique_geos"] = seqs.apply(
        lambda x: len(set(x)) if x != [""] else 0
    )

    # Repetition ratio
    df["repeat_ratio"] = df.apply(
        lambda row: row["unique_geos"] / row["seq_len"]
        if row["seq_len"] > 0 else 0.0,
        axis=1
    )

    # Start / end geohash
    df["start_geohash"] = seqs.apply(
        lambda x: x[0] if x and x != [""] else None
    )
    df["end_geohash"] = seqs.apply(
        lambda x: x[-1] if x and x != [""] else None
    )

    return df


def engineer_features(
    input_parquet: str,
    output_parquet: str
):
    """
    Feature engineering pipeline:
    - load preprocessed parquet
    - add sequence-level features
    - write new parquet
    """

    df = dd.read_parquet(input_parquet)

    df = df.map_partitions(add_sequence_features)

    Path(output_parquet).mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, write_index=False)

    return output_parquet


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")

    args = parser.parse_args()

    engineer_features(
        input_parquet=args.input,
        output_parquet=args.output
    )
