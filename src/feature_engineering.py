import dask.dataframe as dd
import pandas as pd
from pathlib import Path
from typing import List


def split_sequence(seq: str) -> List[str]:
    """
    Split a geohash sequence string into a list.
    """
    if not seq or pd.isna(seq):
        return []
    return seq.split(" ")


def explode_trip_into_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each trip (one row) into multiple transition rows.
    Operates on a pandas partition.
    """

    rows = []

    for _, row in df.iterrows():
        seq = split_sequence(row["geohash_sequence"])

        # Need at least 3 nodes for (prev, curr) -> next
        if len(seq) < 3:
            continue

        for i in range(len(seq) - 2):
            rows.append({
                "prev_node": seq[i],
                "curr_node": seq[i + 1],
                "hour": int(row["hour"]),
                "day_of_week": int(row["day_of_week"]),
                "call_type": row["CALL_TYPE"],
                "target_node": seq[i + 2]
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "prev_node",
                "curr_node",
                "hour",
                "day_of_week",
                "call_type",
                "target_node"
            ]
        )

    return pd.DataFrame(rows)


def build_feature_table(
    input_dir: str,
    output_dir: str,
    train_frac: float = 0.8,
    seed: int = 1312
):
    """
    Feature engineering pipeline:
    - read preprocessed trips
    - explode trips into transitions
    - split into train/test
    - save Parquet outputs
    """

    df = dd.read_parquet(input_dir)

    transitions = df.map_partitions(
        explode_trip_into_transitions,
        meta={
            "prev_node": "object",
            "curr_node": "object",
            "hour": "int64",
            "day_of_week": "int64",
            "call_type": "object",
            "target_node": "object"
        }
    )
    
    train_df, test_df = transitions.random_split(
        [train_frac, 1 - train_frac],
        random_state=seed
    )
    
    output_dir = Path(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_dir / "train", write_index=False)
    test_df.to_parquet(output_dir / "test", write_index=False)

    return {
        "train": str(output_dir / "train"),
        "test": str(output_dir / "test"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering: build transition table")
    parser.add_argument("--input", required=True, help="Input parquet (preprocessed trips)")
    parser.add_argument("--output", required=True, help="Output parquet (training features)")
    parser.add_argument("--train_frac", type=float, default=0.8)

    args = parser.parse_args()

    build_feature_table(
        input_dir=args.input,
        output_dir=args.output,
        train_frac=args.train_frac,
    )
