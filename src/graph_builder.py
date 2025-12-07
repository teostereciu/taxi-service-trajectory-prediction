import json
import dask.dataframe as dd
from pathlib import Path
from collections import defaultdict


def build_transition_graph(df):
    """
    Build a transition count graph:
    curr_node -> {next_node: count}
    """
    graph = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        src = row["curr_node"]
        dst = row["target_node"]
        graph[src][dst] += 1

    return {k: dict(v) for k, v in graph.items()}


def build_and_save_graph(input_parquet: str, output_path: str):
    df = dd.read_parquet(input_parquet).compute()

    graph = build_transition_graph(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(graph, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build transition graph")
    parser.add_argument("--input", required=True, help="Transitions parquet path")
    parser.add_argument("--output", required=True, help="Output graph.json path")

    args = parser.parse_args()

    build_and_save_graph(
        input_parquet=args.input,
        output_path=args.output
    )
