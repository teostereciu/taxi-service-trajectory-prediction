import pandas as pd
from unittest.mock import patch, MagicMock

from src.graph_builder import build_transition_graph, build_and_save_graph

from src.config import PATHS


def test_build_transition_graph_counts_edges():
    df = pd.DataFrame({
        "curr_node": ["a", "a", "b"],
        "target_node": ["b", "b", "c"],
    })

    graph = build_transition_graph(df)

    assert graph == {
        "a": {"b": 2},
        "b": {"c": 1},
    }


@patch("src.graph_builder.dd.read_parquet")
def test_build_and_save_graph_writes_file(mock_read_parquet, tmp_path):
    fake_df = pd.DataFrame({
        "curr_node": ["a"],
        "target_node": ["b"],
    })

    mock_ddf = MagicMock()
    mock_ddf.compute.return_value = fake_df
    mock_read_parquet.return_value = mock_ddf

    output_path = tmp_path / "graph.json"

    build_and_save_graph(
        input_parquet=PATHS["features"],
        output_path=output_path,
    )

    assert output_path.exists()
