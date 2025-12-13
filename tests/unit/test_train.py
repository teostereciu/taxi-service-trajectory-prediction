import pandas as pd
from unittest.mock import patch, MagicMock

from src.train import load_data, train_model_from_paths

from src.config import PATHS


@patch("src.train.dd.read_parquet")
def test_load_data_returns_pandas_df(mock_read_parquet):
    fake_ddf = MagicMock()
    fake_ddf.compute.return_value = pd.DataFrame({"a": [1, 2, 3]})
    mock_read_parquet.return_value = fake_ddf

    df = load_data(PATHS["features"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


@patch("src.train.load_data")
@patch("src.train.train_model")
def test_train_model_from_paths_calls_train(mock_train, mock_load):
    mock_load.return_value = pd.DataFrame()
    mock_train.return_value = {"accuracy": 0.5}

    result = train_model_from_paths(
        train_path="train_path",
        test_path="test_path",
        model_dir="model_dir",
    )

    assert result["accuracy"] == 0.5
    assert mock_load.call_count == 2
