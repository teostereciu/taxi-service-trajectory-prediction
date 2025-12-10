from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from src.data_preprocessing import preprocess_dataset
from src.feature_engineering import build_feature_table
from src.train import train_model_from_paths
from src.graph_builder import build_and_save_graph

PROJECT_DIR = "/app"

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="taxi_next_node_training",
    default_args=default_args,
    catchup=False,
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess_trips",
        python_callable=preprocess_dataset,
        op_kwargs={
            "input_csv": f"{PROJECT_DIR}/data/raw/Porto_taxi_data_test_partial_trajectories.csv",
            "output_parquet": f"{PROJECT_DIR}/data/preprocessed/trips",
        },
    )

    build_features = PythonOperator(
        task_id="build_transition_features",
        python_callable=build_feature_table,
        op_kwargs={
            "input_dir": f"{PROJECT_DIR}/data/preprocessed/trips",
            "output_dir": f"{PROJECT_DIR}/data/transitions",
        },
    )

    build_graph = PythonOperator(
        task_id="build_graph",
        python_callable=build_and_save_graph,
        op_kwargs={
            "input_parquet": f"{PROJECT_DIR}/data/transitions/train",
            "output_path": f"{PROJECT_DIR}/artefacts/train_graph.json",
        },
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model_from_paths,
        op_kwargs={
            "train_path": f"{PROJECT_DIR}/data/transitions/train",
            "test_path": f"{PROJECT_DIR}/data/transitions/test",
            "model_dir": f"{PROJECT_DIR}/models",
        },
    )

    preprocess >> build_features >> build_graph >> train
