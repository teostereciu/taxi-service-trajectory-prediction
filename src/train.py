import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, top_k_accuracy_score


def load_data(path: str) -> pd.DataFrame:
    """
    Load Dask parquet and materialize as pandas DataFrame.
    """
    df = dd.read_parquet(path)
    return df.compute()


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, model_dir: str):
    """
    Train baseline logistic regression model.
    """

    X_train = train_df.drop(columns=["target_node"])
    y_train = train_df["target_node"]

    X_test = test_df.drop(columns=["target_node"])
    y_test = test_df["target_node"]

    categorical_feats = ["prev_node", "curr_node", "call_type"]
    numeric_feats = ["hour", "day_of_week"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_feats),
            ("num", "passthrough", numeric_feats)
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    classes = pipeline.named_steps["model"].classes_
    
    # Filter test set to known classes
    mask = y_test.isin(classes)
    X_test = X_test[mask]
    y_test = y_test[mask]
    y_pred = y_pred[mask]
    y_proba = y_proba[mask]

    acc = accuracy_score(y_test, y_pred)
    top3 = top_k_accuracy_score(
        y_test,
        y_proba,
        k=3,
        labels=classes
    )

    print(f"Accuracy:     {acc:.4f}")
    print(f"Top-3 Acc:    {top3:.4f}")

    # --- Save artifacts ---
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_dir / "logreg.joblib")

    return {
        "accuracy": acc,
        "top3_accuracy": top3
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train next-node prediction model")
    parser.add_argument("--train", required=True, help="Train parquet path")
    parser.add_argument("--test", required=True, help="Test parquet path")
    parser.add_argument("--model_dir", default="models", help="Output model directory")

    args = parser.parse_args()

    train_df = load_data(args.train)
    test_df = load_data(args.test)

    train_model(
        train_df=train_df,
        test_df=test_df,
        model_dir=args.model_dir
    )
