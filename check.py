import pandas as pd

train = pd.read_parquet("data/transitions.parquet/train")
test = pd.read_parquet("data/transitions.parquet/test")

print(len(train), len(test))
print(train.columns)
print(train.head())
