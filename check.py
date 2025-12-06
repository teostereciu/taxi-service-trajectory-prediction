import dask.dataframe as dd

df = dd.read_parquet("data/features/train")
print(df.head())
