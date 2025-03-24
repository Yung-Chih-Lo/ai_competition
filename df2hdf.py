import pandas as pd

csv_path = "ori_datasets/test.csv"
df = pd.read_csv(csv_path, encoding="utf-8")
h5_path = csv_path.replace(".csv", ".h5")
df.to_hdf(h5_path, key="data", mode="w")
del df

csv_path_train = "ori_datasets/train.csv"
csv_path_val = "ori_datasets/val.csv"
df = pd.read_csv(csv_path_train, encoding="utf-8")
print(df["飆股"].value_counts())
df_train = df.sample(frac=0.8, random_state=42)
df_val = df.drop(df_train.index)
h5_path_train = csv_path_train.replace(".csv", ".h5")
h5_path_val = csv_path_val.replace(".csv", ".h5")
df_train.to_hdf(h5_path_train, key="data", mode="w")
df_val.to_hdf(h5_path_val, key="data", mode="w")