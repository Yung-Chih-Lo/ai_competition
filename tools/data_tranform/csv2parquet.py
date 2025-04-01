import pandas as pd
import os

output_parquet_dir = "datasets/parquet_datasets" # 建立新的 Parquet 資料夾
os.makedirs(output_parquet_dir, exist_ok=True)

# --- 處理 test.csv ---
csv_path_test = "datasets/origin/test.csv"
print(f"處理檔案: {csv_path_test}")
df_test = pd.read_csv(csv_path_test, encoding="utf-8")
parquet_path_test = os.path.join(output_parquet_dir, os.path.basename(csv_path_test).replace(".csv", ".parquet"))
print(f"儲存為 Parquet 格式: {parquet_path_test}")
# 改用 to_parquet
df_test.to_parquet(parquet_path_test, index=False) # 通常 Parquet 不儲存 Pandas index
del df_test

# --- 處理 train.csv 並分割 train/val ---
csv_path_train_full = "datasets/origin/train.csv"
csv_path_val_placeholder = "datasets/origin/val.csv" # 用來產生檔名
print(f"\n處理檔案: {csv_path_train_full}")
df_full_train = pd.read_csv(csv_path_train_full, encoding="utf-8")
print("原始訓練資料 '飆股' 標籤分佈:")
print(df_full_train["飆股"].value_counts())

print("\n分割訓練集與驗證集 (80/20)...")
df_train = df_full_train.sample(frac=0.8, random_state=42)
df_val = df_full_train.drop(df_train.index)
del df_full_train

print(f"\n訓練集筆數: {len(df_train)}")
print(f"驗證集筆數: {len(df_val)}")

parquet_path_train = os.path.join(output_parquet_dir, os.path.basename(csv_path_train_full).replace(".csv", ".parquet"))
parquet_path_val = os.path.join(output_parquet_dir, os.path.basename(csv_path_val_placeholder).replace(".csv", ".parquet"))

# --- 儲存 train.parquet ---
print(f"儲存為 Parquet 格式: {parquet_path_train}")
df_train.to_parquet(parquet_path_train, index=False)
del df_train

# --- 儲存 val.parquet ---
print(f"儲存為 Parquet 格式: {parquet_path_val}")
df_val.to_parquet(parquet_path_val, index=False)
del df_val

print("\n所有檔案已成功轉換並儲存為 Parquet 格式。")

# 注意：之後 Dask 讀取時要用 dd.read_parquet()
# train_ddf = dd.read_parquet(parquet_path_train)