import os
import pandas as pd

print("正在讀取資料...")
train_file = "datasets/parquet_datasets/train.parquet"
val_file = "datasets/parquet_datasets/val.parquet"
test_file = "datasets/parquet_datasets/test.parquet"

# 檢查檔案是否存在
if not os.path.exists(train_file):
    raise FileNotFoundError(f"訓練檔案未找到: {train_file}")
if not os.path.exists(val_file):
    raise FileNotFoundError(f"驗證檔案未找到: {val_file}")
if not os.path.exists(test_file):
    print(f"警告：測試檔案未找到: {test_file}.")

# 讀取資料
train_df = pd.read_parquet(train_file)
val_df = pd.read_parquet(val_file)

print(train_df.shape)
print(val_df.shape)

# 針對訓練資料計算每個欄位的缺失值數量
missing_train = train_df.isna().sum()         # 計算各欄位缺失數量
missing_train = missing_train[missing_train > 0]  # 篩選出有缺失值的欄位
missing_train_df = missing_train.reset_index()    # 轉換成 DataFrame 格式
missing_train_df.columns = ['欄位名稱', '缺失數量']

# 輸出訓練資料的缺失欄位資訊到 CSV
missing_train_df.to_csv("train_missing_columns.csv", index=False)
print("訓練資料的缺失欄位資訊已輸出到 train_missing_columns.csv")

# 針對驗證資料計算每個欄位的缺失值數量
missing_val = val_df.isna().sum()
missing_val = missing_val[missing_val > 0]
missing_val_df = missing_val.reset_index()
missing_val_df.columns = ['欄位名稱', '缺失數量']

# 輸出驗證資料的缺失欄位資訊到 CSV
missing_val_df.to_csv("val_missing_columns.csv", index=False)
print("驗證資料的缺失欄位資訊已輸出到 val_missing_columns.csv")
