import pandas as pd

def downsample_train(df, label_col='飆股', random_state=42):
    """
    將訓練集 DataFrame 中 majority class (例如 0) 隨機下採樣到與 minority class (例如 1) 筆數相同
    """
    df_minority = df[df[label_col] == 1]
    df_majority = df[df[label_col] == 0]
    
    # 少數類別筆數
    n_minority = df_minority.shape[0]
    
    # 隨機抽取 majority 的資料
    df_majority_downsampled = df_majority.sample(n=n_minority, random_state=random_state)
    
    # 合併後打散順序
    df_balanced = pd.concat([df_majority_downsampled, df_minority], axis=0).sample(frac=1, random_state=random_state)
    
    return df_balanced


train_df = pd.read_hdf("datasets/origin/train.h5", key="data")
val_df = pd.read_hdf("datasets/origin/val.h5", key="data")

train_balanced = downsample_train(train_df)
print("Train 飆股分佈:")
print(train_balanced['飆股'].value_counts())

print("\nVal 飆股分佈:")
print(val_df['飆股'].value_counts())

# 如果需要儲存平衡後的 train 資料
train_balanced.to_hdf("datasets/balanced/train_balanced.h5", key="data", mode="w")
