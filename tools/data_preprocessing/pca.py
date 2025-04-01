import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# 1. 載入資料
train_df = pd.read_hdf("datasets/origin/train.h5", key="data")
val_df = pd.read_hdf("datasets/origin/val.h5", key="data")
test_df = pd.read_hdf("datasets/origin/test.h5", key="data")

# 2. 定義數值特徵
features = [col for col in train_df.columns if col not in ['ID', '飆股']]
# 測試集沒有標籤，只需排除 'ID'
test_features = [col for col in test_df.columns if col not in ['ID']]

# 3. 以均值補植缺值
imputer = SimpleImputer(strategy="mean")
train_imputed = imputer.fit_transform(train_df[features])
val_imputed = imputer.transform(val_df[features])
test_imputed = imputer.transform(test_df[test_features])

# 4. 資料標準化
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_imputed)
val_scaled = scaler.transform(val_imputed)
test_scaled = scaler.transform(test_imputed)

# 5. PCA 降維到 800 維
pca = PCA(n_components=800, random_state=42)
train_pca = pca.fit_transform(train_scaled)
val_pca = pca.transform(val_scaled)
test_pca = pca.transform(test_scaled)

# 6. 使用 pandas to_hdf 儲存 PCA 結果和標籤
# 對於訓練和驗證資料，包含標籤

if not os.path.exists("datasets/mean_pca_800"):
    os.makedirs("datasets/mean_pca_800", exist_ok=True)

train_pca_df = pd.DataFrame(train_pca)
train_pca_df['飆股'] = train_df['飆股'].values
train_pca_df.to_hdf("datasets/mean_pca_800/train_pca_800.h5", key="data", mode="w", format='table')

val_pca_df = pd.DataFrame(val_pca)
val_pca_df['飆股'] = val_df['飆股'].values
val_pca_df.to_hdf("datasets/mean_pca_800/val_pca_800.h5", key="data", mode="w", format='table')

# 對於測試資料，沒有標籤
test_pca_df = pd.DataFrame(test_pca)
test_pca_df['ID'] = test_df['ID'].values # 保留 ID
test_pca_df.to_hdf("datasets/mean_pca_800/test_pca_800.h5", key="data", mode="w", format='table')

# 儲存 PCA 模型為 pickle (因為 HDF5 不適合儲存 sklearn 模型)
pd.to_pickle(pca, "datasets/mean_pca_800/pca_model.pkl")

# 輸出相關資訊
print("PCA 處理完成並已儲存為 HDF5 格式")
print("PCA 結果的 shape:")
print(f"訓練資料: {train_pca.shape}")
print(f"驗證資料: {val_pca.shape}")
print(f"測試資料: {test_pca.shape}")
print("PCA 結果的 explained variance ratio:")
print(pca.explained_variance_ratio_)
print("累積 explained variance ratio:")
print(pca.explained_variance_ratio_.cumsum())