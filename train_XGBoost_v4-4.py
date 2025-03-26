import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

# ------------------------------
# 1. 版本設定與建立輸出目錄
# ------------------------------
version = "4-4"
output_dir = os.path.join("results", f"v{version}")
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 2. 讀取 PCA 後的資料
# ------------------------------
# 讀取經 PCA 降維後的訓練與驗證資料
train_df = pd.read_hdf("datasets/mean_pca_800/train_pca_800.h5", key="data")
val_df = pd.read_hdf("datasets/mean_pca_800/val_pca_800.h5", key="data")

# 分離特徵與標籤
X_train = train_df.drop('飆股', axis=1)
y_train = train_df['飆股']

X_val = val_df.drop('飆股', axis=1)
y_val = val_df['飆股']

# 若有 'ID' 欄位則移除（一般 PCA 版本不會有，但保險起見）
if 'ID' in X_train.columns:
    X_train = X_train.drop('ID', axis=1)
if 'ID' in X_val.columns:
    X_val = X_val.drop('ID', axis=1)

# 建立 DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# ------------------------------
# 3. 設定與訓練 XGBoost 模型
# ------------------------------
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',  # 使用 AUCPR 作為評估指標
    'tree_method': 'hist',   # GPU 加速
    'device': 'cuda',
}

num_rounds = 500  # 訓練回合數

evals_result = {}
evals = [(dtrain, 'train'), (dval, 'eval')]

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    evals=evals,
    evals_result=evals_result,
    early_stopping_rounds=20  # 若連續 20 輪無提升則停止
)

# ------------------------------
# 4. 模型評估（驗證集）
# ------------------------------
y_pred_proba = model.predict(dval)
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_proba)

print("\nFinal Model - 驗證集評估:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# ------------------------------
# 5. 查看並繪製損失曲線
# ------------------------------
train_aucpr = evals_result['train']['aucpr']
eval_aucpr = evals_result['eval']['aucpr']

plt.figure()
plt.plot(train_aucpr, label='Train AUCPR')
plt.plot(eval_aucpr, label='Eval AUCPR')
plt.xlabel('Rounds')
plt.ylabel('AUCPR')
plt.title('XGBoost Training & Validation AUCPR (PCA)')
plt.legend()
plt.savefig(os.path.join(output_dir, "xgb_aucpr_curve.png"))

loss_df = pd.DataFrame({
    'round': range(len(train_aucpr)),
    'train_aucpr': train_aucpr,
    'eval_aucpr': eval_aucpr
})
loss_df.to_csv(os.path.join(output_dir, "xgb_aucpr.csv"), index=False)

# ------------------------------
# 6. 儲存訓練好的模型
# ------------------------------
model.save_model(os.path.join(output_dir, "xgb_model.json"))

# ------------------------------
# 7. 讀取測試資料並產出預測檔
# ------------------------------
test_df = pd.read_hdf("datasets/mean_pca_800/test_pca_800.h5", key="data")
# 測試資料中保留 'ID' 欄位以便後續輸出
test_id = test_df['ID'] if 'ID' in test_df.columns else None

# 減去 'ID' 欄位（僅保留特徵），因為測試資料沒有標籤
X_test = test_df.drop(['ID'], axis=1, errors='ignore')

dtest = xgb.DMatrix(X_test)
y_test_proba = model.predict(dtest)
y_test_pred = (y_test_proba > 0.5).astype(int)

output_df = pd.DataFrame({
    'ID': test_id,
    '飆股': y_test_pred
})

output_path = os.path.join(output_dir, f"test_predictions_v{version}.csv")
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("Test 預測已儲存至:", output_path)
