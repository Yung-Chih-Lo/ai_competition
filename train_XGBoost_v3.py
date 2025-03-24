import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import os

# ======================
#     1. 讀取資料
# ======================
train_df = pd.read_hdf("ori_datasets/train.h5", key="data")
val_df = pd.read_hdf("ori_datasets/val.h5", key="data")

# 分離特徵與標籤
X_train = train_df.drop('飆股', axis=1)
y_train = train_df['飆股']

X_val = val_df.drop('飆股', axis=1)
y_val = val_df['飆股']

# 如果存在 'ID' 欄位，將其移除（ID 不參與訓練）
if 'ID' in X_train.columns:
    X_train = X_train.drop('ID', axis=1)
if 'ID' in X_val.columns:
    X_val = X_val.drop('ID', axis=1)

# 創建 DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# ======================
#     2. 設定與訓練
# ======================
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',  # 使用 aucpr
    'tree_method': 'hist',   # GPU 加速
    'device': 'cuda',
}

num_rounds = 500  # 總訓練回合數，可自行調整

# 定義要試驗的 scale_pos_weight 值
scale_pos_weights = [50, 100, 150, 200]

# 用來存放每個 scale_pos_weight 的評估指標
metrics = {}

for scale_pos_weight in scale_pos_weights:
    print(f"Training with scale_pos_weight = {scale_pos_weight}")
    
    # 更新 params 中的 scale_pos_weight
    params['scale_pos_weight'] = scale_pos_weight
    
    # 用來存放每個 round 的評估結果
    evals_result = {}
    
    # evals 列表指定要監控哪些資料集的評估指標
    evals = [(dtrain, 'train'), (dval, 'eval')]
    
    # 訓練模型
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=20  # 若連續 20 輪無提升則停止
    )
    
    # 在驗證集上進行預測
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 計算多項指標
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)  # 使用概率計算 ROC-AUC
    
    # 儲存指標
    metrics[scale_pos_weight] = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }
    
    print(f"scale_pos_weight = {scale_pos_weight}:")
    print(f"  Accuracy = {accuracy:.4f}")
    print(f"  F1 Score = {f1:.4f}")
    print(f"  Precision = {precision:.4f}")
    print(f"  Recall = {recall:.4f}")
    print(f"  ROC-AUC = {roc_auc:.4f}")

# 找到最佳的 scale_pos_weight（以 F1 score 為基準）
best_scale_pos_weight = max(metrics, key=lambda k: metrics[k]['f1'])
best_metrics = metrics[best_scale_pos_weight]
print(f"\nBest scale_pos_weight: {best_scale_pos_weight}")
print(f"Best Metrics: Accuracy = {best_metrics['accuracy']:.4f}, F1 Score = {best_metrics['f1']:.4f}, "
      f"Precision = {best_metrics['precision']:.4f}, Recall = {best_metrics['recall']:.4f}, "
      f"ROC-AUC = {best_metrics['roc_auc']:.4f}")

# 使用最佳的 scale_pos_weight 重新訓練最終模型
params['scale_pos_weight'] = best_scale_pos_weight
evals_result = {}
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    evals=evals,
    evals_result=evals_result,
    early_stopping_rounds=20
)

# ======================
#     3. 模型評估
# ======================
# 在驗證集上進行預測
y_pred_proba = model.predict(dval)
y_pred = (y_pred_proba > 0.5).astype(int)

# 計算最終模型的多項指標
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

# ======================
# 4. 查看並繪製損失曲線
# ======================
train_aucpr = evals_result['train']['aucpr']
eval_aucpr = evals_result['eval']['aucpr']

# 繪製 aucpr 曲線
plt.figure()
plt.plot(train_aucpr, label='Train AUCPR')
plt.plot(eval_aucpr, label='Eval AUCPR')
plt.xlabel('Rounds')
plt.ylabel('AUCPR')
plt.title('XGBoost Training & Validation AUCPR')
plt.legend()
plt.savefig("xgb_aucpr_curve.png")  # 儲存圖檔

# ======================
# 5. 儲存損失數據為 CSV
# ======================
loss_df = pd.DataFrame({
    'round': range(len(train_aucpr)),
    'train_aucpr': train_aucpr,
    'eval_aucpr': eval_aucpr
})
loss_df.to_csv("xgb_aucpr.csv", index=False)

# ======================
# 6. 保存訓練好的模型
# ======================
model.save_model("xgb_model.json")

# ======================
# 7. 讀取 test.h5 並產出預測檔
# ======================
test_df = pd.read_hdf("ori_datasets/test.h5", key="data")

# 先記錄下 test 的 ID 以便最後輸出
test_id = test_df['ID'] if 'ID' in test_df.columns else None

# 如果要預測，特徵不含飆股（因為這是未知的標籤）
X_test = test_df.drop(['飆股'], axis=1, errors='ignore')

# 同樣，模型不需要 ID 欄位
if 'ID' in X_test.columns:
    X_test = X_test.drop('ID', axis=1)

dtest = xgb.DMatrix(X_test)
y_test_proba = model.predict(dtest)
y_test_pred = (y_test_proba > 0.5).astype(int)

# 將結果與 ID 合併輸出
output_df = pd.DataFrame({
    'ID': test_id,
    '飆股': y_test_pred
})

# 建立輸出資料夾（如果尚未存在）
os.makedirs("results", exist_ok=True)

# 儲存成 csv
output_path = os.path.join("results", "test_predictions_v3.csv")
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("Test 預測已儲存至:", output_path)