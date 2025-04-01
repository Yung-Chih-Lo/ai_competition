import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

# 指定版本與輸出目錄
version = 3
output_dir = os.path.join("results", f"v{version}")
os.makedirs(output_dir, exist_ok=True)

# ======================
#     1. 讀取資料
# ======================
train_df = pd.read_hdf("datasets/origin/train.h5", key="data")
val_df = pd.read_hdf("datasets/origin/val.h5", key="data")

# 分離特徵與標籤
X_train = train_df.drop('飆股', axis=1)
y_train = train_df['飆股']

X_val = val_df.drop('飆股', axis=1)
y_val = val_df['飆股']

# 移除 'ID' 欄位（若存在）
if 'ID' in X_train.columns:
    X_train = X_train.drop('ID', axis=1)
if 'ID' in X_val.columns:
    X_val = X_val.drop('ID', axis=1)

# 建立 DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# ======================
#     2. 設定與訓練
# ======================
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',  # 使用 aucpr 作為評估指標
    'tree_method': 'hist',   # GPU 加速
    'device': 'cuda',
}

num_rounds = 500  # 訓練回合數

# 定義要試驗的 scale_pos_weight 值
scale_pos_weights = [50, 100, 150, 200]
metrics = {}

for scale_pos_weight in scale_pos_weights:
    print(f"Training with scale_pos_weight = {scale_pos_weight}")
    
    params['scale_pos_weight'] = scale_pos_weight
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
    
    # 在驗證集上進行預測
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 計算多項指標
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
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
    print(f"  ROC-AUC = {roc_auc:.4f}\n")

# 找出最佳的 scale_pos_weight（以 F1 score 為準）
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

# ======================
# 4. 查看並繪製損失曲線
# ======================
train_aucpr = evals_result['train']['aucpr']
eval_aucpr = evals_result['eval']['aucpr']

plt.figure()
plt.plot(train_aucpr, label='Train AUCPR')
plt.plot(eval_aucpr, label='Eval AUCPR')
plt.xlabel('Rounds')
plt.ylabel('AUCPR')
plt.title('XGBoost Training & Validation AUCPR')
plt.legend()
plt.savefig(os.path.join(output_dir, "xgb_aucpr_curve.png"))  # 儲存圖檔至 results/v3

loss_df = pd.DataFrame({
    'round': range(len(train_aucpr)),
    'train_aucpr': train_aucpr,
    'eval_aucpr': eval_aucpr
})
loss_df.to_csv(os.path.join(output_dir, "xgb_aucpr.csv"), index=False)

# ======================
# 5. 保存訓練好的模型
# ======================
model.save_model(os.path.join(output_dir, "xgb_model.json"))

# ======================
# 6. 讀取 test.h5 並產出預測檔
# ======================
test_df = pd.read_hdf("datasets/origin/test.h5", key="data")
test_id = test_df['ID'] if 'ID' in test_df.columns else None

X_test = test_df.drop(['飆股'], axis=1, errors='ignore')
if 'ID' in X_test.columns:
    X_test = X_test.drop('ID', axis=1)

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
