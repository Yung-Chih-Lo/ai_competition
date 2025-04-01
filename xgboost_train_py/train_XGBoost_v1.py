import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
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

num_pos = (y_train == 1).sum()
num_neg = (y_train == 0).sum()
scale_pos_weight = num_neg / num_pos

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
    'eval_metric': 'logloss',
    'tree_method': 'hist',   # GPU 加速
    'device': 'cuda',
    'scale_pos_weight': scale_pos_weight  # 類別不平衡時可加上
}

num_rounds = 500  # 總訓練回合數，可自行調整

# 用來存放每個 round 的評估結果
evals_result = {}

# evals 列表指定要監控哪些資料集的評估指標
evals = [(dtrain, 'train'), (dval, 'eval')]

# 若想使用 early_stopping，需加參數 early_stopping_rounds
# 例如 early_stopping_rounds=20
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    evals=evals,
    evals_result=evals_result,
    early_stopping_rounds=20  # 如果需要早停可加上
)

# ======================
#     3. 模型評估
# ======================
# 在驗證集上進行預測
y_pred_proba = model.predict(dval)
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print("XGBoost GPU - 驗證集 Accuracy:", accuracy)
print("XGBoost GPU - 驗證集 F1 Score:", f1)

# ======================
# 4. 查看並繪製損失曲線
# ======================
# evals_result 會包含每個 round 在 train/eval 上的 logloss
train_logloss = evals_result['train']['logloss']
eval_logloss = evals_result['eval']['logloss']

# 繪製 logloss 曲線
plt.figure()
plt.plot(train_logloss, label='Train Logloss')
plt.plot(eval_logloss, label='Eval Logloss')
plt.xlabel('Rounds')
plt.ylabel('Logloss')
plt.title('XGBoost Training & Validation Logloss')
plt.legend()
plt.savefig("xgb_logloss_curve.png")  # 儲存圖檔
# plt.show()  # 如果你想要直接顯示圖表，可用 plt.show()

# ======================
# 5. 儲存損失數據為 CSV
# ======================
loss_df = pd.DataFrame({
    'round': range(len(train_logloss)),
    'train_logloss': train_logloss,
    'eval_logloss': eval_logloss
})
loss_df.to_csv("xgb_logloss.csv", index=False)

# ======================
# 6. 保存訓練好的模型
# ======================
# JSON 格式或者二進制都可；JSON 方便跨平台、跨語言
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
output_path = os.path.join("results", "test_predictions_v1.csv")
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("Test 預測已儲存至:", output_path)
