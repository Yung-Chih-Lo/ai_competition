import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import os

# 1. 載入已訓練的模型
model = xgb.Booster()
model.load_model("results/v4/xgb_model.json")

# 2. 載入驗證資料集，用於尋找最佳閾值
val_df = pd.read_hdf("datasets/origin/val.h5", key="data")
X_val = val_df.drop('飆股', axis=1)
y_val = val_df['飆股']

# 如果存在 'ID' 欄位，將其移除
if 'ID' in X_val.columns:
    X_val = X_val.drop('ID', axis=1)

dval = xgb.DMatrix(X_val)
y_val_proba = model.predict(dval)

# 3. 嘗試不同的預測概率閾值並評估
thresholds = np.arange(0.1, 0.95, 0.05)  # 從 0.1 到 0.9，步長為 0.05
metrics = {}

# 建立資料夾存放閾值測試結果
os.makedirs("threshold_results", exist_ok=True)

for threshold in thresholds:
    # 根據閾值進行預測
    y_pred = (y_val_proba > threshold).astype(int)
    
    # 計算各項指標
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    # 儲存指標
    metrics[threshold] = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    print(f"閾值 {threshold:.2f}:")
    print(f"  準確率 = {accuracy:.4f}")
    print(f"  F1 分數 = {f1:.4f}")
    print(f"  精確率 = {precision:.4f}")
    print(f"  召回率 = {recall:.4f}")

# 4. 找到 F1 分數最高的閾值
best_threshold_f1 = max(metrics, key=lambda k: metrics[k]['f1'])
best_metrics_f1 = metrics[best_threshold_f1]

print(f"\n基於 F1 分數的最佳閾值: {best_threshold_f1:.2f}")
print(f"最佳指標: 準確率 = {best_metrics_f1['accuracy']:.4f}, F1 分數 = {best_metrics_f1['f1']:.4f}, " 
      f"精確率 = {best_metrics_f1['precision']:.4f}, 召回率 = {best_metrics_f1['recall']:.4f}")

# 5. 繪製各閾值下的指標變化圖
thresholds_list = list(metrics.keys())
metrics_df = pd.DataFrame({
    '閾值': thresholds_list,
    '準確率': [metrics[t]['accuracy'] for t in thresholds_list],
    'F1 分數': [metrics[t]['f1'] for t in thresholds_list],
    '精確率': [metrics[t]['precision'] for t in thresholds_list],
    '召回率': [metrics[t]['recall'] for t in thresholds_list]
})

# 儲存指標資料
metrics_df.to_csv("threshold_results/threshold_metrics.csv", index=False)

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.plot(thresholds_list, metrics_df['準確率'], 'o-', label='準確率')
plt.plot(thresholds_list, metrics_df['F1 分數'], 'o-', label='F1 分數')
plt.plot(thresholds_list, metrics_df['精確率'], 'o-', label='精確率')
plt.plot(thresholds_list, metrics_df['召回率'], 'o-', label='召回率')

plt.axvline(x=best_threshold_f1, color='r', linestyle='--', label=f'最佳 F1 閾值 ({best_threshold_f1:.2f})')

plt.xlabel('預測概率閾值')
plt.ylabel('指標值')
plt.title('不同閾值下的評估指標')
plt.legend()
plt.grid(True)
plt.savefig("threshold_results/threshold_metrics.png")

# 6. 使用不同閾值對測試資料進行預測
test_df = pd.read_hdf("datasets/origin/val.h5", key="data")
test_id = test_df['ID'] if 'ID' in test_df.columns else None

X_test = test_df.drop(['飆股'], axis=1, errors='ignore')
if 'ID' in X_test.columns:
    X_test = X_test.drop('ID', axis=1)

dtest = xgb.DMatrix(X_test)
y_test_proba = model.predict(dtest)

# 儲存每個樣本的預測概率
proba_df = pd.DataFrame({
    'ID': test_id,
    '飆股概率': y_test_proba
})
proba_df.to_csv("threshold_results/test_probabilities.csv", index=False)

# 使用不同閾值產生預測結果
thresholds_to_use = [0.5, best_threshold_f1]  # 標準閾值和最佳 F1 閾值
for threshold in thresholds_to_use:
    y_test_pred = (y_test_proba > threshold).astype(int)
    
    # 將結果與 ID 合併輸出
    output_df = pd.DataFrame({
        'ID': test_id,
        '飆股': y_test_pred
    })
    
    # 儲存成 csv
    output_path = f"threshold_results/test_predictions_threshold_{threshold:.2f}.csv"
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"測試集使用閾值 {threshold:.2f} 的預測已儲存至:", output_path)

print("\n所有閾值的評估結果和預測已完成！")