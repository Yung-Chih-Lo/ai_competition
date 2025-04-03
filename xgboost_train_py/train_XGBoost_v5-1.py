import os
import pandas as pd
import numpy as np
import xgboost as xgb # 改回導入 xgboost
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, auc
)
import time

# ------------------------------
# 1. 版本設定與建立輸出目錄
# ------------------------------
version = "6" # 更新版本號，表明使用 XGBoost + CUDA
output_dir = os.path.join("results", f"v{version}")
os.makedirs(output_dir, exist_ok=True)
print(f"輸出將儲存於: {output_dir}")

# ------------------------------
# 2. 讀取 Parquet 資料 & 基礎處理
# ------------------------------
start_time = time.time()
print("正在讀取資料...")
train_file = "datasets/parquet_datasets/train.parquet"
val_file = "datasets/parquet_datasets/val.parquet"
test_file = "datasets/parquet_datasets/test.parquet"

# ... (省略檔案存在性檢查) ...
if not os.path.exists(train_file): raise FileNotFoundError(f"訓練檔案未找到: {train_file}")
if not os.path.exists(val_file): raise FileNotFoundError(f"驗證檔案未找到: {val_file}")
if not os.path.exists(test_file): print(f"警告：測試檔案未找到: {test_file}.")


train_df = pd.read_parquet(train_file)
val_df = pd.read_parquet(val_file)

TARGET = '飆股'
X_train = train_df.drop(TARGET, axis=1)
y_train = train_df[TARGET]
X_val = val_df.drop(TARGET, axis=1)
y_val = val_df[TARGET]

# 移除指定欄位 + ID
cols_to_drop = ['日外資_與前日異動原因', '財務信評', 'ID']
print(f"正在移除欄位: { [col for col in cols_to_drop if col in X_train.columns] }")
X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
X_val = X_val.drop(columns=cols_to_drop, errors='ignore')

# ------------------------------
# 2.2 定義並轉換分類特徵
# ------------------------------
cols_to_categorize_explicit = ['季IFRS財報_Z等級', '季IFRS財報_DPZ等級']
cols_to_categorize_suffix = [col for col in X_train.columns if col.endswith('券商代號')]
potential_categorical_features = cols_to_categorize_explicit + cols_to_categorize_suffix
categorical_features = [col for col in potential_categorical_features if col in X_train.columns]

if categorical_features:
    print(f"找到 {len(categorical_features)} 個分類特徵，正在轉換 dtype...")
    mem_before = X_train.memory_usage(deep=True).sum() / 1024**2
    for col in categorical_features:
        X_train[col] = X_train[col].astype('category')
        if col in X_val.columns:
             X_val[col] = X_val[col].astype('category')
             try:
                 X_val[col] = pd.Categorical(X_val[col], categories=X_train[col].cat.categories, ordered=False)
             except ValueError:
                 print(f"警告: 欄位 '{col}' 在驗證集中的類別與訓練集不完全匹配。")
    mem_after = X_train.memory_usage(deep=True).sum() / 1024**2
    print(f"分類特徵 dtype 轉換完成。記憶體使用: {mem_before:.2f} MB -> {mem_after:.2f} MB")
else:
    print("警告: 未指定任何分類特徵。")

# 檢查特徵一致性
if list(X_train.columns) != list(X_val.columns):
    raise ValueError("處理後訓練集和驗證集的特徵欄位不一致！")

print(f"資料讀取與預處理完成，耗時: {time.time() - start_time:.2f} 秒")


# ------------------------------
# 3. 快速初步訓練以獲取特徵重要性 (使用 XGBoost + CUDA)
# ------------------------------
start_time_fs = time.time()
print("\n步驟 3: 使用 XGBoost 快速訓練以獲取特徵重要性...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 使用 XGBoost Classifier
temp_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr', # 評估指標
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1, # CPU 核數用於非 GPU 部分
    tree_method='hist', # 使用 hist 算法
    device='cuda', # *** 指定使用 GPU ***
    enable_categorical=True # *** 啟用原生分類處理 ***
    # 可以設定 n_estimators 和 learning_rate, 但這裡主要為了獲取重要性
)

print("開始初步訓練 (僅用於獲取重要性)...")
# XGBoost 的 fit 不需要 categorical_feature 參數, 它依賴 enable_categorical=True 和 dtype
temp_xgb.fit(X_train, y_train, verbose=False) # verbose=False 避免過多輸出

importances = temp_xgb.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

# *** 新增：儲存特徵重要性列表 ***
importance_path = os.path.join(output_dir, f"feature_importance_v{version}.csv")
feature_importance_df.to_csv(importance_path, index=False)
print(f"特徵重要性列表已儲存至: {importance_path}")

print(f"獲取特徵重要性完成，耗時: {time.time() - start_time_fs:.2f} 秒")


# ------------------------------
# 4. 選擇重要特徵
# ------------------------------
cumulative_importance_threshold = 0.99
feature_importance_df['cumulative_importance_ratio'] = feature_importance_df['importance'].cumsum() / feature_importance_df['importance'].sum()
selected_features = feature_importance_df[feature_importance_df['cumulative_importance_ratio'] <= cumulative_importance_threshold]['feature'].tolist()

# 防止選擇過少或過多的特徵 (可選)
MIN_FEATURES = 50
MAX_FEATURES = 2000 # 可以設定一個上限
if len(selected_features) < MIN_FEATURES:
    print(f"警告: 基於 99% 閾值選出的特徵少於 {MIN_FEATURES} 個，將使用 Top {MIN_FEATURES}。")
    selected_features = feature_importance_df['feature'].head(MIN_FEATURES).tolist()
elif len(selected_features) > MAX_FEATURES:
    print(f"警告: 基於 99% 閾值選出的特徵多於 {MAX_FEATURES} 個，將使用 Top {MAX_FEATURES}。")
    selected_features = feature_importance_df['feature'].head(MAX_FEATURES).tolist()

print(f"\n步驟 4: 已選擇 Top {len(selected_features)} 個特徵。")

X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]

selected_categorical_features = [col for col in categorical_features if col in selected_features]
if selected_categorical_features:
    print(f"選定的特徵中包含 {len(selected_categorical_features)} 個分類特徵。")


# ------------------------------
# 5. 訓練單一 XGBoost 模型 (使用選定特徵和 GPU)
# ------------------------------
start_time_train = time.time()
print("\n步驟 5: 訓練單一 XGBoost 模型...")

# 使用相對合理的固定參數
single_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr', # 評估指標
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    n_estimators=1000,         # 較大數值，讓 early stopping 決定
    max_depth=7,               # XGBoost 常用 max_depth 控制複雜度
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,                 # 最小分裂增益
    reg_alpha=0.1,             # L1
    reg_lambda=1.0,            # L2 (XGBoost 預設為 1)
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    device='cuda',             # *** 指定使用 GPU ***
    enable_categorical=True,    # *** 啟用原生分類處理 ***
    early_stopping_rounds=50,
)

print("開始訓練...")
# 設定 Early Stopping Callback
# 注意：XGBoost 的 eval_metric 會自動使用 eval_set 中的第一個數據集 (此處是驗證集)

eval_set = [(X_val_selected, y_val)] # 驗證集用於 Early Stopping

single_xgb.fit(
    X_train_selected,
    y_train,
    eval_set=eval_set,
    verbose=1 # 每 50 輪打印一次評估結果
)

print(f"單一模型訓練完成，耗時: {time.time() - start_time_train:.2f} 秒")
# 如果 save_best=True, best_iteration 可能不准確或不需要，模型已是最佳狀態
# print(f"最佳迭代次數 (n_estimators): {single_xgb.best_iteration}")


# ------------------------------
# 6. 在驗證集上評估模型
# ------------------------------
print("\n步驟 6: 在獨立驗證集上評估模型...")
# 使用訓練好的模型直接預測 (如果 save_best=True, 它已是最佳模型)
y_pred_val_proba = single_xgb.predict_proba(X_val_selected)[:, 1]
y_pred_val = (y_pred_val_proba > 0.5).astype(int)

# ... (評估指標計算和打印與之前相同) ...
accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val, pos_label=1)
precision_val = precision_score(y_val, y_pred_val, pos_label=1)
recall_val = recall_score(y_val, y_pred_val, pos_label=1)
try:
    roc_auc_val = roc_auc_score(y_val, y_pred_val_proba)
except ValueError:
    roc_auc_val = np.nan # Handle cases with only one class present in y_true
precision_pr, recall_pr, _ = precision_recall_curve(y_val, y_pred_val_proba)
auc_pr_val = auc(recall_pr, precision_pr)

print("\n單一模型 - 驗證集評估:")
print(f"Accuracy: {accuracy_val:.4f}")
print(f"F1 Score: {f1_val:.4f}")
print(f"Precision: {precision_val:.4f}")
print(f"Recall: {recall_val:.4f}")
print(f"ROC-AUC: {roc_auc_val:.4f}")
print(f"AUC-PR: {auc_pr_val:.4f}")
print("\n分類報告:")
print(classification_report(y_val, y_pred_val))
print("混淆矩陣:")
print(confusion_matrix(y_val, y_pred_val))


# ------------------------------
# 7. 儲存模型和結果
# ------------------------------
# XGBoost 模型儲存
model_path = os.path.join(output_dir, f"xgb_model_v{version}.json") # 或 .ubj
single_xgb.save_model(model_path)
print(f"\n模型已儲存至: {model_path}")

# 儲存結果摘要
results_summary = {
    "model_type": "XGBoost",
    "device": "cuda",
    "feature_selection": f"Top {len(selected_features)} features (from initial run)",
    "hyperparameters": single_xgb.get_params(),
    "best_iteration": single_xgb.best_iteration if hasattr(single_xgb, 'best_iteration') else 'N/A (save_best=True used)', # best_iteration 可能不存在或不準確
    "validation_f1": f1_val,
    "validation_precision": precision_val,
    "validation_recall": recall_val,
    "validation_roc_auc": roc_auc_val,
    "validation_auc_pr": auc_pr_val,
    "num_features_used": len(selected_features),
    "categorical_features_used_count": len(selected_categorical_features)
}
results_df = pd.DataFrame([results_summary])
results_path = os.path.join(output_dir, f"results_summary_v{version}.csv")
results_df.to_csv(results_path, index=False)
print(f"結果摘要已儲存至: {results_path}")


# ------------------------------
# 8. (可選) 測試集預測
# ------------------------------
if os.path.exists(test_file):
    print("\n正在處理測試集...")
    test_df = pd.read_parquet(test_file)
    test_id = test_df['ID'] if 'ID' in test_df.columns else None
    X_test = test_df
    X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

    # **重要**: 只選擇訓練時使用的特徵
    # 先檢查測試集是否包含所有需要的特徵
    missing_cols = [col for col in selected_features if col not in X_test.columns]
    if missing_cols:
        raise ValueError(f"測試集缺少以下必要的特徵: {missing_cols}")
    X_test_selected = X_test[selected_features]

    # **重要**: 對測試集應用相同的分類特徵轉換並對齊 categories
    if selected_categorical_features:
        print(f"轉換測試集選定特徵的 dtype 並對齊 categories...")
        for col in selected_categorical_features:
             if col in X_test_selected.columns:
                try:
                    X_test_selected[col] = pd.Categorical(X_test_selected[col], categories=X_train[col].cat.categories, ordered=False)
                except Exception as e:
                    print(f"處理測試集欄位 '{col}' 時發生錯誤: {e}.")
                    # 對於測試集中的新類別，轉換後會變為 NaN，XGBoost 需要能處理 NaN (通常可以)
             else:
                 print(f"警告：分類特徵 '{col}' 不在測試集中。")

    # 再次檢查特徵是否一致
    if list(X_train_selected.columns) != list(X_test_selected.columns):
         raise ValueError("處理後訓練集和測試集的特徵子集欄位不一致！")

    print("正在進行測試集預測...")
    y_test_proba = single_xgb.predict_proba(X_test_selected)[:, 1]
    y_test_pred = (y_test_proba > 0.5).astype(int)

    if test_id is not None:
         output_df = pd.DataFrame({'ID': test_id, TARGET: y_test_pred})
         output_path = os.path.join(output_dir, f"test_predictions_v{version}.csv")
         output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
         print(f"Test 預測已儲存至: {output_path}")
    else:
         print("警告：測試資料中未找到 'ID' 欄位，無法儲存標準格式的預測檔。")
else:
    print("\n未找到測試檔案，跳過測試集預測。")

total_elapsed_time = time.time() - start_time
print(f"\n版本 v{version} 全部處理完成，總耗時: {total_elapsed_time:.2f} 秒")