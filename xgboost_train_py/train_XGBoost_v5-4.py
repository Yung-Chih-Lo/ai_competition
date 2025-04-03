import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, auc
)
import time
import gc

# ------------------------------
# 1. 版本設定與建立輸出目錄
# ------------------------------
version = "5-4"  # 可自行調整版本號
output_dir = os.path.join("results", f"v{version}")
os.makedirs(output_dir, exist_ok=True)
print(f"輸出將儲存於: {output_dir}")

# ------------------------------
# 2. 讀取 Parquet 資料 & **最小**處理
# ------------------------------
start_time = time.time()
print("正在讀取資料...")
train_file = "datasets/parquet_datasets/train.parquet"
val_file = "datasets/parquet_datasets/val.parquet"
test_file = "datasets/parquet_datasets/test.parquet"

if not os.path.exists(train_file):
    raise FileNotFoundError(f"訓練檔案未找到: {train_file}")
if not os.path.exists(val_file):
    raise FileNotFoundError(f"驗證檔案未找到: {val_file}")
test_exists = os.path.exists(test_file)
if not test_exists:
    print(f"警告：測試檔案未找到: {test_file}.")

train_df = pd.read_parquet(train_file)
val_df = pd.read_parquet(val_file)
if test_exists:
    test_df = pd.read_parquet(test_file)
    test_id = test_df['ID'] if 'ID' in test_df.columns else None
else:
    test_df = None
    test_id = None

TARGET = '飆股'
X_train = train_df.drop(TARGET, axis=1)
y_train = train_df[TARGET]
X_val = val_df.drop(TARGET, axis=1)
y_val = val_df[TARGET]
if test_exists:
    X_test = test_df
else:
    X_test = None

del train_df, val_df, test_df
gc.collect()

# --- **僅**移除指定欄位 + ID ---
cols_to_drop_minimal = ['ID']
print(f"正在移除欄位: { [col for col in cols_to_drop_minimal if col in X_train.columns] }")
X_train = X_train.drop(columns=cols_to_drop_minimal, errors='ignore')
X_val = X_val.drop(columns=cols_to_drop_minimal, errors='ignore')
if X_test is not None:
    X_test = X_test.drop(columns=cols_to_drop_minimal, errors='ignore')


# ------------------------------
# 2.1 定義「想讓 XGBoost 當作類別」處理的欄位
# ------------------------------
# 例如：季IFRS財報_Z等級, 季IFRS財報_DPZ等級, 以及所有以 '券商代號' 結尾的欄位
categorical_features_original_list = (
    ['季IFRS財報_Z等級', '季IFRS財報_DPZ等級']
    + [col for col in X_train.columns if col.endswith('券商代號')]
)
categorical_features_existing = [col for col in categorical_features_original_list if col in X_train.columns]

print(f"定義的分類特徵共 {len(categorical_features_original_list)} 個，訓練集中實際存在 {len(categorical_features_existing)} 個。")

# --- F. 轉換分類特徵為 category dtype ---
#  這樣 XGBoost 才能在 enable_categorical=True 時把它們當作 category 分裂
for col in categorical_features_existing:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
    if col in X_val.columns:
        X_val[col] = X_val[col].astype('category')
    if X_test is not None and col in X_test.columns:
        X_test[col] = X_test[col].astype('category')

# ------------------------------
# 不做任何缺失值填補，不刪除高缺失欄位
# ------------------------------
print("\n最小處理：不進行缺失值填補或刪除高缺失欄位。")

# ------------------------------
# 檢查特徵一致性 (如果測試集存在)
# ------------------------------
if X_test is not None:
    if list(X_train.columns) != list(X_test.columns):
        print("警告：訓練集和測試集的欄位不完全一致！確保這是預期的。")
        # 此處示範：取交集確保至少可以跑，但實際上會丟失一些欄位
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        print(f"將只使用共同的 {len(common_cols)} 個欄位進行訓練和預測。")
        X_train = X_train[common_cols]
        X_val = X_val[common_cols]
        X_test = X_test[common_cols]

print(f"資料讀取與最小處理完成，耗時: {time.time() - start_time:.2f} 秒")
print(f"訓練特徵數量: {X_train.shape[1]}")
gc.collect()

# ------------------------------
# 3. 訓練 XGBoost 模型 (全特徵, 最小處理, GPU, enable_categorical=True)
# ------------------------------
start_time_train = time.time()
print("\n步驟 3: 訓練最小處理基準 XGBoost 模型...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"使用 scale_pos_weight: {scale_pos_weight:.2f}")

# 與之前版本類似，但開啟 enable_categorical=True
minimal_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    device='cuda',
    enable_categorical=True,  # 開啟類別處理
    early_stopping_rounds=50
)

print("開始訓練...")
eval_set = [(X_val, y_val)]
minimal_xgb.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=10
)

print(f"最小處理基準模型訓練完成，耗時: {time.time() - start_time_train:.2f} 秒")
best_iteration = (
    minimal_xgb.best_iteration 
    if hasattr(minimal_xgb, 'best_iteration') and minimal_xgb.best_iteration is not None
    else minimal_xgb.get_booster().best_iteration
)
print(f"最佳迭代次數 (n_estimators): {best_iteration}")

# ------------------------------
# 4. 在驗證集上評估模型
# ------------------------------
print("\n步驟 4: 在獨立驗證集上評估最小處理基準模型...")
y_pred_val_proba = minimal_xgb.predict_proba(X_val)[:, 1]

# 注意：這裡門檻硬寫了 0.1，如果只是想看標準 0.5，可以改成 (y_pred_val_proba > 0.5).astype(int)
y_pred_val = (y_pred_val_proba > 0.1).astype(int)  

accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val, pos_label=1)
precision_val = precision_score(y_val, y_pred_val, pos_label=1)
recall_val = recall_score(y_val, y_pred_val, pos_label=1)
try:
    roc_auc_val = roc_auc_score(y_val, y_pred_val_proba)
except ValueError:
    roc_auc_val = np.nan

precision_pr, recall_pr, _ = precision_recall_curve(y_val, y_pred_val_proba)
auc_pr_val = auc(recall_pr, precision_pr)

print("\n最小處理基準模型 - 驗證集評估:")
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
# 5. 儲存模型和結果
# ------------------------------
model_path = os.path.join(output_dir, f"xgb_model_v{version}.json")
minimal_xgb.save_model(model_path)
print(f"\n模型已儲存至: {model_path}")

results_summary = {
    "model_type": "XGBoost",
    "device": "cuda",
    "feature_selection": "All Features",
    "preprocessing": "Minimal (Drop ID, convert known categorical to `category`, no imputation)",
    "hyperparameters": minimal_xgb.get_params(),
    "best_iteration": best_iteration,
    "validation_f1": f1_val,
    "validation_precision": precision_val,
    "validation_recall": recall_val,
    "validation_roc_auc": roc_auc_val,
    "validation_auc_pr": auc_pr_val,
    "num_features_used": X_train.shape[1],
    "categorical_features_used_count": len(categorical_features_existing)
}
results_df = pd.DataFrame([results_summary])
results_path = os.path.join(output_dir, f"results_summary_v{version}.csv")
results_df.to_csv(results_path, index=False)
print(f"結果摘要已儲存至: {results_path}")

# ------------------------------
# 6. (可選) 測試集預測
# ------------------------------
if test_exists and X_test is not None:
    print("\n正在進行測試集預測...")

    # 先檢查欄位一致性
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("最終訓練集和測試集的特徵欄位不一致，請確認前面是否有做欄位交集或其他處理。")

    print(f"正在對 {X_test.shape[1]} 個特徵進行測試集預測...")
    y_test_proba = minimal_xgb.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba > 0.1).astype(int)

    if test_id is not None:
        output_df = pd.DataFrame({'ID': test_id, TARGET: y_test_pred})
        output_path = os.path.join(output_dir, f"test_predictions_v{version}.csv")
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Test 預測已儲存至: {output_path}")
    else:
        print("警告：測試資料中未找到 'ID' 欄位，無法儲存標準格式的預測檔。")
else:
    print("\n未找到測試檔案或測試集無法使用，跳過測試集預測。")

total_elapsed_time = time.time() - start_time
print(f"\n版本 v{version} 全部處理完成，總耗時: {total_elapsed_time:.2f} 秒")
