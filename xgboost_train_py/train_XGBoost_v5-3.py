import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer # 用於填補中位數
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, auc
)
import time
import gc # 引入垃圾回收模組

# ------------------------------
# 1. 版本設定與建立輸出目錄
# ------------------------------
version = "5-3"  # 更新版本號
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

del train_df, val_df, test_df # 嘗試釋放記憶體
gc.collect()

# --- 移除指定欄位 + ID ---
cols_to_drop_manual = ['季IFRS財報_財務信評', 'ID']
print(f"正在手動移除欄位: {[col for col in cols_to_drop_manual if col in X_train.columns]}")
X_train = X_train.drop(columns=cols_to_drop_manual, errors='ignore')
X_val = X_val.drop(columns=cols_to_drop_manual, errors='ignore')
if X_test is not None:
    X_test = X_test.drop(columns=cols_to_drop_manual, errors='ignore')

# ------------------------------
# 2.1 *** 新增：處理缺失值 ***
# ------------------------------
print("\n步驟 2.1: 正在處理缺失值...")

# --- A. 刪除缺失率極高的欄位 ---
cols_to_drop_high_missing = [
    '日外資_外資自營商買張', '日外資_外資自營商賣張', '日外資_外資自營商買賣超',
    '月營收_預估年營收(千)', '月營收_累計營收達成率(%)', '月營收_重要子公司本月營業收入淨額(千)', '月營收_重要子公司本年累計營收淨額(千)',
    '季IFRS財報_稅額扣抵比率(%)', '季IFRS財報_預計稅額扣抵比率(%)',
    '日投信_投信買均價', '日投信_投信賣均價',  # 缺失 > 90%
    '日自營_自營商買均價',  # 缺失 > 60%
    '日自營_自營商賣均價'   # 缺失 > 60%
]
print(f"正在刪除高缺失欄位...")
cols_to_drop_high_missing_exist = [col for col in cols_to_drop_high_missing if col in X_train.columns]
print(f"  將刪除: {cols_to_drop_high_missing_exist}")
X_train = X_train.drop(columns=cols_to_drop_high_missing_exist, errors='ignore')
X_val = X_val.drop(columns=cols_to_drop_high_missing_exist, errors='ignore')
if X_test is not None:
    X_test = X_test.drop(columns=cols_to_drop_high_missing_exist, errors='ignore')
gc.collect()  # 刪除後清理內存

# --- B. 用 0 填補特定欄位 ---
cols_to_fill_zero = [col for col in X_train.columns if
                     '買賣力' in col or
                     '長期負債' in col or
                     '長期資金' in col or
                     '長期負債對淨值' in col or
                     '短期借款' in col or
                     '研發費用比率' in col or
                     '利息費用對營業收入比率' in col
                    ]
ranked_broker_numeric_cols = [
    col for col in X_train.columns
    if ('買超第' in col or '賣超第' in col) and not col.endswith('券商代號') and not col.endswith('均價')
]
cols_to_fill_zero.extend(ranked_broker_numeric_cols)
cols_to_fill_zero.extend([
    '官股券商_張增減', '官股券商_金額增減(千)', '官股券商_買張', '官股券商_賣張', '官股券商_庫存',
    '官股券商_買金額(千)', '官股券商_賣金額(千)', '官股券商_買筆數', '官股券商_賣筆數'
])
cols_to_fill_zero = list(set([col for col in cols_to_fill_zero if col in X_train.columns]))

if cols_to_fill_zero:
    print(f"正在對 {len(cols_to_fill_zero)} 個欄位填補 0...")
    X_train[cols_to_fill_zero] = X_train[cols_to_fill_zero].fillna(0)
    X_val[cols_to_fill_zero] = X_val[cols_to_fill_zero].fillna(0)
    if X_test is not None:
        X_test[cols_to_fill_zero] = X_test[cols_to_fill_zero].fillna(0)

# --- C. 特殊填補利息保障倍數 ---
cols_interest_coverage = ['季IFRS財報_利息保障倍數(倍)', '季IFRS財報_利息保障倍數累季(倍)']
large_value = 999999
for col in cols_interest_coverage:
    if col in X_train.columns:
        print(f"正在對欄位 '{col}' 的 NaN 填補極大值 {large_value}...")
        X_train[col] = X_train[col].fillna(large_value)
        X_val[col] = X_val[col].fillna(large_value)
        if X_test is not None and col in X_test.columns:
            X_test[col] = X_test[col].fillna(large_value)

# --- D. 用中位數填補剩餘的數值欄位 NaN ---
print("準備用中位數填補剩餘數值欄位...")
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
remaining_missing_numeric_cols = [col for col in numeric_cols if X_train[col].isnull().sum() > 0]
median_imputer = None  # 初始化
if remaining_missing_numeric_cols:
    print(f"正在對 {len(remaining_missing_numeric_cols)} 個數值欄位用中位數填補...")
    median_imputer = SimpleImputer(strategy='median')
    X_train[remaining_missing_numeric_cols] = median_imputer.fit_transform(X_train[remaining_missing_numeric_cols])
    X_val[remaining_missing_numeric_cols] = median_imputer.transform(X_val[remaining_missing_numeric_cols])
    if X_test is not None:
        test_cols_to_impute = [col for col in remaining_missing_numeric_cols if col in X_test.columns]
        if test_cols_to_impute:
            X_test[test_cols_to_impute] = median_imputer.transform(X_test[test_cols_to_impute])
else:
    print("沒有剩餘的數值欄位需要中位數填補。")

# --- E. 處理分類特徵的 NaN (在轉換 dtype 前) ---
categorical_features_original_list = ['季IFRS財報_Z等級', '季IFRS財報_DPZ等級'] + [
    col for col in X_train.columns if col.endswith('券商代號')
]
categorical_features_existing = [col for col in categorical_features_original_list if col in X_train.columns]
missing_in_categoricals = {
    col: X_train[col].isnull().sum()
    for col in categorical_features_existing if X_train[col].isnull().any()
}

if missing_in_categoricals:
    print(f"\n找到以下分類欄位存在缺失值: {missing_in_categoricals}")
    print("將用 'MISSING' 字符串填補...")
    for col in missing_in_categoricals.keys():
        fill_value = 'MISSING'
        X_train[col] = X_train[col].fillna(fill_value)
        X_val[col] = X_val[col].fillna(fill_value)
        if X_test is not None and col in X_test.columns:
            X_test[col] = X_test[col].fillna(fill_value)
else:
    print("\n定義的分類特徵中沒有缺失值。")

# --- F. 最後檢查 ---
print("\n缺失值處理後檢查:")
print("Train NaN count:", X_train.isnull().sum().sum())
print("Val NaN count:", X_val.isnull().sum().sum())
if X_test is not None:
    print("Test NaN count:", X_test.isnull().sum().sum())
if X_train.isnull().sum().sum() > 0:
    print(f"警告：訓練集仍有 {X_train.isnull().sum().sum()} 個缺失值！")
if X_val.isnull().sum().sum() > 0:
    print(f"警告：驗證集仍有 {X_val.isnull().sum().sum()} 個缺失值！")

# ------------------------------
# 2.3 轉換分類特徵 dtype
# ------------------------------
if categorical_features_existing:
    print(f"\n步驟 2.3: 找到 {len(categorical_features_existing)} 個分類特徵，正在轉換 dtype...")
    mem_before = X_train.memory_usage(deep=True).sum() / 1024**2
    train_categories = {}  # 儲存訓練集的 categories
    for col in categorical_features_existing:
        X_train[col] = X_train[col].astype('category')
        train_categories[col] = X_train[col].cat.categories
        if col in X_val.columns:
            try:
                # 使用訓練集的 categories 來轉換驗證集
                X_val[col] = pd.Categorical(X_val[col], categories=train_categories[col], ordered=False)
                # 驗證集若有新類別 -> 將變成 NaN
                if X_val[col].isnull().any():
                    print(f"警告: 欄位 '{col}' 在驗證集中包含訓練集未出現的類別，已轉換為 NaN。")
            except Exception as e:
                print(f"警告: 欄位 '{col}' 在驗證集轉換 category 時出錯: {e}。")

    mem_after = X_train.memory_usage(deep=True).sum() / 1024**2
    print(f"分類特徵 dtype 轉換完成。記憶體使用: {mem_before:.2f} MB -> {mem_after:.2f} MB")
else:
    print("警告: 未指定或未找到任何要轉換為 category 的特徵。")

print(f"資料讀取與預處理完成，耗時: {time.time() - start_time:.2f} 秒")
gc.collect()

# ------------------------------
# 3. 快速初步訓練以獲取特徵重要性 (使用 XGBoost + CUDA)
# ------------------------------
start_time_fs = time.time()
print("\n步驟 3: 使用 XGBoost 快速訓練以獲取特徵重要性...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

temp_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    device='cuda',
    enable_categorical=True
)
print("實際傳遞給初步訓練的特徵數量:", X_train.shape[1])
temp_xgb.fit(X_train, y_train, verbose=False)

importances = temp_xgb.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

importance_path = os.path.join(output_dir, f"feature_importance_v{version}.csv")
feature_importance_df.to_csv(importance_path, index=False)
print(f"特徵重要性列表已儲存至: {importance_path}")
print(f"獲取特徵重要性完成，耗時: {time.time() - start_time_fs:.2f} 秒")

del temp_xgb
gc.collect()

# ==============================================================
# === 下面「原本」是第 4 步：根據累積重要度做特徵選擇，現在整段拿掉或註解 ===
# === 改成「不進行篩選，直接使用全部補值後的特徵」                   ===
# ==============================================================

# ------------------------------
# 5. 訓練單一 XGBoost 模型 (使用 **全特徵** 和 GPU)
# ------------------------------
start_time_train = time.time()
print("\n步驟 5: 訓練單一 XGBoost 模型 (全特徵)...")

single_xgb = xgb.XGBClassifier(
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
    enable_categorical=True,
    early_stopping_rounds=50,  # 直接指定 Early Stopping 回合數
)

print("開始訓練...")
eval_set = [(X_val, y_val)]
single_xgb.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=10
)

print(f"單一模型訓練完成，耗時: {time.time() - start_time_train:.2f} 秒")

# ------------------------------
# 6. 在驗證集上評估模型
# ------------------------------
print("\n步驟 6: 在獨立驗證集上評估模型...")
y_pred_val_proba = single_xgb.predict_proba(X_val)[:, 1]
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
model_path = os.path.join(output_dir, f"xgb_model_v{version}.json")
single_xgb.save_model(model_path)
print(f"\n模型已儲存至: {model_path}")

results_summary = {
    "model_type": "XGBoost",
    "device": "cuda",
    "feature_selection": "None (used all features after imputation)",
    "imputation_strategy": "Drop high missing, Zero fill, Median fill",
    "hyperparameters": single_xgb.get_params(),
    "best_iteration": single_xgb.best_iteration if hasattr(single_xgb, 'best_iteration') else 'N/A',
    "validation_f1": f1_val,
    "validation_precision": precision_val,
    "validation_recall": recall_val,
    "validation_roc_auc": roc_auc_val,
    "validation_auc_pr": auc_pr_val,
    "num_features_used": X_train.shape[1]  # 全特徵數
}
results_df = pd.DataFrame([results_summary])
results_path = os.path.join(output_dir, f"results_summary_v{version}.csv")
results_df.to_csv(results_path, index=False)
print(f"結果摘要已儲存至: {results_path}")

# ------------------------------
# 8. (可選) 測試集預測 (同樣使用全特徵)
# ------------------------------
if test_exists:
    print("\n正在處理測試集...")
    # 這裡因為我們在前面(步驟 2)已經對 X_test 做過手動刪除高缺失欄位 & ID，
    # 並填補過指定欄位(若有需要)，故只要再次做以下處理即可。

    # （若上面已經在全域 X_test 處理過，可以省略重新讀入）
    # test_df = pd.read_parquet(test_file)
    # test_id = test_df['ID'] if 'ID' in test_df.columns else None
    # X_test = test_df.drop(columns=cols_to_drop_manual, errors='ignore')

    # 再做一次與訓練相同的 NaN 補植流程 (若上面確實沒有做，就要在這裡做):
    if cols_to_fill_zero:
        X_test[cols_to_fill_zero] = X_test[cols_to_fill_zero].fillna(0)
    for col in cols_interest_coverage:
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(large_value)
    if median_imputer and remaining_missing_numeric_cols:
        test_cols_to_impute = [col for col in remaining_missing_numeric_cols if col in X_test.columns]
        if test_cols_to_impute:
            X_test[test_cols_to_impute] = median_imputer.transform(X_test[test_cols_to_impute])
    if missing_in_categoricals:
        for col in missing_in_categoricals.keys():
            if col in X_test.columns:
                X_test[col] = X_test[col].fillna('MISSING')

    # 分類特徵轉型 (對照訓練集 categories)
    if categorical_features_existing:
        print(f"轉換測試集的分類特徵 dtype 並對齊 categories...")
        for col in categorical_features_existing:
            if col in X_test.columns:
                try:
                    X_test[col] = pd.Categorical(
                        X_test[col],
                        categories=train_categories[col],
                        ordered=False
                    )
                    if X_test[col].isnull().any():
                        print(f"警告: 欄位 '{col}' 在測試集中包含訓練集未出現的類別，已轉換為 NaN。")
                except Exception as e:
                    print(f"處理測試集欄位 '{col}' 時發生錯誤: {e}.")

    # 直接用完整欄位去預測
    print("正在進行測試集預測...")
    y_test_proba = single_xgb.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba > 0.1).astype(int)

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