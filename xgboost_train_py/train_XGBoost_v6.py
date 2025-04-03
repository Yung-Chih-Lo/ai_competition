import os
import gc
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler  # 可以改成你想用的 sampler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, auc
)

# ------------------------------
# 1. 版本設定與建立輸出目錄
# ------------------------------
version = "6"
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

del train_df, val_df, test_df
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

# A. 刪除缺失率極高的欄位
cols_to_drop_high_missing = [
    '日外資_外資自營商買張', '日外資_外資自營商賣張', '日外資_外資自營商買賣超',
    '月營收_預估年營收(千)', '月營收_累計營收達成率(%)', '月營收_重要子公司本月營業收入淨額(千)', '月營收_重要子公司本年累計營收淨額(千)',
    '季IFRS財報_稅額扣抵比率(%)', '季IFRS財報_預計稅額扣抵比率(%)',
    '日投信_投信買均價', '日投信_投信賣均價',  # 缺失 > 90%
    '日自營_自營商買均價',                    # 缺失 > 60%
    '日自營_自營商賣均價'                    # 缺失 > 60%
]
print(f"正在刪除高缺失欄位...")
cols_to_drop_high_missing_exist = [col for col in cols_to_drop_high_missing if col in X_train.columns]
print(f"  將刪除: {cols_to_drop_high_missing_exist}")
X_train = X_train.drop(columns=cols_to_drop_high_missing_exist, errors='ignore')
X_val = X_val.drop(columns=cols_to_drop_high_missing_exist, errors='ignore')
if X_test is not None:
    X_test = X_test.drop(columns=cols_to_drop_high_missing_exist, errors='ignore')
gc.collect()

# B. 用 0 填補特定欄位
cols_to_fill_zero = [
    col for col in X_train.columns
    if ('買賣力' in col or
        '長期負債' in col or
        '長期資金' in col or
        '長期負債對淨值' in col or
        '短期借款' in col or
        '研發費用比率' in col or
        '利息費用對營業收入比率' in col)
]
ranked_broker_numeric_cols = [
    col for col in X_train.columns
    if ('買超第' in col or '賣超第' in col) and not col.endswith('券商代號') and not col.endswith('均價')
]
cols_to_fill_zero.extend(ranked_broker_numeric_cols)
cols_to_fill_zero.extend([
    '官股券商_張增減', '官股券商_金額增減(千)',
    '官股券商_買張', '官股券商_賣張', '官股券商_庫存',
    '官股券商_買金額(千)', '官股券商_賣金額(千)',
    '官股券商_買筆數', '官股券商_賣筆數'
])
cols_to_fill_zero = list(set([col for col in cols_to_fill_zero if col in X_train.columns]))

if cols_to_fill_zero:
    print(f"正在對 {len(cols_to_fill_zero)} 個欄位填補 0...")
    X_train[cols_to_fill_zero] = X_train[cols_to_fill_zero].fillna(0)
    X_val[cols_to_fill_zero] = X_val[cols_to_fill_zero].fillna(0)
    if X_test is not None:
        X_test[cols_to_fill_zero] = X_test[cols_to_fill_zero].fillna(0)

# C. 特殊填補利息保障倍數
cols_interest_coverage = ['季IFRS財報_利息保障倍數(倍)', '季IFRS財報_利息保障倍數累季(倍)']
large_value = 999999
for col in cols_interest_coverage:
    if col in X_train.columns:
        print(f"正在對欄位 '{col}' 的 NaN 填補極大值 {large_value}...")
        X_train[col] = X_train[col].fillna(large_value)
        X_val[col] = X_val[col].fillna(large_value)
        if X_test is not None and col in X_test.columns:
            X_test[col] = X_test[col].fillna(large_value)

# D. 用中位數填補剩餘的數值欄位 NaN
print("準備用中位數填補剩餘數值欄位...")
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
remaining_missing_numeric_cols = [col for col in numeric_cols if X_train[col].isnull().sum() > 0]
median_imputer = None
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

# E. 處理分類特徵的 NaN (在轉換 dtype 前)
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

# F. 最後檢查
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
# 2.3 **重要：** 轉換分類特徵 dtype
# ------------------------------
if categorical_features_existing:
    print(f"\n步驟 2.3: 找到 {len(categorical_features_existing)} 個分類特徵，正在轉換 dtype...")
    mem_before = X_train.memory_usage(deep=True).sum() / 1024**2
    train_categories = {}
    for col in categorical_features_existing:
        X_train[col] = X_train[col].astype('category')
        train_categories[col] = X_train[col].cat.categories
        if col in X_val.columns:
            try:
                X_val[col] = pd.Categorical(X_val[col], categories=train_categories[col], ordered=False)
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

# ------------------------------
# 4. 選擇重要特徵 (累積重要度 99%)
# ------------------------------
cumulative_importance_threshold = 0.99
feature_importance_df['cumulative_importance_ratio'] = (
    feature_importance_df['importance'].cumsum() / feature_importance_df['importance'].sum()
)
selected_features = feature_importance_df[
    feature_importance_df['cumulative_importance_ratio'] <= cumulative_importance_threshold
]['feature'].tolist()

MIN_FEATURES = 50
MAX_FEATURES = 2000
if len(selected_features) < MIN_FEATURES:
    selected_features = feature_importance_df['feature'].head(MIN_FEATURES).tolist()
elif len(selected_features) > MAX_FEATURES:
    selected_features = feature_importance_df['feature'].head(MAX_FEATURES).tolist()

print(f"\n步驟 4: 已選擇 Top {len(selected_features)} 個特徵 (99% cum importance).")

X_train_selected = X_train[selected_features].copy()
X_val_selected = X_val[selected_features].copy()
selected_categorical_features = [col for col in categorical_features_existing if col in selected_features]
if selected_categorical_features:
    print(f"選定的特徵中包含 {len(selected_categorical_features)} 個分類特徵。")

# 不需要的資料可以刪除節省記憶體
del X_train, X_val
gc.collect()

# ------------------------------
# 5. 超參數搜索 (Optuna) - 使用 50% train data + 固定 val
# ------------------------------
print("\n步驟 5: 使用 Optuna 進行超參數搜尋 (在 50% Training Data 上).")

# 先拆出 50% 來做 Optuna search，其餘將不參與搜索，但最後模型會用全部資料做再訓練
X_train_opt, _, y_train_opt, _ = train_test_split(
    X_train_selected,
    y_train,
    train_size=0.5,  # 使用 50% 做 search
    stratify=y_train,
    random_state=42
)

def objective(trial):
    # 可以調整搜尋空間
    param = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "device": "cuda",
        "enable_categorical": True,
        "early_stopping_rounds": 50,
        # 以下為要搜尋的超參數
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": 1000  # 先固定上限大，等 early_stopping
    }

    # 建立模型
    model = xgb.XGBClassifier(**param)

    # fit & 驗證
    model.fit(
        X_train_opt, y_train_opt,
        eval_set=[(X_val_selected, y_val)],
        verbose=False
    )
    # 用 validation 計算 AUC-PR 當作目標
    y_proba = model.predict_proba(X_val_selected)[:, 1]
    precision_pr, recall_pr, _ = precision_recall_curve(y_val, y_proba)
    val_aucpr = auc(recall_pr, precision_pr)

    return val_aucpr  # Optuna 預設 direction='minimize', 需要改成 maximize

# 建立 study
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)  # 可自行調整

best_trial = study.best_trial
print(f"Optuna 最佳分數(Val AUC-PR) = {best_trial.value:.6f}")
print("最佳參數:", best_trial.params)

# ------------------------------
# 6. 最終訓練：用最佳參數 + 選定特徵，在完整訓練集上做訓練
# ------------------------------
print("\n步驟 6: 使用最佳參數，在完整訓練集(選定特徵)上訓練最終模型...")
best_params = best_trial.params
# 固定一些非搜尋範圍的參數
final_params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "device": "cuda",
    "enable_categorical": True,
    "early_stopping_rounds": 50,
    "n_estimators": 1000,  # 留相同
}
# 將最佳搜尋到的參數合併
final_params.update(best_params)

final_xgb = xgb.XGBClassifier(**final_params)

print("開始訓練最終模型...")
start_time_train = time.time()
eval_set = [(X_val_selected, y_val)]
final_xgb.fit(
    X_train_selected,
    y_train,
    eval_set=eval_set,
    verbose=50
)
train_cost = time.time() - start_time_train
print(f"最終模型訓練完成，耗時: {train_cost:.2f} 秒")

# ------------------------------
# 7. 在驗證集上評估(使用 0.1 閾值)
# ------------------------------
print("\n步驟 7: 在獨立驗證集上評估最終模型 (閾值=0.1)...")
y_pred_val_proba = final_xgb.predict_proba(X_val_selected)[:, 1]
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

print("\n最終模型 - 驗證集評估 (閾值=0.1):")
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
# 8. 儲存最終模型和結果
# ------------------------------
model_path = os.path.join(output_dir, f"xgb_model_v{version}.json")
final_xgb.save_model(model_path)
print(f"\n模型已儲存至: {model_path}")

results_summary = {
    "model_type": "XGBoost",
    "device": "cuda",
    "feature_selection": f"Top {len(selected_features)} features (99% cum importance)",
    "imputation_strategy": "Drop high missing, Zero fill, Median fill etc.",
    "hyperparameters": final_xgb.get_params(),
    "best_optuna_val_aucpr": best_trial.value,
    "best_iteration": final_xgb.best_iteration if hasattr(final_xgb, 'best_iteration') else 'N/A',
    "validation_f1(0.1_thr)": f1_val,
    "validation_precision(0.1_thr)": precision_val,
    "validation_recall(0.1_thr)": recall_val,
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
# 9. (可選) 測試集預測 (使用同樣的特徵子集、同樣補值策略)
# ------------------------------
if test_exists:
    print("\n正在處理測試集...")
    test_df = pd.read_parquet(test_file)
    test_id = test_df['ID'] if 'ID' in test_df.columns else None
    X_test = test_df.drop(columns=cols_to_drop_manual, errors='ignore')

    # **重要**: 應用相同補值
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

    # 篩選特徵
    missing_cols_test = [col for col in selected_features if col not in X_test.columns]
    if missing_cols_test:
        raise ValueError(f"測試集缺少以下必要的特徵: {missing_cols_test}")
    X_test_selected = X_test[selected_features].copy()

    # 對齊分類特徵類別
    if selected_categorical_features:
        print(f"轉換測試集的分類特徵 dtype 並對齊 categories...")
        for col in selected_categorical_features:
            if col in X_test_selected.columns:
                try:
                    X_test_selected[col] = pd.Categorical(
                        X_test_selected[col],
                        categories=train_categories[col],
                        ordered=False
                    )
                    if X_test_selected[col].isnull().any():
                        print(f"警告: 欄位 '{col}' 在測試集中包含訓練集未出現的類別，已轉為 NaN。")
                except Exception as e:
                    print(f"處理測試集欄位 '{col}' 時發生錯誤: {e}.")
            else:
                print(f"警告：分類特徵 '{col}' 不在測試集中。")

    # 預測
    print("正在進行測試集預測 (閾值=0.1)...")
    y_test_proba = final_xgb.predict_proba(X_test_selected)[:, 1]
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
