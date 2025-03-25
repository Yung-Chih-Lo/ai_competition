## 更新
- 2025/03/26：更新 v2資料集連結 
🔗 [Google Drive 資料下載連結](https://drive.google.com/file/d/1lOwGViIj5XKenjSgXEhBn3YjaKzlrYor/view?usp=sharing)
- 

## 📦 環境安裝
請先安裝必要套件：
```bash
pip install -r requirements.txt
```

---

## 📁 專案結構說明
```
root
├─ 38_Submmision_Template/
│  └─ submission_template_public.csv                → 提交格式範例
├─ datasets/                                        → 資料集
│  ├─ mean_pca_200/                                 → PCA 處理後的資料
│  │  ├─ pca_model.pkl                              → PCA 模型
│  │  ├─ test_pca_200.h5
│  │  ├─ train_pca_200.h5
│  │  └─ val_pca_200.h5
│  └─ origin/                                       → 原始資料
│     ├─ origin.zip
│     ├─ test.csv
│     ├─ test.h5
│     ├─ train.csv
│     ├─ train.h5
│     ├─ training.csv
│     └─ val.h5
├─ df2hdf.py                                        → 將 CSV 轉換為 HDF5 檔案（未處理）
├─ important_features/                              → 特徵重要性分析相關
│  ├─ cols.txt
│  ├─ feature_importance.py
│  ├─ xgb_aucpr.csv
│  ├─ xgb_aucpr_curve.png
│  ├─ xgb_feature_importances.csv
│  ├─ xgb_feature_importances_top20.png
│  └─ xgb_model.json
├─ pca.py                                           → 進行 PCA 特徵壓縮（壓到 200 維）
├─ predict.py                                       → 使用訓練好的 XGBoost 模型預測
├─ Readme.md                                        → 專案說明文件
├─ requirements.txt                                 → Python 套件清單
├─ results/                                         → 各版本模型預測結果與訓練紀錄
│  ├─ v1/
│  │  ├─ test_predictions_v1.csv
│  │  ├─ xgb_logloss.csv
│  │  ├─ xgb_logloss_curve.png
│  │  └─ xgb_model.json
│  ├─ v2/
│  │  ├─ test_predictions_v2.csv
│  │  ├─ xgb_logloss.csv
│  │  ├─ xgb_logloss_curve.png
│  │  └─ xgb_model.json
│  └─ v3/
│     ├─ test_predictions_v3.csv
│     ├─ xgb_aucpr.csv
│     ├─ xgb_aucpr_curve.png
│     └─ xgb_model.json
├─ test_train_val_distribution.py                   → 查看 Train/Val/Test 飆股分布
├─ train_XGBoost_v1.py                              → XGBoost 模型訓練程式 - V1
├─ train_XGBoost_v2.py                              → XGBoost 模型訓練程式 - V2
└─ train_XGBoost_v3.py                              → XGBoost 模型訓練程式 - V3
```

---

## 🔍 模組說明

| 檔案/資料夾 | 功能說明 |
|-------------|---------|
| `df2hdf.py` | 將原始 CSV 檔轉換為 HDF5 格式，**無任何預處理**。 |
| `pca.py` | 對原始資料進行 PCA 壓縮至 200 維並儲存為新的 HDF5 檔。 |
| `downsample.py` | 對訓練資料進行下採樣。 |
| `predict.py` | 使用訓練好的 XGBoost 模型進行預測。 |
| `test_train_val_distribution.py` | 分析 Train/Val/Test 資料集中飆股的分布情形。 |
| `important_features/` | 儲存特徵重要性分析結果與圖表。 |
| `results/` | 儲存各版本訓練模型與預測結果。 |
| `train_XGBoost_v*.py` | 各版本的 XGBoost 模型訓練腳本。 |


---

## 📂 資料下載與使用
大型檔案已上傳至 Google Drive，請從以下連結下載並解壓縮到專案目錄中對應位置：
!!已更新 v2 資料，請看前面更新

🔗 [Google Drive 資料下載連結](https://drive.google.com/drive/folders/1O41PjWAtuVImqqaxg7X8SYeK7lOgTiGC?usp=sharing)

---

## 📄 更多訓練細節說明
請參考以下 Notion 筆記頁面，裡面有各版本模型訓練的詳細資訊與參數設定：

🔗 [Notion 說明文件](https://elfin-poinsettia-e39.notion.site/stock-train-1bede8154c9c807ebc99c0637c365d60?pvs=4)

---

## ⚠ 注意事項
- **請確認資料路徑是否正確**，尤其是 `datasets/` 內部結構是否符合預期。
- 若資料夾結構不同，請於程式碼中手動修改對應路徑。

---