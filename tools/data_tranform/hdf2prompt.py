import pandas as pd
import json
from tqdm import tqdm
# import h5py # 不再需要 h5py

def row_to_prompt(row):
    """
    將單一資料列轉換成 prompt 格式：
    - instruction：請根據下列股票基本面特徵判斷該股票是否具有顯著上漲潛力，請回答「是」或「否」：
    - input：將所有特徵（排除 '飆股' 標籤，且值不為空）以 key: value 形式串接
    - output：若 '飆股' 為 1 則輸出「是」，若為 0 則輸出「否」
    """
    # --- 修改：增加 pd.notna() 檢查，只包含非空值的特徵 ---
    features = [f"{col}: {row[col]}" for col in row.index if col != "飆股" and pd.notna(row[col])]
    # --- 修改結束 ---
    input_str = ", ".join(features)
    output_str = "是" if row["飆股"] == 1 else "否"
    prompt = {
        "instruction": "請根據下列股票的基本面特徵，綜合評估該股票是否具有顯著上漲潛力，請回答「是」或「否」：",
        "input": input_str,
        "output": output_str
    }
    return prompt

# process_chunk 函數保持不變
def process_chunk(chunk, f, total_converted):
    """處理單一區塊中的所有資料並寫入 JSONL"""
    for index, row in chunk.iterrows():
        try:
            prompt = row_to_prompt(row)
            json.dump(prompt, f, ensure_ascii=False)
            f.write("\n")
            total_converted += 1
        except Exception as e:
            print(f"\n處理第 {index} 行時發生錯誤: {e}")
            print("錯誤資料行：", row)
    return total_converted

# --- 修改：將函數名稱和邏輯改為處理 CSV ---
def convert_csv_to_jsonl(csv_file, jsonl_file, chunksize=10000):
    """
    讀取 CSV 檔案，分區處理、轉換成 prompt 格式（跳過空值），
    並將所有資料存成 JSONL 檔案。

    參數：
      csv_file：輸入 CSV 檔案路徑
      jsonl_file：輸出 JSONL 檔案名稱
      chunksize：每次讀取的筆數（根據記憶體大小調整）
    """
    try:
        total_converted = 0
        # 使用 pd.read_csv 並指定 chunksize 來分塊讀取
        print(f"開始讀取 CSV 檔案：{csv_file}")
        with open(jsonl_file, "w", encoding="utf-8") as f:
            # 使用 tqdm 包裹 read_csv 的迭代器以顯示進度
            for chunk in tqdm(pd.read_csv(csv_file, chunksize=chunksize), desc="處理 CSV 分區"):
                total_converted = process_chunk(chunk, f, total_converted)

        print(f"轉換完成，共處理 {total_converted} 筆資料，檔案已存成：{jsonl_file}")

    except FileNotFoundError:
        print(f"錯誤：找不到 CSV 檔案 '{csv_file}'")
    except Exception as e:
        print(f"處理 CSV 檔案時發生錯誤: {e}")
# --- 修改結束 ---

if __name__ == "__main__":
    # --- 修改：指定 CSV 檔案路徑 ---
    # 假設你的原始 CSV 檔案叫做 train.csv 且在 datasets/origin/ 目錄下
    # 請根據你的實際情況修改路徑
    csv_file = "datasets/origin/train.csv"  # 輸入 CSV 檔案路徑
    jsonl_file = "finetune_dataset.jsonl"   # 輸出 JSONL 檔案名稱
    # --- 修改結束 ---

    # 呼叫修改後的函數
    convert_csv_to_jsonl(csv_file, jsonl_file, chunksize=10000)