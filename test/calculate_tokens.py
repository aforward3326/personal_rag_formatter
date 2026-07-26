import json
import argparse
import statistics
from pathlib import Path

def calculate_tokens(input_path):
    path = Path(input_path)
    
    if path.is_file():
        files_to_process = [path]
    elif path.is_dir():
        files_to_process = list(path.glob('**/*.jsonl'))
        if not files_to_process:
            print(f"錯誤：在目錄 {input_path} 中找不到任何 .jsonl 檔案")
            return
    else:
        print(f"錯誤：找不到路徑 {input_path}")
        return

    inputs = []
    outputs = []
    total_tokens = 0

    for file_path in files_to_process:
        with file_path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    # 安全地取得 usageMetadata 節點
                    usage = data.get("response", {}).get("usageMetadata", {})

                    inp = usage.get("promptTokenCount", 0)
                    out = usage.get("candidatesTokenCount", 0)
                    tot = usage.get("totalTokenCount", 0)

                    # 記錄每一筆的值與對應的檔案名稱
                    inputs.append((inp, file_path.name))
                    outputs.append((out, file_path.name))
                    total_tokens += tot

                except json.JSONDecodeError:
                    print(f"警告：檔案 {file_path.name} 第 {line_num} 行不是有效的 JSON 格式")
                except Exception as e:
                    print(f"警告：處理檔案 {file_path.name} 第 {line_num} 行時發生錯誤: {e}")

    total_records = len(inputs)
    if total_records == 0:
        print("沒有找到任何有效的資料筆數。")
        return

    input_values = [x[0] for x in inputs]
    output_values = [x[0] for x in outputs]

    print(f"=== Token 統計分析 ===")
    print(f"處理檔案總數: {len(files_to_process)}")
    print(f"總資料筆數:   {total_records:,}")
    print(f"總 Token 數:  {total_tokens:,}\n")

    max_in = max(inputs, key=lambda x: x[0])
    min_in = min(inputs, key=lambda x: x[0])
    print(f"--- Input Token (Prompt) ---")
    print(f"平均:   {statistics.mean(input_values):.2f}")
    print(f"中位數: {statistics.median(input_values)}")
    print(f"最大值: {max_in[0]:,} (來自檔案: {max_in[1]})")
    print(f"最小值: {min_in[0]:,} (來自檔案: {min_in[1]})\n")

    max_out = max(outputs, key=lambda x: x[0])
    min_out = min(outputs, key=lambda x: x[0])
    print(f"--- Output Token (Candidates) ---")
    print(f"平均:   {statistics.mean(output_values):.2f}")
    print(f"中位數: {statistics.median(output_values)}")
    print(f"最大值: {max_out[0]:,} (來自檔案: {max_out[1]})")
    print(f"最小值: {min_out[0]:,} (來自檔案: {min_out[1]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="計算 JSONL 檔案中的 Token 統計數據")
    parser.add_argument("input_path", help=".jsonl 檔案的路徑或目錄")
    args = parser.parse_args()

    calculate_tokens(args.input_path)