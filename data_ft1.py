import pandas as pd
import os
import glob
import json
import chardet
from tqdm import tqdm
import re

# 请确保设置你的PROMPT
PROMPT =  "你是一个医学专家，你需要根据用户的问题，提炼出核心问题，指定科室, 并给出诊断结果和治疗意见。"


def is_row_valid(data, required_columns):
    """
    检查行数据是否有效
    无效数据过滤
    """
    for col in required_columns:
        if pd.isna(data[col]) or data[col] is None:
            return False, col
        # 确保数据是字符串类型再调用strip()
        if isinstance(data[col], str):
            if data[col].strip() == "" or data[col].strip() == "无" or len(data[col].strip()) < 4:
                return False, col
        else:
            # 如果不是字符串类型，认为无效
            return False, col
    return True, None


def remove_after_qingwen_regex(text):
    """
    使用正则表达式去除"请问："及后面的内容
    """
    if pd.isna(text) or text is None:
        return text

    text = str(text)

    # 匹配"请问："及后面的所有内容
    pattern = r'请问：.*$'
    result = re.sub(pattern, '', text)

    return result.strip()

def convert_csv_encoding(directory_path, source_encoding = 'utf-8', target_encoding='utf-8', overwrite=False):
    """
    将目录下所有CSV文件转换为目标编码（默认UTF-8）

    参数:
    - directory_path: 包含CSV文件的目录路径
    - target_encoding: 目标编码 (默认: utf-8)
    - overwrite: 是否覆盖原文件 (默认 False，新文件加后缀 "_utf8.csv")
    """
    if not os.path.exists(directory_path):
        print(f"错误: 目录 '{directory_path}' 不存在")
        return {"success": [], "failed": []}

    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    if not csv_files:
        print(f"在目录 '{directory_path}' 中没有找到CSV文件")
        return {"success": [], "failed": []}

    print(f"找到 {len(csv_files)} 个CSV文件，开始编码转换...")

    results = {"success": [], "failed": []}

    for source_file in tqdm(csv_files, desc="转换编码"):
        try:

            # 如果已经是目标编码，跳过
            if source_encoding.lower() == target_encoding.lower():
                results["success"].append(source_file)
                continue

            # 输出路径
            if overwrite:
                target_file = source_file
            else:
                target_file = source_file.replace(".csv", "_utf8.csv")

            with open(source_file, 'r', encoding=source_encoding) as f:
                content = f.read()

            # 写入UTF-8编码
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ 成功转换: {source_file} -> {target_file} (UTF-8)")
            results["success"].append(target_file)
        except Exception as e:
            print(f"✗ 转换 {os.path.basename(source_file)} 失败: {e}")
            results["failed"].append(source_file)

    return results


def process_and_merge_csv_files(directory_path, output_jsonl_path, output_format_jsonl_path, chunk_size=1000):
    """
    处理目录下所有CSV文件并合并为JSONL格式

    参数:
    directory_path: 包含CSV文件的目录路径
    output_jsonl_path: 输出的JSONL文件路径
    chunk_size: 分块读取的大小
    """
    # 首先转换编码
    csv_files = convert_csv_encoding(directory_path)

    if not csv_files["success"]:
        print("没有找到可处理的CSV文件")
        return

    print(f"\n开始处理 {len(csv_files['success'])} 个CSV文件...")

    all_messages = []
    all_format_messages = []
    total_rows_processed = 0
    total_valid_rows = 0

    for file_path in tqdm(csv_files['success'], desc="处理文件"):
        try:
            file_total_rows = 0

            for chunk in pd.read_csv(
                    file_path,
                    encoding='utf-8',
                    usecols=['department', 'title', 'ask', 'answer'],
                    na_values=['Unknown', 'N/A', ''],
                    chunksize=chunk_size
            ):
                required_columns = ["ask", "title", "department", "answer"]

                for index, data in chunk.iterrows():
                    # 无效数据过滤
                    is_valid, empty_col = is_row_valid(data, required_columns)
                    if not is_valid:
                        continue

                    # 解析每一行的数据
                    input_text = remove_after_qingwen_regex(data["ask"]) # 将用户问题和提炼问题分割开
                    core_question = data["title"]
                    dept = data["department"]
                    answer = data["answer"]

                    message = {
                        "input_text": input_text,
                        "core_question": core_question,
                        "dept": dept,
                        "answer": answer
                    }

                    output = f"<department>{dept}</department> \n <core_question>{core_question}</core_question> \n {answer}"
                    message_format = {
                        "instruction": PROMPT,
                        "input": f"{input_text}",
                        "output": output,
                    }
                    all_messages.append(message)
                    all_format_messages.append(message_format)
                    file_total_rows += 1
                    total_rows_processed += 1
        except Exception as e:
            print(f"处理文件时出错: {e}")

    # 保存结果
    print(f"保存结果到 {output_jsonl_path}...")
    with open(output_jsonl_path, "w", encoding="utf-8") as file:
        for message in tqdm(all_messages, desc="写入JSONL"):
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    print(f"保存结果到 {output_jsonl_path}...")
    with open(output_format_jsonl_path, "w", encoding="utf-8") as file:
        for message in tqdm(all_format_messages, desc="写入JSONL"):
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    print(f"\n处理完成!")
    print(f"总处理行数: {total_rows_processed}")
    print(f"有效行数: {total_valid_rows}")
    print(f"过滤率: {(total_rows_processed - total_valid_rows) / total_rows_processed * 100:.2f}%")
    print(f"输出文件: {output_jsonl_path}")


def process_single_large_csv(csv_file_path, output_jsonl_path, output_format_jsonl_path, chunk_size=500):
    """
    处理单个大CSV文件（如果文件太大无法一次性读取）

    参数:
    csv_file_path: CSV文件路径
    output_jsonl_path: 输出的JSONL文件路径
    chunk_size: 分块读取的大小
    """
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 '{csv_file_path}' 不存在")
        return

    print(f"开始处理大文件: {os.path.basename(csv_file_path)}")

    all_messages = []
    all_format_messages = []
    total_rows_processed = 0
    total_valid_rows = 0

    try:
        for chunk in tqdm(pd.read_csv(
                csv_file_path,
                encoding='utf-8',
                usecols=['department', 'title', 'ask', 'answer'],
                na_values=['Unknown', 'N/A', ''],
                chunksize=chunk_size
        ), desc="处理数据块"):

            required_columns = ["ask", "title", "department", "answer"]

            for index, data in chunk.iterrows():
                total_rows_processed += 1

                # 无效数据过滤
                is_valid, empty_col = is_row_valid(data, required_columns)
                if not is_valid:
                    continue

                # 解析每一行的数据
                input_text = data["ask"]
                core_question = data["title"]
                dept = data["department"]
                answer = data["answer"]

                message = {
                    "input_text": input_text,
                    "core_question": core_question,
                    "dept": dept,
                    "answer": answer
                }

                output = f"<department>{dept}</department> \n <core_question>{core_question}</core_question> \n {answer}"
                message_format = {
                    "instruction": PROMPT,
                    "input": f"{input_text}",
                    "output": output,
                }
                all_messages.append(message)
                all_format_messages.append(message_format )
                total_valid_rows += 1

        # 保存结果
        print(f"保存结果到 {output_jsonl_path}...")
        with open(output_jsonl_path, "w", encoding="utf-8") as file:
            for message in tqdm(all_messages, desc="写入JSONL"):
                file.write(json.dumps(message, ensure_ascii=False) + "\n")
        print(f"保存结果到 {output_jsonl_path}...")
        with open(output_format_jsonl_path, "w", encoding="utf-8") as file:
            for message in tqdm(all_format_messages, desc="写入JSONL"):
                file.write(json.dumps(message, ensure_ascii=False) + "\n")

        print(f"\n处理完成!")
        print(f"总处理行数: {total_rows_processed}")
        print(f"有效行数: {total_valid_rows}")
        print(f"过滤率: {(total_rows_processed - total_valid_rows) / total_rows_processed * 100:.2f}%")

    except Exception as e:
        print(f"处理文件时出错: {e}")


# 使用示例
if __name__ == "__main__":
    # 安装所需库
    # pip install pandas chardet tqdm

    # 方法1: 处理目录下所有CSV文件
    script_path = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_path, "data/rawdata")  # 修改为你的目录路径
    output_file = os.path.join(script_path, "data/full_data_ft1.jsonl")
    output_format_file = os.path.join(script_path, "data/full_data_format_ft1.jsonl")

    process_and_merge_csv_files(directory, output_file, output_format_file, chunk_size=1000)