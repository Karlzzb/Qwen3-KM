import pandas as pd
import os
import json

# 读取医学数据集
def read_simple(file_path, encoding):
    medical_data = pd.read_csv(
        file_path,
        encoding=encoding,
        usecols=['department', 'title', 'ask', 'answer'],
        # parse_dates=['visit_date'],  # 自动解析日期
        na_values=['Unknown', 'N/A', '']
    )

    print(f"数据集形状: {medical_data.shape}")
    print(f"列名: {medical_data.columns.tolist()}")
    print("\n前5行数据:")
    print(medical_data.head())


def convert_to_utf8(source_file, target_file=None, source_encoding=None):
    """
    将文件转换为UTF-8编码

    参数:
        source_file: 源文件路径
        target_file: 目标文件路径（默认覆盖原文件）
        source_encoding: 源文件编码（自动检测）
    """
    import chardet

    if target_file is None:
        target_file = source_file

    # 检测源文件编码
    if source_encoding is None:
        with open(source_file, 'rb') as f:
            raw_data = f.read()
            encoding_info = chardet.detect(raw_data)
            source_encoding = encoding_info['encoding']
            print(f"检测到编码: {source_encoding} (置信度: {encoding_info['confidence']:.2f})")

    # 读取并转换
    try:
        with open(source_file, 'r', encoding=source_encoding) as f:
            content = f.read()

        # 写入UTF-8编码
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ 成功转换: {source_file} -> {target_file} (UTF-8)")
        return True

    except UnicodeDecodeError:
        print(f"❌ 解码失败，尝试其他编码...")
        return False


# 使用示例
file_path = 'data/train_f1.csv'
source_encoding = 'gb18030'
# new_file = 'data/train_f1.csv'
# read_simple(file_path, source_encoding)
# convert_to_utf8(file_path, 'data/train_f1.csv', source_encoding)
# read_simple(new_file, 'utf-8')
PROMPT = "你是一个医学专家，你需要根据用户的问题，提炼出核心问题，并指定科室"

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    chunk_size = 1000
    for chunk in pd.read_csv(
        origin_path,
        encoding='utf-8',
        usecols=['department', 'title', 'ask', 'answer'],
        na_values=['Unknown', 'N/A', ''],
        chunksize=chunk_size
    ):

        for index, data in chunk.iterrows():
            # 解析每一行的数据
            input = data["ask"]
            core_question = data["title"]
            dept = data["department"]
            answer = data["answer"]
            output = f"<department>{dept}</department> \n <core_question>{core_question}</core_question> \n {answer}"
            message = {
                "instruction": PROMPT,
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")



script_path = os.path.dirname(os.path.abspath(__file__))
train_dataset_path = os.path.join(script_path, "data/train_f1.csv")
train_jsonl_new_path = os.path.join(script_path, "data/train_f1_format.jsonl")

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)