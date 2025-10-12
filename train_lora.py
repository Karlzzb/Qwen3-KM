import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import os
import swanlab
import gc
from sklearn.model_selection import train_test_split

os.environ["SWANLAB_PROJECT"]="qwen3-sft-medical"
PROMPT = "你是一个医学专家，你需要根据用户的问题，提炼出核心问题，指定科室, 并给出诊断结果和治疗意见。"
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 512
swanlab.config.update({
    "model": "Qwen/Qwen3-0.6B-Lora",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
    })

def is_row_valid(data, required_columns):
    """
    检查行数据是否有效
    无效数据过滤
    """
    for col in required_columns:
        if pd.isna(data[col]) or data[col] is None:
            return False, col
        if data[col].strip() == "" or data[col].strip() == "无" or len(data[col].strip()) < 4:
            return False, col
    return True, None

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
        required_columns = ["ask", "title", "department", "answer"]
        for index, data in chunk.iterrows():
            # 无效数据过滤
            is_valid, empty_col = is_row_valid(data, required_columns)
            if not is_valid:
                # print(f"跳过第 {index} 行，列 '{empty_col}' 为空")
                continue

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


def process_func(example):
    """
    将数据集进行预处理
    """
    # 使用聊天模板构建输入
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": example['input']},
        {"role": "assistant", "content": example['output']}
    ]

    # 使用tokenizer的apply_chat_template方法，更可靠
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # 分词
    tokenized = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding=False)

    # 标签就是input_ids的副本，但需要mask掉用户和系统部分
    labels = tokenized["input_ids"].copy()

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

def check_model_health(model):
    """检查模型权重是否健康"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"警告: 参数 {name} 包含 NaN 或 Inf 值")
            return False
    return True

def select_device_and_dtype():
    # 优先尝试 CUDA，但对不被当前PyTorch支持的架构（如 sm_120）回退CPU
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major >= 12:
                # 当前PyTorch不支持sm_120，回退CPU
                raise RuntimeError("Unsupported CUDA capability for current PyTorch")
            _ = torch.zeros(1, device="cuda")  # 实测一次分配
            print("INFO: Using CUDA (GPU) for training.")
            return "cuda", torch.float32
        except Exception as e:
            print(f"WARN: CUDA available but failed to use, falling back to CPU. Reason: {e}")
            pass
    print("INFO: No compatible CUDA device found, using CPU for training.")
    return "cpu", torch.float32

def safe_generate(model, input_ids, attention_mask, tokenizer, device,
                  temperature=0.5, top_p=0.85, repetition_penalty=1.2):
    """安全的生成函数，包含多重保护"""

    # 检查模型健康状态
    if not check_model_health(model):
        print("模型权重异常，尝试修复...")
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()

    # 设置生成参数
    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": max(0.01, temperature),  # 避免温度过低
        "top_p": max(0.01, min(0.99, top_p)),  # 限制top_p范围
        "repetition_penalty": max(1.0, min(2.0, repetition_penalty)),
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "output_scores": False,
        "return_dict_in_generate": False,
        "use_cache": True,
    }

    # 确保输入在正确设备上
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    try:
        with torch.no_grad():
            # 使用torch.cuda.amp.autocast进行混合精度推理
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_config
                    )
            else:
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )

        return generated

    except RuntimeError as e:
        if "device-side assert" in str(e):
            print(f"CUDA设备端断言失败: {e}")
            print("尝试降低生成参数...")

            # 降低参数重试
            generation_config["temperature"] = 0.7
            generation_config["top_p"] = 0.9
            generation_config["max_new_tokens"] = min(512, MAX_NEW_TOKENS)

            try:
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_config
                    )
                return generated
            except:
                print("降低参数后仍然失败，切换到CPU...")
                return None
        else:
            raise e

def predict(messages, model, tokenizer, device, attempt=0,
            temperature=0.5, top_p=0.85, repetition_penalty=1.2, RETRY_ATTEMPTS=3):
    """预测函数，包含重试机制"""
    if attempt >= RETRY_ATTEMPTS:
        print(f"已达到最大重试次数({RETRY_ATTEMPTS})，无法生成有效回答")
        return None

    try:
        # 应用聊天模板
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask if hasattr(inputs, "attention_mask") else None

        # 安全生成
        generated = safe_generate(model, input_ids, attention_mask, tokenizer, device,
                                  temperature, top_p, repetition_penalty)

        if generated is None:
            print("生成失败，尝试使用CPU...")
            # 切换到CPU
            model_cpu = model.cpu()
            input_ids_cpu = input_ids.cpu()
            attention_mask_cpu = attention_mask.cpu() if attention_mask is not None else None

            generated = safe_generate(model_cpu, input_ids_cpu, attention_mask_cpu, tokenizer, "cpu",
                                      temperature, top_p, repetition_penalty)

            if generated is None:
                return predict(messages, model, tokenizer, device, attempt + 1,
                               temperature * 1.1, top_p * 0.95, repetition_penalty * 0.95)

        # 解码输出
        new_tokens = generated[:, input_ids.shape[1]:]
        response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        # 清理可能的重复内容
        if response.count('!') > len(response) * 0.5:
            print("检测到异常输出，尝试重新生成...")
            return predict(messages, model, tokenizer, device, attempt + 1,
                           temperature * 1.1, top_p * 0.95, repetition_penalty * 0.95)

        return response

    except Exception as e:
        print(f"生成过程中发生错误: {e}")
        if attempt < RETRY_ATTEMPTS - 1:
            print(f"重试 {attempt + 1}/{RETRY_ATTEMPTS}...")
            return predict(messages, model, tokenizer, device, attempt + 1,
                           temperature * 1.1, top_p * 0.95, repetition_penalty * 0.95)
        else:
            return None

# Transformers加载模型权重（本地）
device, load_dtype = select_device_and_dtype()

# 自动查找之前最新的 checkpoint
# 构造相对于脚本所在目录的路径，使其不受运行位置的影响
script_path = os.path.dirname(os.path.abspath(__file__))
pre_output_dir = os.path.join(script_path, "output/Qwen3-0.6B-ft1")
latest_checkpoint = "checkpoint-1000"
if os.path.isdir(pre_output_dir):
    checkpoints = [
        d for d in os.listdir(pre_output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(pre_output_dir, d))
    ]
    if checkpoints:
        # 通过 checkpoint 编号排序找到最新的
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        latest_checkpoint = os.path.join(pre_output_dir, checkpoints[-1])
        print(f"INFO: 自动找到最新的 checkpoint: {latest_checkpoint}")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(latest_checkpoint, device_map="auto" if device == "cuda" else None, dtype=torch.float32)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 配置lora
"""
第一优先级：调整 target_modules（如果效果不佳）
    如果显存紧张或想更快训练：可以尝试只适配注意力层，即 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]。这是最核心的模块。    
    如果任务非常复杂，需要模型深度适配：保持当前配置是最好的。
第二优先级：调整 r 和 lora_alpha（用于平衡效果和效率）
    如果模型能力似乎不足（欠拟合）：尝试增大 r（例如从8到16），并保持 lora_alpha 为 r 的2-4倍（例如 r=16, alpha=32 或 64）。   
    如果训练不稳定或过拟合：尝试减小 r（例如从8到4），同时等比例减小 alpha（例如 r=4, alpha=16）。
第三优先级：调整 lora_dropout（用于控制过拟合）
    如果训练损失和验证损失差距很大：提高 lora_dropout（例如到0.2）。
    如果模型学习很慢或效果一直不好：可以尝试降低 lora_dropout（例如到0.05或0）。
"""
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
)

model = get_peft_model(model, config)

# 加载、处理数据集和测试集
jsonl_path = os.path.join(script_path, "data/full_data_format_ft1.jsonl")
if not os.path.exists(jsonl_path):
    raise ValueError(f"数据集地址：{jsonl_path} 不存在")

full_df = pd.read_json(jsonl_path, lines=True)
sampled_df = full_df.sample(frac=0.04, random_state=42) #只取20%数据做一个预研
train_df, eval_df = train_test_split(
    sampled_df,
    test_size=0.2,
    random_state=42  # 设置随机种子保证可重复性
)

# 得到训练集
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 得到验证集
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

args = TrainingArguments(
    output_dir="./output/Qwen3-0.6B-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    ddp_find_unused_parameters=False,
    remove_unused_columns=True,     # 节省内存
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=400,
    learning_rate=5e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="Qwen3-0.6B-ft1-lora",
    # 添加混合精度训练
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 用测试集的前3条，主观看模型
test_df = eval_df[:3]

test_text_list = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer,device)

    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
    
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})

swanlab.finish()