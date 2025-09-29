import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
import gc
from config import global_config

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["SWANLAB_PROJECT"]="qwen3-sft-medical"
PROMPT = "你是一个医学专家，你需要根据用户的问题，提炼出核心问题，指定科室, 并给出诊断结果和治疗意见。"
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 512
swanlab.config.update({
    "model": f"{global_config.TRAIN_MODEL}-ft1",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
    })



def process_func(example):
    """
    将数据集进行预处理
    """ 
    # 构造prompt和回答
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    # 不手动添加pad，由collator负责padding
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]

    # labels 仅对assistant部分计算损失，prompt部分置为 -100
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    # 将所有标签值加1，同时保持-100不变（所有的标签都加上1，标签里是不能有数字0的，让模型矛盾，导致程序崩溃）
    # labels = [x + 1 if x != -100 else -100 for x in labels]

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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

def check_model_health(model):
    """检查模型权重是否健康"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"警告: 参数 {name} 包含 NaN 或 Inf 值")
            return False
    return True

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


# 获取脚本所在目录，并创建模型缓存路径
script_path = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(script_path, "models")

# Transformers加载模型权重（本地）
device, load_dtype = select_device_and_dtype()

# 自动查找之前最新的 checkpoint
# 构造相对于脚本所在目录的路径，使其不受运行位置的影响
script_dir = os.path.dirname(os.path.abspath(__file__))
pre_output_dir = os.path.join(script_dir, f"{global_config.OUTPUT_DIR}")
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
model = AutoModelForCausalLM.from_pretrained(latest_checkpoint, dtype=load_dtype)
tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint, use_fast=False, trust_remote_code=True)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
model.to(device)


# 确保存在pad token，便于padding
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载、处理数据集和测试集
jsonl_path = os.path.join(script_path, "data/full_data_format_ft1.jsonl")
if not os.path.exists(jsonl_path):
    raise ValueError(f"数据集地址：{jsonl_path} 不存在")

full_df = pd.read_json(jsonl_path, lines=True)

# 整体数据量较大，我们只取一部分用作训练
sampled_df = full_df.sample(frac=global_config.DATA_USE_FRAC, random_state=42)

# 将DataFrame转换为Dataset
dataset = Dataset.from_pandas(sampled_df)

# 分割数据集 - 这里返回的是DatasetDict对象
dataset_split = dataset.train_test_split(
    test_size=0.2,
    seed=42  # 设置随机种子保证可重复性
)

# 得到训练集和验证集
train_df = dataset_split['train']
eval_df = dataset_split['test']

print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(eval_df)}")

# 应用处理函数
train_dataset = train_df.map(
    process_func,
    remove_columns=train_df.column_names
)

eval_dataset = eval_df.map(
    process_func,
    remove_columns=eval_df.column_names
)

# 使用能够为labels补 -100 的 collator
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

args = TrainingArguments(
    output_dir=os.path.join(script_path, f"{global_config.OUTPUT_DIR}-ft1"),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=400,
    # learning_rate=5e-5,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=False, # 开启以节省显存，关闭追求最大训练速度
    report_to="swanlab",
    run_name=f"{global_config.MODEL_VERSION}-ft1",
    # 添加以下优化参数
    warmup_steps=100,  # 添加热身
    weight_decay=0.01,  # 添加权重衰减
    max_grad_norm=1.0,  # 梯度裁剪
    lr_scheduler_type="cosine",  # 使用cosine学习率调度
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
)

trainer.train()

# 用验证集的前3条，主观看模型
test_samples = eval_df[:3]

test_text_list = []

for index in range(len(test_samples) - 1):
    instruction = test_samples['instruction'][index]
    input_value = test_samples['input'][index]

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer, device)

    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
    
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})

swanlab.finish()