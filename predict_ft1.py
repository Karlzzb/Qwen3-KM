import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import warnings
import gc

# 设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # 启用设备端断言

warnings.filterwarnings("ignore", message="The pynvml package is deprecated.")

PROMPT = "你是一个医学专家，你需要根据用户的问题，提炼出核心问题，并指定科室"
MAX_NEW_TOKENS = 2048
RETRY_ATTEMPTS = 3


def select_device_and_dtype(force_cpu=False):
    """选择设备和数据类型"""
    if not force_cpu and torch.cuda.is_available():
        try:
            # 测试CUDA是否正常工作
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            torch.cuda.empty_cache()
            return "cuda", torch.float32
        except Exception as e:
            print(f"CUDA不可用: {e}")
            return "cpu", torch.float32
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
        "top_p": max(0.01, min(0.99, top_p)),   # 限制top_p范围
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
            temperature=0.5, top_p=0.85, repetition_penalty=1.2):
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


def main():
    # 自动查找最新的 checkpoint
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output/Qwen3-0.6B-ft1")
    latest_checkpoint = "checkpoint-1000"
    
    if os.path.isdir(output_dir):
        checkpoints = [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
            print(f"INFO: 自动找到最新的 checkpoint: {latest_checkpoint}")

    parser = argparse.ArgumentParser(description="Qwen3 推理脚本（修复版）")
    parser.add_argument("--input", "-i", type=str, help="用户输入的问题文本")
    parser.add_argument("--instruction", "-s", type=str, default=PROMPT, help="system 提示词")
    parser.add_argument("--checkpoint", "-c", type=str, default=latest_checkpoint,
                       help="checkpoint 路径")
    parser.add_argument("--max_new_tokens", "-m", type=int, default=MAX_NEW_TOKENS, 
                       help="生成的最大新token数")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU")
    args = parser.parse_args()

    # 检查 checkpoint 路径
    if not args.checkpoint or not os.path.isdir(args.checkpoint):
        print(f"\n错误：必须提供一个有效的 checkpoint 路径。")
        print(f"路径 '{args.checkpoint}' 无效或不存在。")
        exit(1)

    # 获取输入
    user_input = args.input
    if not user_input:
        try:
            user_input = input("请输入用户问题：").strip()
        except EOFError:
            user_input = ""

    # 选择设备
    device, dtype = select_device_and_dtype(force_cpu=args.force_cpu)
    print(f"使用设备: {device}, 数据类型: {dtype}")

    # 加载模型和分词器
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, 
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to("cpu")
    else:
        model = model.to(device)
    
    model.eval()
    
    # 检查模型健康状态
    if not check_model_health(model):
        print("警告: 模型权重可能存在问题")
    
    # 构建消息
    messages = [
        {"role": "system", "content": args.instruction},
        {"role": "user", "content": user_input},
    ]

    print("开始生成...")
    output = predict(messages, model, tokenizer, device)
    
    if output:
        print("\n" + "="*50)
        print("回答:")
        print(output)
        print("="*50)
    else:
        print("生成失败，请检查模型和输入")


if __name__ == "__main__":
    main()
