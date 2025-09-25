import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import warnings
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings("ignore", message="The pynvml package is deprecated.")

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_NEW_TOKENS = 2048
RETRY_ATTEMPTS = 3


def select_device_and_dtype(force_cpu=False):
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



def clean_probabilities(probs):
    probs = torch.where(torch.isinf(probs) | torch.isnan(probs), torch.tensor(0.0, device=probs.device), probs)
    probs = torch.where(probs < 0, torch.tensor(0.0, device=probs.device), probs)
    if probs.sum() == 0:
        probs = torch.ones_like(probs) / probs.numel()
    else:
        probs = probs / probs.sum()
    return probs


def predict(messages, model, tokenizer, device, attempt=0,
            temperature=0.5, top_p=0.85, repetition_penalty=1.2):
    if attempt >= RETRY_ATTEMPTS:
        print(f"已达到最大重试次数({RETRY_ATTEMPTS})，无法生成有效回答")
        return None

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None

    try:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=False,
            return_dict_in_generate=False,
        )

        new_tokens = generated[:, input_ids.shape[1]:]
        response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        # 清理可能的重复感叹号
        if response.count('!') > len(response) * 0.5:
            print("检测到异常输出，尝试重新生成...")
            return predict(messages, model, tokenizer)

        return response
    except RuntimeError as e:
        print(f"生成过程中发生错误: {e}")
        print("尝试使用CPU进行生成...")
        model = model.to("cpu")
        input_ids = input_ids.to("cpu")
        attention_mask


if __name__ == "__main__":
    # 自动查找最新的 checkpoint
    # 构造相对于脚本所在目录的路径，使其不受运行位置的影响
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output/Qwen3-0.6B")
    latest_checkpoint = "checkpoint-1000"
    if os.path.isdir(output_dir):
        checkpoints = [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            # 通过 checkpoint 编号排序找到最新的
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
            print(f"INFO: 自动找到最新的 checkpoint: {latest_checkpoint}")

    parser = argparse.ArgumentParser(description="Qwen3 推理脚本（命令行读取）")
    parser.add_argument("--input", "-i", type=str, help="用户输入的问题文本。如果不提供，将在命令行交互读取。")
    parser.add_argument("--instruction", "-s", type=str, default=PROMPT, help="system 提示词")
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=latest_checkpoint,
        help=f"checkpoint 路径。默认为自动查找的最新 checkpoint"
    )
    parser.add_argument("--max_new_tokens", "-m", type=int, default=MAX_NEW_TOKENS, help="生成的最大新token数")
    args = parser.parse_args()

    # 检查 checkpoint 路径是否有效
    if not args.checkpoint or not os.path.isdir(args.checkpoint):
        print(f"\n错误：必须提供一个有效的 checkpoint 路径。")
        print(f"路径 '{args.checkpoint}' 无效或不存在。")
        print(f"请确保您已经在 '{output_dir}' 目录下完成了训练并生成了 checkpoint，或通过 -c 参数手动指定。")
        exit(1)

    # 覆盖默认最大生成长度（如用户提供）
    MAX_NEW_TOKENS = args.max_new_tokens

    # 获取输入
    user_input = args.input
    if not user_input:
        try:
            user_input = input("请输入用户问题：").strip()
        except EOFError:
            user_input = ""

    device, dtype = select_device_and_dtype()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=dtype)
    model.to(device)
    model.eval()

    messages = [
        {"role": "system", "content": args.instruction},
        {"role": "user", "content": user_input},
    ]

    output = predict(messages, model, tokenizer,device)
    print(output)


