import torch

# 查看 PyTorch 版本
print("PyTorch 版本:", torch.__version__)

# 查看 CUDA 是否可用，以及对应的版本
print("是否支持 CUDA:", torch.cuda.is_available())
print("编译时的 CUDA 版本:", torch.version.cuda)
print("运行时的 cuDNN 版本:", torch.backends.cudnn.version())

# 如果有显卡，输出显卡信息
if torch.cuda.is_available():
    print("GPU 数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} 名称:", torch.cuda.get_device_name(i))
        print(f"GPU {i} 计算能力:", torch.cuda.get_device_capability(i))

def select_precision():
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] >= 7:  # Volta架构及以上
            return "fp16"  # 支持Tensor Cores
        else:
            return "fp32"
    else:
        return "fp32"  # CPU训练通常用fp32

print(f"选择的精度为{i}:", select_precision())