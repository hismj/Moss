import torch

# 检查CUDA是否可用
print("CUDA available:", torch.cuda.is_available())

# 获取可用的设备数
print("Available GPUs:", torch.cuda.device_count())

# 获取当前设备
print("Current CUDA device:", torch.cuda.current_device())

# 获取当前设备的名称
print("CUDA device name:", torch.cuda.get_device_name(0))
