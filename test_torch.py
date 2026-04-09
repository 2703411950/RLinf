import torch

# 检查 PyTorch 版本
print(f"PyTorch Version: {torch.__version__}")

# 检查 CUDA 是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")

# 检查当前 PyTorch 编译所用的 CUDA 版本
print(f"CUDA Version (PyTorch): {torch.version.cuda}")

# 尝试在 GPU 上创建一个张量（如果不报错，说明基础兼容性没问题）
x = torch.rand(3, 3).cuda()
print("Tensor created successfully on GPU!")