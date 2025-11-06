import torch
import torchvision
import sys

print("=== 环境版本信息 ===")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"Torchvision版本: {torchvision.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")