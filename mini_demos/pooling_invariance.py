# C:\Users\Administrator\Desktop\PythonProject\mini_demos\pooling_invariance.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def pooling_invariance_demo():
    print("=== 池化不变性实验 ===")

    torch.manual_seed(42)

    # 创建测试图像
    original_img = torch.randn(1, 1, 6, 6)

    # 创建平移1像素的图像
    shifted_img = torch.roll(original_img, shifts=1, dims=3)  # 水平平移

    # 最大池化层
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    # 对两张图分别池化
    original_pooled = maxpool(original_img)
    shifted_pooled = maxpool(shifted_img)

    print(f"原始图像形状: {original_img.shape}")
    print(f"平移图像形状: {shifted_img.shape}")
    print(f"池化后原始: {original_pooled.shape}")
    print(f"池化后平移: {shifted_pooled.shape}")

    # 计算差异
    original_shifted = torch.roll(original_pooled, shifts=1, dims=3)
    diff = torch.abs(shifted_pooled - original_shifted)
    print(f"池化后差异最大值: {diff.max().item():.4f}")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 原始图像
    im1 = axes[0, 0].imshow(original_img[0, 0].detach().numpy(), cmap='viridis')
    axes[0, 0].set_title('原始图像 (6×6)')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 平移图像
    im2 = axes[0, 1].imshow(shifted_img[0, 0].detach().numpy(), cmap='viridis')
    axes[0, 1].set_title('平移1像素图像')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # 差异
    im3 = axes[0, 2].imshow(torch.abs(original_img - shifted_img)[0, 0].detach().numpy(), cmap='hot')
    axes[0, 2].set_title('平移前后差异')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # 池化后原始
    im4 = axes[1, 0].imshow(original_pooled[0, 0].detach().numpy(), cmap='viridis')
    axes[1, 0].set_title('原始池化后 (3×3)')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    # 池化后平移
    im5 = axes[1, 1].imshow(shifted_pooled[0, 0].detach().numpy(), cmap='viridis')
    axes[1, 1].set_title('平移池化后')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

    # 池化后差异
    im6 = axes[1, 2].imshow(diff[0, 0].detach().numpy(), cmap='hot')
    axes[1, 2].set_title('池化后差异')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'pooling_invariance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"实验完成! 图像已保存为 {output_path}")


if __name__ == "__main__":
    pooling_invariance_demo()