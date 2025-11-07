# C:\Users\Administrator\Desktop\PythonProject\mini_demos\conv_channel_alignment.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def conv_channel_demo():
    print("=== 卷积通道对齐实验 ===")

    # 设置随机种子保证可复现
    torch.manual_seed(42)

    # 构造输入张量: (batch, channels, height, width)
    x = torch.randn(1, 3, 8, 8)
    print(f"输入形状: {x.shape}")

    # 创建卷积层: 输入3通道，输出4通道，3x3卷积核
    conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, bias=False)
    print(f"卷积核形状: {conv.weight.shape}")  # (4, 3, 3, 3)

    # 前向传播
    y = conv(x)
    print(f"输出形状: {y.shape}")

    # 可视化卷积过程示意图
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 显示输入特征图的一个通道
    im1 = axes[0, 0].imshow(x[0, 0].detach().numpy(), cmap='viridis')
    axes[0, 0].set_title('输入通道1')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 显示一个卷积核
    kernel_vis = conv.weight[0, 0].detach().numpy()
    im2 = axes[0, 1].imshow(kernel_vis, cmap='hot')
    axes[0, 1].set_title('卷积核通道1')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # 显示输出特征图的一个通道
    im3 = axes[1, 0].imshow(y[0, 0].detach().numpy(), cmap='viridis')
    axes[1, 0].set_title('输出通道1')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # 显示形状变化示意图
    axes[1, 1].text(0.1, 0.8, f'输入: {tuple(x.shape)}', fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'卷积核: {tuple(conv.weight.shape)}', fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'输出: {tuple(y.shape)}', fontsize=12)
    axes[1, 1].text(0.1, 0.2, '计算: (8-3)//1 + 1 = 6', fontsize=12)
    axes[1, 1].set_title('形状演化')
    axes[1, 1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'conv_channel_alignment.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"实验完成! 图像已保存为 {output_path}")


if __name__ == "__main__":
    conv_channel_demo()