# C:\Users\Administrator\Desktop\PythonProject\mini_demos\activation_nonlinearity.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def activation_nonlinearity_demo():
    print("=== 激活非线性实验 ===")

    torch.manual_seed(42)

    # 生成测试数据
    z = torch.linspace(-3, 3, 100).reshape(-1, 1)

    # 应用ReLU激活函数
    relu = nn.ReLU()
    a = relu(z)

    # 模拟线性层前后的变化
    linear = nn.Linear(1, 1, bias=False)
    linear.weight.data = torch.tensor([[0.5]])  # 固定权重便于演示

    z_before = z
    z_after = linear(z)
    a_after = relu(z_after)

    print(f"输入范围: [{z.min().item():.1f}, {z.max().item():.1f}]")
    print(f"ReLU输出范围: [{a.min().item():.1f}, {a.max().item():.1f}]")
    print(f"线性变换后范围: [{z_after.min().item():.1f}, {z_after.max().item():.1f}]")
    print(f"线性+ReLU后范围: [{a_after.min().item():.1f}, {a_after.max().item():.1f}]")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ReLU函数图像
    axes[0, 0].plot(z.detach().numpy(), a.detach().numpy(), 'b-', linewidth=2)
    axes[0, 0].set_xlabel('输入 z')
    axes[0, 0].set_ylabel('输出 a = ReLU(z)')
    axes[0, 0].set_title('ReLU激活函数')
    axes[0, 0].grid(True, alpha=0.3)

    # 线性变换前后散点图
    axes[0, 1].scatter(z_before.detach().numpy(), z_after.detach().numpy(), alpha=0.6, c='red')
    axes[0, 1].plot([-3, 3], [-1.5, 1.5], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel('线性层前 z')
    axes[0, 1].set_ylabel('线性层后 z\'')
    axes[0, 1].set_title('线性变换: z\' = 0.5 × z')
    axes[0, 1].grid(True, alpha=0.3)

    # 激活前后对比
    axes[1, 0].scatter(z_after.detach().numpy(), a_after.detach().numpy(), alpha=0.6, c='green')
    axes[1, 0].plot([-1.5, 0, 1.5], [0, 0, 1.5], 'b-', linewidth=2)
    axes[1, 0].set_xlabel('线性层输出 z\'')
    axes[1, 0].set_ylabel('ReLU输出 a\'')
    axes[1, 0].set_title('非线性引入: a\' = ReLU(z\')')
    axes[1, 0].grid(True, alpha=0.3)

    # 完整流程形状演化
    x_demo = torch.randn(1, 1, 5, 5)
    conv_demo = nn.Conv2d(1, 2, kernel_size=3)
    relu_demo = nn.ReLU()
    pool_demo = nn.MaxPool2d(2)

    shapes = []
    operations = []

    # 卷积
    x_conv = conv_demo(x_demo)
    shapes.append(x_conv.shape)
    operations.append("卷积后")

    # ReLU
    x_relu = relu_demo(x_conv)
    shapes.append(x_relu.shape)
    operations.append("ReLU后")

    # 池化
    x_pool = pool_demo(x_relu)
    shapes.append(x_pool.shape)
    operations.append("池化后")

    # 形状演化图
    for i, (op, shape) in enumerate(zip(operations, shapes)):
        axes[1, 1].text(0.1, 0.8 - i * 0.2, f'{op}: {tuple(shape)}', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Conv→ReLU→Pool形状演化')
    axes[1, 1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'activation_nonlinearity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"实验完成! 图像已保存为 {output_path}")


if __name__ == "__main__":
    activation_nonlinearity_demo()