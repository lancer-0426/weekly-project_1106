# Mini Demos

本目录包含三个核心神经网络概念的最小可复现实验。

## 环境要求
- Python 3.7+
- PyTorch 1.8+
- Matplotlib
- NumPy

## 实验列表

###
```bash
cd mini-demos
python conv_channel_alignment.py
=== 卷积通道对齐实验 ===
输入形状: torch.Size([1, 3, 8, 8])
卷积核形状: torch.Size([4, 3, 3, 3])
输出形状: torch.Size([1, 4, 6, 6])
实验完成! 图像已保存为 conv_channel_alignment.png

python pooling_invariance.py
=== 池化不变性实验 ===
原始图像形状: torch.Size([1, 1, 6, 6])
平移图像形状: torch.Size([1, 1, 6, 6])
池化后原始: torch.Size([1, 1, 3, 3])
池化后平移: torch.Size([1, 1, 3, 3])
池化后差异最大值: 0.0000
实验完成! 图像已保存为 pooling_invariance.png

python activation_nonlinearity.py
=== 激活非线性实验 ===
输入范围: [-3.0, 3.0]
ReLU输出范围: [0.0, 3.0]
线性变换后范围: [-1.5, 1.5]
线性+ReLU后范围: [0.0, 1.5]
实验完成! 图像已保存为 activation_nonlinearity.png