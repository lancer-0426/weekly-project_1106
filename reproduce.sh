#!/bin/bash

# VGG-16 CIFAR-10 一键复现脚本
set -e  # 遇到错误立即退出

echo "================================================"
echo "   VGG-16 CIFAR-10 复现脚本"
echo "================================================"
reproduce.sh
# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python，请先安装Python 3.8+"
    exit 1
fi

# 检查Python版本
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python版本: $PYTHON_VERSION"

# 检查并安装依赖
echo ""
echo "1. 检查环境依赖..."
REQUIRED_PACKAGES=("torch" "torchvision" "tensorboard")

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" &> /dev/null; then
        echo "   ✅ $package 已安装"
    else
        echo "   ❌ $package 未安装，正在安装..."
        pip install $package
    fi
done

# 显示版本信息
echo ""
echo "2. 环境版本信息:"
python -c "import torch; print(f'   PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'   Torchvision: {torchvision.__version__}')"
python -c "import tensorboard; print(f'   TensorBoard: {tensorboard.__version__}')"

# 检查CUDA
echo ""
echo "3. 硬件信息:"
python -c "import torch; print(f'   CUDA可用: {torch.cuda.is_available()}')"
if torch.cuda.is_available(); then
    python -c "import torch; print(f'   GPU设备: {torch.cuda.get_device_name(0)}')"
fi

# 创建必要的目录
echo ""
echo "4. 创建项目目录..."
mkdir -p logs
mkdir -p models
mkdir -p data

# 开始训练
echo ""
echo "5. 开始训练 VGG-16..."
echo "   训练脚本: train_vgg16_tensorboard.py"
echo "   随机种子: [42, 123, 456]"
echo "   训练轮次: 50"
echo "================================================"

# 运行训练脚本
python train_vgg16_tensorboard.py

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ 训练完成!"
    echo "================================================"

    # 显示训练结果摘要
    if [ -f "./logs/training_results.md" ]; then
        echo ""
        echo "训练结果摘要:"
        cat ./logs/training_results.md
    fi

    echo ""
    echo "📊 可视化训练结果:"
    echo "   运行: tensorboard --logdir=./logs"
    echo "   然后在浏览器打开: http://localhost:6006"

    echo ""
    echo "📁 生成的文件:"
    echo "   📂 logs/          - TensorBoard 日志"
    echo "   📂 models/        - 训练好的模型权重"
    echo "   📂 data/          - CIFAR-10 数据集"

else
    echo ""
    echo "❌ 训练失败，请检查错误信息"
    exit 1
fi

echo ""
echo "================================================"
echo "🎉 复现完成！感谢使用 VGG-16 CIFAR-10 项目"
echo "================================================"