import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import time
from torchvision.models import VGG16_Weights
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def setup_data():
    """设置CIFAR-10数据加载器"""
    print("正在下载CIFAR-10数据集...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    print("数据集加载完成!")
    return trainloader, testloader


def create_vgg16():
    """创建VGG-16模型"""
    model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # 冻结特征层
    for param in model.features.parameters():
        param.requires_grad = False

    # 修改分类头
    model.classifier[6] = nn.Linear(4096, 10)
    model = model.to(device)

    return model


def train_single_seed(seed, epochs=50):
    """训练单个种子"""
    print(f"\n{'=' * 60}")
    print(f"开始训练种子: {seed}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # 创建日志目录
    log_dir = f'./logs/seed_{seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    # 初始化TensorBoard - 每个种子一个writer
    writer = SummaryWriter(log_dir=log_dir)

    # 获取数据和模型
    trainloader, testloader = setup_data() if seed == 42 else (None, None)
    model = create_vgg16()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.classifier.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 复用数据加载器（避免重复下载）
    if seed != 42:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    best_acc = 0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # 计算指标
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(trainloader)
        avg_val_loss = val_loss / len(testloader)

        # 记录到列表
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)

        # 添加直方图（每10个epoch记录一次）
        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                    writer.add_histogram(f'Weights/{name}', param, epoch)

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'seed': seed
            }, f'./models/vgg16_seed_{seed}_best.pth')

        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f'Seed {seed} | Epoch [{epoch + 1}/{epochs}] | '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
                  f'Best Acc: {best_acc:.2f}%')

    training_time = time.time() - start_time

    # 记录最终结果
    writer.add_hparams(
        {'seed': seed, 'lr': 0.01, 'batch_size': 128, 'epochs': epochs},
        {'hparam/best_accuracy': best_acc, 'hparam/best_epoch': best_epoch}
    )

    writer.close()

    print(f"\n种子 {seed} 训练完成!")
    print(f"训练时间: {training_time:.2f}秒")
    print(f"最佳验证准确率: {best_acc:.2f}% (第{best_epoch}轮)")

    return {
        'seed': seed,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }


def create_comparison_tensorboard(all_results):
    """创建对比TensorBoard日志"""
    comparison_writer = SummaryWriter(log_dir='./logs/comparison')

    # 找到最长的训练记录
    max_epochs = max(len(result['train_accs']) for result in all_results)

    for epoch in range(max_epochs):
        for result in all_results:
            seed = result['seed']
            if epoch < len(result['train_accs']):
                # 对比训练准确率
                comparison_writer.add_scalars('Comparison/Train_Accuracy',
                                              {f'Seed_{seed}': result['train_accs'][epoch]}, epoch)
                # 对比验证准确率
                comparison_writer.add_scalars('Comparison/Validation_Accuracy',
                                              {f'Seed_{seed}': result['val_accs'][epoch]}, epoch)
                # 对比训练损失
                comparison_writer.add_scalars('Comparison/Train_Loss',
                                              {f'Seed_{seed}': result['train_losses'][epoch]}, epoch)

    comparison_writer.close()


def generate_result_card(all_results):
    """生成结果卡片"""
    print(f"\n{'=' * 80}")
    print("VGG-16 CIFAR-10 训练结果汇总")
    print(f"{'=' * 80}")

    # 表格头
    print("\n| 随机种子 | 最佳准确率 (%) | 最佳轮次 | 最终训练准确率 (%) | 最终验证准确率 (%) | 训练时间 (秒) |")
    print("|----------|----------------|----------|-------------------|-------------------|--------------|")

    for result in all_results:
        print(f"| {result['seed']:8} | {result['best_acc']:14.2f} | {result['best_epoch']:8} | "
              f"{result['final_train_acc']:17.2f} | {result['final_val_acc']:17.2f} | "
              f"{result['training_time']:12.2f} |")

    # 计算统计信息
    best_accs = [result['best_acc'] for result in all_results]
    mean_acc = np.mean(best_accs)
    std_acc = np.std(best_accs)

    print(f"\n**统计结果**:")
    print(f"- 平均最佳准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"- 准确率范围: {min(best_accs):.2f}% - {max(best_accs):.2f}%")
    print(f"- 总训练时间: {sum(result['training_time'] for result in all_results):.2f}秒")

    # 保存结果到文件
    with open('./logs/training_results.md', 'w') as f:
        f.write("# VGG-16 CIFAR-10 训练结果\n\n")
        f.write("| 随机种子 | 最佳准确率 (%) | 最佳轮次 | 最终训练准确率 (%) | 最终验证准确率 (%) | 训练时间 (秒) |\n")
        f.write("|----------|----------------|----------|-------------------|-------------------|--------------|\n")
        for result in all_results:
            f.write(f"| {result['seed']} | {result['best_acc']:.2f} | {result['best_epoch']} | "
                    f"{result['final_train_acc']:.2f} | {result['final_val_acc']:.2f} | "
                    f"{result['training_time']:.2f} |\n")
        f.write(f"\n**平均准确率**: {mean_acc:.2f}% ± {std_acc:.2f}%\n")


def main():
    """主函数"""
    print("=== VGG-16 TensorBoard 三种子训练开始 ===")
    print(f"设备: {device}")
    print(f"随机种子: [42, 123, 456]")
    print(f"训练轮次: 50")

    # 确保目录存在
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    # 训练三个种子
    seeds = [42, 123, 456]
    all_results = []

    for seed in seeds:
        result = train_single_seed(seed, epochs=50)
        all_results.append(result)

    # 创建对比TensorBoard日志
    create_comparison_tensorboard(all_results)

    # 生成结果卡片
    generate_result_card(all_results)

    print(f"\n{'=' * 80}")
    print("训练完成!")
    print(f"{'=' * 80}")
    print("TensorBoard日志结构:")
    print("  ./logs/seed_42/      - 种子42的详细训练日志")
    print("  ./logs/seed_123/     - 种子123的详细训练日志")
    print("  ./logs/seed_456/     - 种子456的详细训练日志")
    print("  ./logs/comparison/   - 三个种子的对比日志")
    print("\n启动TensorBoard命令:")
    print("  tensorboard --logdir=./logs --port=6006")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()