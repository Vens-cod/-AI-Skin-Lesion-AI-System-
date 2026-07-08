# check_performance.py
# 训练前性能测试 - 估算每轮训练耗时

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 导入模型和配置
from model import SkinLesionCNN
from train import Config

# ============================================================
# 1. 硬件信息检测
# ============================================================

print("=" * 60)
print("硬件信息检测")
print("=" * 60)

# CPU核心数
physical_cores = os.cpu_count()
print(f"CPU 逻辑核心数: {physical_cores}")

# PyTorch 当前使用的线程数
current_threads = torch.get_num_threads()
print(f"PyTorch 当前线程数: {current_threads}")

# 建议优化：CPU 训练时，线程数设为物理核心数能获得最佳性能
recommended_threads = physical_cores
print(f"建议线程数: {recommended_threads}")

# 检测是否有 GPU
if torch.cuda.is_available():
    print(f"✅ GPU 可用: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU 不可用，将使用 CPU 训练")

# 检测 MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✅ Apple MPS (Metal) 可用")
else:
    print("❌ MPS 不可用")


# ============================================================
# 2. 模拟训练性能测试
# ============================================================

def test_training_speed():
    """模拟训练过程，估算每轮耗时"""

    print("\n" + "=" * 60)
    print("性能测试 - 模拟训练")
    print("=" * 60)

    # 配置参数（与 train.py 保持一致）
    BATCH_SIZE = Config.BATCH_SIZE
    IMG_SIZE = Config.IMG_SIZE
    NUM_CLASSES = 7
    DEVICE = torch.device("cpu")  # 强制使用 CPU 测试

    # 创建模拟数据：7000张图像（约等于 HAM10000 训练集大小）
    NUM_SAMPLES = 7000
    NUM_BATCHES = NUM_SAMPLES // BATCH_SIZE
    print(f"模拟训练集样本数: {NUM_SAMPLES}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"每轮批次数: {NUM_BATCHES}")

    # 生成随机数据（模拟真实数据）
    print("\n生成模拟数据...")
    fake_images = torch.randn(NUM_SAMPLES, 3, IMG_SIZE, IMG_SIZE)
    fake_labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))

    # 创建 DataLoader
    dataset = TensorDataset(fake_images, fake_labels)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0  # CPU训练建议用0，避免额外开销
    )

    # 创建模型
    print("创建模型...")
    model = SkinLesionCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 预热（让CPU缓存和PyTorch初始化完成）
    print("\n预热中（忽略本次耗时）...")
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break  # 只跑一个batch预热

    # 正式测试
    print("\n开始正式测试（模拟1轮）...")
    model.train()
    start_time = time.time()

    batch_times = []
    for batch_idx, (images, labels) in enumerate(dataloader):
        batch_start = time.time()

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)

        # 每50个batch打印一次进度
        if (batch_idx + 1) % 50 == 0:
            avg_time = sum(batch_times[-50:]) / len(batch_times[-50:])
            print(f"  Batch {batch_idx + 1}/{NUM_BATCHES} | "
                  f"平均耗时: {avg_time:.3f}s/batch | "
                  f"预计剩余: {avg_time * (NUM_BATCHES - batch_idx - 1):.1f}s")

    total_time = time.time() - start_time

    # ============================================================
    # 3. 结果汇总
    # ============================================================

    print("\n" + "=" * 60)
    print("性能测试结果")
    print("=" * 60)

    avg_batch_time = sum(batch_times) / len(batch_times)
    print(f"平均每 Batch 耗时: {avg_batch_time:.3f} 秒")
    print(f"一轮总耗时 ({NUM_BATCHES} batches): {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

    # 根据测试结果推算完整训练时间
    EPOCHS = Config.EPOCHS
    estimated_total = total_time * EPOCHS

    print("\n" + "=" * 60)
    print("📊 训练时间预估")
    print("=" * 60)
    print(f"当前配置: {EPOCHS} 轮")
    print(f"预估总训练时间: {estimated_total:.2f} 秒 ({estimated_total / 60:.2f} 分钟)")

    if estimated_total > 3600:
        print(f"  ⚠️ 超过1小时 ({estimated_total / 3600:.1f} 小时)，建议:")
        print(f"    - 减小 BATCH_SIZE 到 16 或 8（虽然会稍微减慢每轮速度，但内存更安全）")
        print(f"    - 减少 EPOCHS 到 15-20（如果过拟合提前发生）")
        print(f"    - 增加 NUM_WORKERS = 0（避免多进程开销）")
    elif estimated_total > 600:
        print(f"  ⏱️ 约 {estimated_total / 60:.1f} 分钟，可以接受")
    else:
        print(f"  ✅ 速度很快，不到 {estimated_total / 60:.1f} 分钟")

    # 线程优化建议
    if current_threads != recommended_threads:
        print(f"\n💡 建议设置 PyTorch 线程数为 {recommended_threads} 以充分利用 CPU")
        print(f"   在 train.py 开头添加: torch.set_num_threads({recommended_threads})")

    return avg_batch_time, total_time


# ============================================================
# 4. 测试不同线程数的效果（可选）
# ============================================================

def test_thread_scaling():
    """测试不同线程数对速度的影响（可选）"""
    print("\n" + "=" * 60)
    print("线程数伸缩性测试")
    print("=" * 60)
    print("（仅测试单batch速度，快速评估）")

    DEVICE = torch.device("cpu")
    model = SkinLesionCNN(num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 单batch数据
    images = torch.randn(32, 3, 224, 224).to(DEVICE)
    labels = torch.randint(0, 7, (32,)).to(DEVICE)

    thread_counts = [1, 2, 4, os.cpu_count()]
    thread_counts = [t for t in thread_counts if t <= os.cpu_count()]
    thread_counts = sorted(set(thread_counts))

    results = []
    for num_threads in thread_counts:
        torch.set_num_threads(num_threads)

        # 预热
        for _ in range(5):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 正式测试
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(30):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elapsed = time.time() - start

        results.append((num_threads, elapsed / 30))
        print(f"  线程数 {num_threads}: {elapsed / 30:.4f} 秒/batch")

    # 找出最佳线程数
    best = min(results, key=lambda x: x[1])
    print(f"\n✅ 最佳线程数: {best[0]} (速度: {best[1]:.4f}s/batch)")

    # 恢复推荐线程数
    torch.set_num_threads(os.cpu_count())
    return dict(results)


if __name__ == "__main__":
    # 设置 PyTorch 线程数为 CPU 核心数
    recommended = os.cpu_count()
    torch.set_num_threads(recommended)
    print(f"已设置 PyTorch 线程数为: {recommended}")

    # 运行主测试
    avg_time, epoch_time = test_training_speed()

    # 可选：运行线程伸缩测试（耗时约30秒）
    run_scaling_test = input("\n是否运行线程伸缩测试？(y/n, 约需30秒): ").strip().lower()
    if run_scaling_test == 'y':
        test_thread_scaling()

    print("\n" + "=" * 60)
    print("测试完成！根据结果调整 train.py 中的参数后开始训练。")
    print("=" * 60)