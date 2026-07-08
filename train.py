# train.py
# 模型训练脚本 - ResNet50 + 加权损失（解决类别不平衡）

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

# ========== 设置 CPU 线程数 ==========
torch.set_num_threads(16)
print(f"✅ PyTorch 已设置线程数: {torch.get_num_threads()}")

# ========== 解决 matplotlib 中文显示 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

from model import SkinLesionCNN


# ============================================================
# 1. 配置参数
# ============================================================

class Config:
    DATA_DIR = "data/HAM10000_Tree"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    EPOCHS = 25                  # 稍多几轮让加权损失充分收敛
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SAVE_DIR = "./checkpoints"
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_model.pth")
    CURVE_SAVE_PATH = "./training_curves.png"

    SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(Config.SEED)
print(f"使用设备: {Config.DEVICE}")


# ============================================================
# 2. 数据加载与预处理
# ============================================================

def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def load_datasets():
    train_transform, val_transform = get_data_transforms()
    if not os.path.exists(Config.DATA_DIR):
        raise FileNotFoundError(f"数据集目录不存在: {Config.DATA_DIR}")

    train_dataset = datasets.ImageFolder(
        root=os.path.join(Config.DATA_DIR, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(Config.DATA_DIR, "val"),
        transform=val_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(Config.DATA_DIR, "test"),
        transform=val_transform
    )

    print(f"\n训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    return train_loader, val_loader, test_loader


# ============================================================
# 3. 训练函数（与之前相同）
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="训练", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})
    return running_loss / total, 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="验证", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})
    return running_loss / total, 100 * correct / total


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 50}\nEpoch {epoch}/{epochs}\n{'=' * 50}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(Config.SAVE_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': train_loader.dataset.classes,
            }, Config.MODEL_SAVE_PATH)
            print(f"  ✅ 最佳模型已保存 ({val_acc:.2f}%)")

    return history, best_val_acc


def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history['train_loss'], 'b-o', label='训练损失')
    ax1.plot(epochs, history['val_loss'], 'r-o', label='验证损失')
    ax1.set_title('损失曲线')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, history['train_acc'], 'b-o', label='训练准确率')
    ax2.plot(epochs, history['val_acc'], 'r-o', label='验证准确率')
    ax2.set_title('准确率曲线')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"训练曲线已保存: {save_path}")
    plt.show()


# ============================================================
# 主函数（核心改动：加权损失）
# ============================================================

def main():
    print("=" * 60)
    print("皮肤病变识别系统 - ResNet50 + 加权损失训练")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        print("\n[1] 加载数据集...")
        train_dataset, val_dataset, test_dataset = load_datasets()
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset
        )

        print("\n[2] 创建 ResNet50 模型...")
        model = SkinLesionCNN(num_classes=len(train_dataset.classes))
        model = model.to(Config.DEVICE)
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

        print("\n[3] 计算类别权重（处理类别不平衡）...")
        # 获取训练集所有标签
        train_labels = train_dataset.targets
        classes = np.unique(train_labels)
        weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weights = torch.FloatTensor(weights).to(Config.DEVICE)
        print(f"类别权重: {dict(zip(classes, weights))}")

        print("\n[4] 配置加权损失函数和优化器...")
        criterion = nn.CrossEntropyLoss(weight=class_weights)  # 关键改动
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        print("\n[5] 开始训练...")
        history, best_acc = train(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            Config.DEVICE, Config.EPOCHS
        )

        print("\n[6] 绘制训练曲线...")
        plot_curves(history, Config.CURVE_SAVE_PATH)

        print("\n[7] 测试集评估...")
        if os.path.exists(Config.MODEL_SAVE_PATH):
            checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            test_loss, test_acc = validate(model, test_loader, criterion, Config.DEVICE)
            print(f"📊 测试集准确率: {test_acc:.2f}%")

        print("\n🎉 训练完成!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()