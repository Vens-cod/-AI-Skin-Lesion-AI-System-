# inference.py
# 推理引擎 - 加载模型、预处理、预测（支持 TTA 测试时增强）

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from model import SkinLesionCNN
from config import CLASS_NAMES_EN, CLASS_NAMES_CN

# ========== 全局配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model.pth"
IMG_SIZE = 224

# ========== 推理时的预处理（与训练时完全一致）==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ========== 模型加载 ==========
def load_model(model_path=MODEL_PATH, device=DEVICE):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # ====== 关键修复：从 checkpoint 读取类别顺序 ======
    class_names = checkpoint.get('class_names', CLASS_NAMES_EN)
    num_classes = len(class_names)

    print(f"✅ 加载的类别顺序: {class_names}")

    model = SkinLesionCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✅ 模型加载成功 (验证准确率: {checkpoint.get('val_acc', 'N/A'):.2f}%)")
    return model, class_names


# ========== 图像预处理 ==========
def preprocess_image(image):
    """将输入图像转换为模型可接受的张量"""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = Image.fromarray(image.astype(np.uint8), mode='L')
        elif image.ndim == 3 and image.shape[2] == 3:
            image = Image.fromarray(image.astype(np.uint8), mode='RGB')
        else:
            raise ValueError(f"不支持的图像维度: {image.ndim}")
    elif isinstance(image, Image.Image):
        pass
    else:
        raise TypeError(f"不支持的图像类型: {type(image)}")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor


# ========== TTA 增强函数 ==========
def get_tta_transforms(tensor):
    """对输入张量生成多个增强版本（用于 TTA）"""
    tta_list = []
    tta_list.append(tensor)
    tta_list.append(torch.flip(tensor, dims=[3]))
    tta_list.append(torch.flip(tensor, dims=[2]))
    tta_list.append(torch.flip(torch.flip(tensor, dims=[3]), dims=[2]))
    tta_list.append(torch.rot90(tensor, k=1, dims=[2, 3]))
    tta_list.append(torch.rot90(tensor, k=3, dims=[2, 3]))
    tta_list.append(torch.flip(torch.rot90(tensor, k=1, dims=[2, 3]), dims=[3]))
    tta_list.append(torch.flip(torch.rot90(tensor, k=3, dims=[2, 3]), dims=[3]))
    return tta_list


# ========== 推理预测（支持 TTA）==========
def predict(image, model=None, device=DEVICE, use_tta=True):
    """
    执行推理，返回预测结果和概率
    """
    if model is None:
        model, class_names = load_model(device=device)
    else:
        # 如果 model 已传入但没有 class_names，需要获取
        if not hasattr(predict, '_cached_class_names'):
            _, class_names = load_model(device=device)
            predict._cached_class_names = class_names
        else:
            class_names = predict._cached_class_names

    # 预处理
    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)

    if use_tta:
        tta_tensors = get_tta_transforms(input_tensor)
        all_probs = []
        with torch.no_grad():
            for aug_tensor in tta_tensors:
                outputs = model(aug_tensor)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy().flatten())
        avg_probs = np.mean(all_probs, axis=0)
        probs_np = avg_probs
        max_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[max_idx])
    else:
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
        probs_np = probs.cpu().numpy().flatten()
        max_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs_np[max_idx])

    # ====== 关键修复：使用从 checkpoint 加载的 class_names ======
    pred_class_en = class_names[max_idx]

    # 查找对应的中文名（从 config 映射）
    try:
        idx = CLASS_NAMES_EN.index(pred_class_en)
        pred_class_cn = CLASS_NAMES_CN[idx]
    except ValueError:
        pred_class_cn = pred_class_en

    # 概率字典使用 config 中的标准顺序（便于展示）
    probs_dict = {}
    for cls_en, prob in zip(class_names, probs_np):
        probs_dict[cls_en] = float(prob)

    return {
        'pred_class_en': pred_class_en,
        'pred_class_cn': pred_class_cn,
        'confidence': confidence,
        'probs': probs_np.tolist(),
        'probs_dict': probs_dict
    }


# ========== 单例缓存 ==========
_cached_model = None
_cached_class_names = None


def get_model():
    global _cached_model, _cached_class_names
    if _cached_model is None:
        _cached_model, _cached_class_names = load_model()
    return _cached_model, _cached_class_names


def predict_cached(image, use_tta=True):
    """使用缓存的模型进行预测"""
    model, class_names = get_model()
    # 将 class_names 缓存到 predict 函数
    predict._cached_class_names = class_names
    return predict(image, model=model, use_tta=use_tta)


# ========== 随机抽样测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("随机抽样测试 - 验证单张预测与整体准确率是否一致")
    print("=" * 60)

    try:
        model, class_names = load_model()
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        exit(1)

    print(f"模型类别顺序: {class_names}")

    import random

    # 加载测试集路径
    test_dir = "data/HAM10000_Tree/test"

    all_images = []
    for cls_name in os.listdir(test_dir):
        cls_path = os.path.join(test_dir, cls_name)
        if os.path.isdir(cls_path):
            for img_name in os.listdir(cls_path):
                if img_name.endswith('.jpg'):
                    all_images.append((os.path.join(cls_path, img_name), cls_name))

    sample_size = min(20, len(all_images))
    random.seed(42)
    samples = random.sample(all_images, sample_size)

    print(f"\n随机抽取 {sample_size} 张图片进行测试：")
    correct = 0
    for img_path, true_label in samples:
        img = Image.open(img_path).convert('RGB')
        result = predict(img, model=model, use_tta=True)
        pred = result['pred_class_en']
        is_correct = (pred == true_label)
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"  {status} 真实: {true_label:6s} | 预测: {pred:6s} | 置信度: {result['confidence']:.2%}")

    print(f"\n{'=' * 60}")
    print(f"抽样准确率: {correct}/{sample_size} ({100 * correct / sample_size:.2f}%)")
    print("=" * 60)