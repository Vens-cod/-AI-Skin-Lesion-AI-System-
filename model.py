# model.py
# 使用预训练 ResNet50 作为骨干网络（更强特征提取）

import torch
import torch.nn as nn
import torchvision.models as models
import ssl

# ========== 临时禁用 SSL 证书验证，解决 Windows 证书问题 ==========
ssl._create_default_https_context = ssl._create_unverified_context


class SkinLesionCNN(nn.Module):
    """
    皮肤病变识别网络 - 基于 ResNet50 预训练模型
    输入：3x224x224
    输出：7类
    """

    def __init__(self, num_classes=7):
        super(SkinLesionCNN, self).__init__()

        # 加载预训练好的 ResNet50（使用 weights 参数）
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # ResNet50 最后一层全连接层的输入特征数是 2048
        in_features = self.backbone.fc.in_features

        # 替换最后一层全连接层，改为输出 7 类
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# 测试代码
if __name__ == "__main__":
    model = SkinLesionCNN(num_classes=7)
    print("✅ ResNet50 模型创建成功")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"输出形状: {output.shape}")