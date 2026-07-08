# utils.py
# 辅助工具函数

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from config import CLASS_NAMES_CN, BENIGN_MALIGN

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题


def plot_probs(probs, class_names=None):
    """
    绘制7类概率的条形图

    参数:
        probs: 7个类别的概率列表 (长度为7)
        class_names: 类别中文名列表（默认为 CLASS_NAMES_CN）

    返回:
        matplotlib.figure.Figure
    """
    if class_names is None:
        class_names = CLASS_NAMES_CN

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 4))

    # 颜色：最高概率用深绿色，其他用浅蓝色
    colors = ['#2ecc71' if i == np.argmax(probs) else '#3498db' for i in range(len(probs))]

    # 绘制水平条形图
    bars = ax.barh(class_names, probs, color=colors, edgecolor='white', linewidth=0.5)

    # 在条形末端显示百分比数值
    for bar, prob in zip(bars, probs):
        if prob > 0.01:  # 只显示大于1%的概率
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{prob:.1%}', va='center', fontsize=9, fontweight='bold')

    # 设置X轴为百分比格式
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('概率', fontsize=10)
    ax.set_title('7类皮肤病变概率分布', fontsize=12, fontweight='bold')

    # 美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    return fig


def get_benign_malign_tip(class_name_cn):
    """
    获取良恶性提示

    参数:
        class_name_cn: 类别中文名

    返回:
        str: 提示文本
    """
    return BENIGN_MALIGN.get(class_name_cn, '未知')


def format_probs_to_text(probs, class_names=None):
    """
    将概率列表格式化为可读文本（用于Gradio显示）

    返回:
        str: 格式化后的文本
    """
    if class_names is None:
        class_names = CLASS_NAMES_CN

    lines = []
    # 按概率降序排序
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    for i, idx in enumerate(sorted_indices):
        marker = '🏆' if i == 0 else f'{i + 1}.'
        lines.append(f"{marker} {class_names[idx]}: {probs[idx]:.2%}")
    return "\n".join(lines)