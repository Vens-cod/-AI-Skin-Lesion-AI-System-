# app.py
# 表现层 - Gradio Web 交互界面（真实推理版）

import os
import gradio as gr
from datetime import datetime, timedelta

from config import CLASS_NAMES_EN, CLASS_NAMES_CN
from utils import plot_probs, get_benign_malign_tip, format_probs_to_text
from database import db
from inference import predict_cached  # 真实推理引擎


# ============================================================
# 识别函数（真实推理 + 数据库入库）
# ============================================================

def recognize_image(image):
    """
    识别图像的核心函数（真实推理）
    """
    if image is None:
        return "⚠️ 请先上传一张图片！", None, ""

    try:
        # 1. 执行真实推理（使用缓存模型，速度快）
        result = predict_cached(image)

        pred_class_en = result['pred_class_en']
        pred_class_cn = result['pred_class_cn']
        confidence = result['confidence']

        # ====== 关键修复：直接使用 inference 返回的 probs_dict ======
        probs_dict = result['probs_dict']  # 键是字母序的类别名

        # ====== 提取按 CLASS_NAMES_EN 顺序排列的概率列表（用于显示） ======
        probs_display = [probs_dict[cls] for cls in CLASS_NAMES_EN]

        # 3. 存入 MySQL 数据库（使用正确的 probs_dict）
        record_id = db.insert_record(
            predicted_class=pred_class_en,
            predicted_class_cn=pred_class_cn,
            confidence=confidence,
            all_probs=probs_dict,  # 存字典，键是类别名
            image_path=None,
            ip_address="192.168.100.3"
        )

        # 4. 生成结果文本（使用 probs_display）
        result_text = f"""
        ## 🏥 识别结果

        **预测类别**: {pred_class_cn} ({pred_class_en})

        **置信度**: {confidence:.2%}

        **记录ID**: {record_id}

        ---
        ### 全部概率排名
        {format_probs_to_text(probs_display)}
        """

        # 5. 生成概率条形图（使用 probs_display）
        fig = plot_probs(probs_display)

        # 6. 良恶性提示
        tip = get_benign_malign_tip(pred_class_cn)
        tip_text = f"💡 **提示**: 该病变为 **{tip}**"
        if '恶性' in tip or '癌前' in tip:
            tip_text += " ⚠️ 建议及时就医咨询"
        else:
            tip_text += " ✅ 请继续关注皮肤健康"

        return result_text, fig, tip_text

    except Exception as e:
        return f"❌ 识别失败: {str(e)}", None, ""


# ============================================================
# 历史查询
# ============================================================

def query_history(limit, category, date_range):
    """
    查询历史记录（真实数据库查询）
    """
    try:
        limit = int(limit) if limit else 20
    except (TypeError, ValueError):
        limit = 20

    # 类别筛选：中文 -> 英文
    category_en = None
    if category and category != "全部":
        try:
            idx = CLASS_NAMES_CN.index(category)
            category_en = CLASS_NAMES_EN[idx]
        except ValueError:
            category_en = None

    # 日期范围处理
    start_date = None
    end_date = None
    today = datetime.now().date()

    if date_range == "今天":
        start_date = today.strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
    elif date_range == "最近7天":
        start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
    elif date_range == "最近30天":
        start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

    try:
        result = db.get_paginated_records(
            page=1,
            page_size=limit,
            category=category_en,
            start_date=start_date,
            end_date=end_date
        )
        records = result['records']
    except Exception as e:
        return [[f"查询失败: {e}", "", "", ""]]

    if not records:
        return [["暂无记录", "", "", ""]]

    data = []
    for r in records:
        data.append([
            r['id'],
            r['created_at'],
            r['predicted_class_cn'],
            f"{r['confidence']:.2%}"
        ])
    return data


# ============================================================
# 统计信息
# ============================================================

def refresh_stats():
    """刷新统计信息（真实数据库查询）"""
    try:
        stats = db.get_category_stats()
    except Exception as e:
        return f"❌ 统计查询失败: {e}"

    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    lines = []
    for cls_en, count in sorted_stats:
        try:
            idx = CLASS_NAMES_EN.index(cls_en)
            cls_cn = CLASS_NAMES_CN[idx]
        except ValueError:
            cls_cn = cls_en
        lines.append(f"• {cls_cn}: {count} 条")
    return "\n".join(lines) if lines else "暂无统计数据"


# ============================================================
# 删除记录
# ============================================================

def delete_only(record_id):
    """删除单条记录（真实数据库操作）"""
    if not record_id:
        return "⚠️ 请输入要删除的记录ID"
    try:
        record_id = int(record_id)
        success = db.delete_record(record_id)
        if success:
            return f"✅ 记录 ID={record_id} 已删除"
        else:
            return f"❌ 删除失败，ID={record_id} 不存在"
    except ValueError:
        return "⚠️ 请输入有效的数字ID"
    except Exception as e:
        return f"❌ 删除异常: {e}"


# ============================================================
# 点击表格行自动填入删除ID
# ============================================================

def on_table_select(table_value, evt: gr.SelectData):
    """提取点击行的ID，返回给删除ID输入框"""
    try:
        if not evt or not hasattr(evt, "index"):
            return None

        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index

        # 情况1：pandas DataFrame（interactive=False 时的默认格式）
        try:
            import pandas as pd
            if isinstance(table_value, pd.DataFrame):
                if 0 <= row_index < len(table_value):
                    val = table_value.iloc[row_index, 0]
                    return int(val) if val is not None else None
        except ImportError:
            pass

        # 情况2：list of lists（备选）
        if isinstance(table_value, (list, tuple)) and 0 <= row_index < len(table_value):
            row = table_value[row_index]
            if isinstance(row, (list, tuple)) and len(row) > 0:
                return int(row[0])

        return None
    except Exception:
        return None


# ============================================================
# 清除输入
# ============================================================

def clear_input():
    """清除所有输入和输出"""
    return None, None, None, "", ""


# ============================================================
# Gradio 界面
# ============================================================

custom_css = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
h1 {
    text-align: center;
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}
"""

with gr.Blocks(title="皮肤病变智能识别系统", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 基于深度学习的皮肤病变智能识别系统
    ### 支持7类皮肤病变识别：黑素细胞痣 | 黑色素瘤 | 良性角化病 | 基底细胞癌 | 日光性角化病 | 血管病变 | 皮肤纤维瘤
    """)

    # ========== 主区域：两列布局 ==========
    with gr.Row(equal_height=False):
        # -------- 左列：输入区域 --------
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📤 上传图片")
            input_image = gr.Image(
                label="拖拽或点击上传皮肤病变图片",
                type="numpy",
                height=300,
                sources=["upload", "webcam"]
            )
            gr.Markdown("**📷 示例图片**: 可使用 data 文件夹中的图片测试")

            with gr.Row():
                recognize_btn = gr.Button("🔍 开始识别", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ 清除", variant="secondary")

            status_text = gr.Markdown("")

        # -------- 右列：输出区域 --------
        with gr.Column(scale=1.5, variant="panel"):
            gr.Markdown("### 📊 识别结果")
            result_output = gr.Markdown("👆 请上传图片并点击「开始识别」")
            plot_output = gr.Plot(label="概率分布")
            tip_output = gr.Markdown("")

    # ========== 历史记录区域 ==========
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 识别历史记录")

            with gr.Row():
                history_limit = gr.Number(
                    label="显示条数",
                    value=20,
                    minimum=1,
                    maximum=100,
                    step=1
                )
                history_category = gr.Dropdown(
                    label="按类别筛选",
                    choices=["全部"] + CLASS_NAMES_CN,
                    value="全部"
                )
                history_date = gr.Dropdown(
                    label="时间范围",
                    choices=["全部", "今天", "最近7天", "最近30天"],
                    value="全部"
                )
                query_btn = gr.Button("🔍 查询", variant="primary")

            history_table = gr.Dataframe(
                headers=["ID", "识别时间", "预测类别", "置信度"],
                label="历史记录（点击行自动填入ID）",
                interactive=False,
                wrap=True
            )

            with gr.Row():
                stats_output = gr.Markdown("**📊 各类别统计**: 点击「刷新统计」查看")
                refresh_stats_btn = gr.Button("🔄 刷新统计", size="sm", variant="secondary")

            with gr.Row():
                delete_id_input = gr.Number(
                    label="删除记录ID（点击表格行自动填入）",
                    value=None,
                    minimum=1
                )
                delete_btn = gr.Button("🗑️ 删除记录", variant="stop", size="sm")
                delete_result = gr.Markdown("")

    # ============================================================
    # 事件绑定
    # ============================================================

    # 1. 识别按钮
    recognize_btn.click(
        fn=recognize_image,
        inputs=[input_image],
        outputs=[result_output, plot_output, tip_output]
    ).then(
        fn=lambda: "✅ 识别完成",
        inputs=None,
        outputs=[status_text]
    )

    # 2. 清除按钮
    clear_btn.click(
        fn=clear_input,
        inputs=None,
        outputs=[input_image, result_output, plot_output, tip_output, status_text]
    )

    # 3. 查询历史
    query_btn.click(
        fn=query_history,
        inputs=[history_limit, history_category, history_date],
        outputs=[history_table]
    )

    # 4. 刷新统计
    refresh_stats_btn.click(
        fn=refresh_stats,
        inputs=None,
        outputs=[stats_output]
    )

    # 5. 删除记录（删除后自动刷新历史和统计）
    delete_btn.click(
        fn=delete_only,
        inputs=[delete_id_input],
        outputs=[delete_result]
    ).then(
        fn=query_history,
        inputs=[history_limit, history_category, history_date],
        outputs=[history_table]
    ).then(
        fn=refresh_stats,
        inputs=None,
        outputs=[stats_output]
    )

    # 6. 表格行点击 -> 填入删除ID
    history_table.select(
        fn=on_table_select,
        inputs=[history_table],
        outputs=[delete_id_input]
    )

    # 7. 页面加载时自动加载数据和统计
    demo.load(
        fn=query_history,
        inputs=[history_limit, history_category, history_date],
        outputs=[history_table]
    )
    demo.load(
        fn=refresh_stats,
        inputs=None,
        outputs=[stats_output]
    )


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)

    # 初始化数据库表
    try:
        db.init_table()
        print("✅ 数据库表已就绪")
    except Exception as e:
        print(f"⚠️ 数据库初始化警告: {e}")

    # 启动服务
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )