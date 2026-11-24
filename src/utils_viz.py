# -*- coding: utf-8 -*-
"""
utils_viz.py —— 全项目统一图表风格

要点（按你的偏好）：
1. 中文字体优先 Microsoft YaHei，回退 SimHei，避免方块字；
2. 负号正常显示；
3. 去掉上/右边框，网格浅灰且置于图层下方；
4. 保存时 bbox_inches="tight"，不留多余白边；
5. dpi 160，足够放进报告和 PPT。
"""

import matplotlib
matplotlib.use("Agg")           # 只出图不弹窗，脚本才能后台跑
import matplotlib.pyplot as plt
from matplotlib import font_manager


# 语义色（后面各步骤共用，别每张图换一套颜色）
VALUE_COLORS = {
    "低价值": "#9DB4D6",
    "中价值": "#5B8FF9",
    "高价值": "#1D3A6E",
}
BEHAVIOR_COLORS = {
    "全响应型": "#4CC38A",
    "选择性响应型": "#5B8FF9",
    "零完成型": "#F08C4A",
    "低活跃型": "#A8A8A8",
}
SEGMENT_CMAP = "Blues"


def use_style():
    """设置全局图表风格，返回实际使用的中文字体名"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans CN"]:
        if cand in available:
            chosen = cand
            break
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "font.size": 10.5,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.dpi": 120,
    })
    for spine in ["top", "right"]:
        plt.rcParams[f"axes.spines.{spine}"] = False
    return chosen


def save_fig(fig, path):
    """统一保存：紧凑布局、去白边、白底"""
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
